import math
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image, ImageFilter
from transformers import pipeline, AutoTokenizer
from huggingface_hub import hf_hub_download
from transformers import pipeline
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor

detectors = {}

def get_attribution_attention(img, inputs, outputs, processor, target_start, target_end, layer_idx=-1, head_idx="avg"):
    tok = processor.tokenizer
    ids = inputs["input_ids"][0]

    # image token id + positions in the full seq (len=126)
    image_token_id = getattr(tok, "image_token_id", None) or getattr(tok, "img_token_id", None)
    if image_token_id is None:
        image_token_id = tok.convert_tokens_to_ids("<image>") or tok.convert_tokens_to_ids('<|image_pad|>') or tok.convert_tokens_to_ids("<image_soft_token>")
    image_pos = (ids == image_token_id).nonzero(as_tuple=False).squeeze(-1)
    if image_pos.numel() == 0:
        raise RuntimeError(f"No {tok.convert_ids_to_tokens(image_token_id)} tokens found in input_ids.")

    # attn: (1, H, S, S) -> (H, S, S)
    attn = outputs.attentions[layer_idx][0]
    qs = slice(int(target_start), int(target_end))

    # aggregate over query span and (optionally) heads -> (S,)
    if head_idx == "avg":
        a = attn[:, qs, :].mean(dim=1).mean(dim=0)          # (S,)
    else:
        a = attn[int(head_idx), qs, :].mean(dim=0)          # (S,)

    # keep only image-token keys -> (N_img,)
    img_attn = a.index_select(0, image_pos.to(a.device)).float()
    n = img_attn.numel()

    # reshape into square grid (optionally skipping 1 special token)
    off = 0
    s = int(math.sqrt(n))
    if s * s != n:
        s2 = int(math.sqrt(n - 1))
        if s2 * s2 != n - 1:
            raise RuntimeError(f"Can't make square grid from {n} image tokens (need n or n-1 to be a square).")
        off, s = 1, s2
    gh = gw = s
    patch_map = img_attn[off:off + gh * gw].view(gh, gw)

    # upsample to the ORIGINAL input image resolution (e.g. 224x224)
    W, H = img.size
    heatmap = F.interpolate(patch_map[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap.detach().cpu()

def get_attribution_integrated_gradients(
    img, inputs, model, processor,
    target_start, target_end,
    steps=24, baseline="zero",
):
    """
    IG w.r.t. inputs["pixel_values"] in your current teacher-forced setting.
    Returns a (H,W) heatmap on CPU in [0,1], upsampled to img.size.
    """
    model.eval()
    fwd = {k: v for k, v in inputs.items() if k != "pixel_values"}  # keep image_grid_thw, etc.

    ids = fwd["input_ids"]
    ts, te = max(int(target_start), 1), int(target_end)
    pos = torch.arange(ts - 1, te - 1, device=ids.device)
    tgt = ids[:, ts:te]

    x = inputs["pixel_values"].detach().float()
    if baseline == "zero":
        b = torch.zeros_like(x)
    elif baseline == "mean":
        if x.dim() == 4:
            b = x.mean(dim=(2,3), keepdim=True).expand_as(x)
        else:
            b = x.mean().expand_as(x)
    d = x - b

    # alphas = torch.linspace(0, 1, steps + 1, device=x.device, dtype=x.dtype)  # m intervals => m+1 points
    alphas = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)  # m intervals => m+1 points
    total = torch.zeros_like(x)

    for i, a in enumerate(alphas):
        pv = (b + a * d).detach().requires_grad_(True)

        out = model(**fwd, pixel_values=pv, use_cache=False)
        logp = F.log_softmax(out.logits.index_select(1, pos), dim=-1)
        obj = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()

        grad = torch.autograd.grad(obj, pv, retain_graph=False, create_graph=False)[0]

        # w = 0.5 if (i == 0 or i == len(alphas) - 1) else 1.0  # trapezoid weights
        w = 1.0
        total += w * grad

    avg_grad = total / float(steps)   # divide by number of intervals
    ig = d * avg_grad

    if ig.dim() == 4:
        ig = ig.permute(0, 2, 3, 1)[0]
    else:
        if ig.shape[0] == 1:
            ig = ig[0]
        assert ig.dim() == 2
        h_tmp = math.isqrt(img.size[1] * ig.shape[0] // img.size[0])
        w_tmp = ig.shape[0] // h_tmp
        ig = ig.reshape(h_tmp, w_tmp, -1)
    patch = ig.abs().sum(dim=-1)

    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm.detach().cpu()

def get_attribution_wasserstein_ig(
    img, inputs, model, processor,
    target_start, target_end,
    steps=16, baseline="zero",
    topk=256, max_subset=512,
    cost_metric="cosine",
    sinkhorn_eps=0.05, sinkhorn_iters=30,
):
    model.eval()
    fwd = {k: v for k, v in inputs.items() if k != "pixel_values"}  # keep image_grid_thw, etc.
    ids = fwd["input_ids"]
    x = inputs["pixel_values"].detach().float()
    if baseline == "zero":
        b = torch.zeros_like(x)
    elif baseline == "mean":
        if x.dim() == 4:
            b = x.mean(dim=(2,3), keepdim=True).expand_as(x)
        else:
            b = x.mean().expand_as(x)
    d = x - b

    ts, te = max(int(target_start), 1), int(target_end)
    pos = torch.arange(ts - 1, te - 1, device=ids.device)  # logits[t-1] predicts token t

    def logits_at(pv):
        out = model(**fwd, pixel_values=pv, use_cache=False)
        return out.logits.index_select(1, pos).float()  # (1,L,V)

    def subset_probs(logits, token_ids):
        logZ = torch.logsumexp(logits, dim=-1, keepdim=True)     # (1,L,1)
        p = torch.exp(logits[:, :, token_ids] - logZ).mean(1)[0] # (K,)
        return p / p.sum().clamp_min(1e-12)

    def sinkhorn(p, q, C):
        p = (p + 1e-12) / p.sum().clamp_min(1e-12)
        q = (q + 1e-12) / q.sum().clamp_min(1e-12)
        lp, lq, lK = p.log(), q.log(), (-C / sinkhorn_eps)
        lu, lv = lp*0, lq*0
        for _ in range(sinkhorn_iters):
            lu = lp - torch.logsumexp(lK + lv[None, :], dim=1)
            lv = lq - torch.logsumexp(lK.t() + lu[None, :], dim=1)
        P = torch.exp(lu[:, None] + lK + lv[None, :])
        return (P * C).sum()

    # fixed token subset + cost + reference distribution (once)
    with torch.no_grad():
        lo, lb = logits_at(x), logits_at(b)
        so, sb = lo[0].amax(0), lb[0].amax(0)  # (V,)
        to = torch.topk(so, k=min(topk, so.numel())).indices
        tb = torch.topk(sb, k=min(topk, sb.numel())).indices
        token_ids = torch.unique(torch.cat([to, tb])).to(ids.device)

        if token_ids.numel() > max_subset:
            s = torch.maximum(so[token_ids], sb[token_ids])
            token_ids = token_ids[torch.topk(s, k=max_subset).indices]

        emb = model.get_input_embeddings().weight[token_ids].float()
        if cost_metric == "cosine":
            emb = F.normalize(emb, dim=-1)
            C = (1.0 - emb @ emb.t()).clamp_min(0.0)
        else:
            C = torch.cdist(emb, emb, p=2)

        p_ref = subset_probs(lo, token_ids).detach()

    # IG
    # alphas = torch.linspace(0, 1, steps + 1, device=x.device, dtype=x.dtype)  # m intervals => m+1 points
    alphas = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)  # m intervals => m+1 points
    total = torch.zeros_like(x)

    for i, a in enumerate(alphas):
        pv = (b + a * d).detach().requires_grad_(True)

        q = subset_probs(logits_at(pv), token_ids)
        obj = -sinkhorn(p_ref, q, C)
        # g += torch.autograd.grad(obj, pv, retain_graph=False, create_graph=False)[0]
        grad = torch.autograd.grad(obj, pv, retain_graph=False, create_graph=False)[0]

        # w = 0.5 if (i == 0 or i == len(alphas) - 1) else 1.0  # trapezoid weights
        w = 1.0  # trapezoid weights
        total += w * grad

    # ig = d * (g / float(steps))

    avg_grad = total / float(steps)   # divide by number of intervals
    ig = d * avg_grad

    if ig.dim() == 4:
        # N = ig.size(0) * ig.size(2) * ig.size(3)
        ig = ig.permute(0, 2, 3, 1)[0]
    else:
        if ig.shape[0] == 1:
            ig = ig[0]
        assert ig.dim() == 2
        h_tmp = math.isqrt(img.size[1] * ig.shape[0] // img.size[0])
        w_tmp = ig.shape[0] // h_tmp
        ig = ig.reshape(h_tmp, w_tmp, -1)
    patch = ig.abs().sum(dim=-1)

    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm.detach().cpu()

def get_attribution_igospp(
    img, inputs, model, processor,
    target_start, target_end,
    iters=20, ig_steps=8, step_size=1.0,
    baseline="mean", mask_res=None,
    l1_weight=0.05, tv_weight=0.2, clamp_logits=8.0,
):
    model.eval()
    fwd = {k: v for k, v in inputs.items() if k != "pixel_values"}  # keep image_grid_thw, etc.
    ids = fwd["input_ids"]
    x = inputs["pixel_values"].detach().float()
    if x.dim() == 2: x = x.unsqueeze(0)               # (1,N,D)
    B, N, D = x.shape
    dev = x.device

    # baseline in feature space
    b = torch.zeros_like(x) if baseline == "zero" else x.mean(dim=1, keepdim=True).expand_as(x)
    ts, te = max(int(target_start), 1), int(target_end)
    pos = torch.arange(ts - 1, te - 1, device=dev)
    tgt = ids[:, ts:te]

    # span score: sum log p(token_t | prefix) over [ts,te)
    def score(pv):
        out = model(**fwd, pixel_values=pv, use_cache=False)
        logp = F.log_softmax(out.logits.index_select(1, pos), dim=-1)
        return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()

    # grid for N patches (allow 1 special token offset)
    off, s = 0, int(math.isqrt(N))
    if s * s != N:
        s2 = int(math.isqrt(N - 1))
        if s2 * s2 != N - 1: raise RuntimeError(f"Can't grid {N} tokens (need N or N-1 square).")
        off, s = 1, s2
    r = s if mask_res is None else int(mask_res)

    def upmask(logits):  # (1,1,r,r)->(1,N,1) in [0,1]
        m = torch.sigmoid(F.interpolate(logits, size=(s, s), mode="bilinear", align_corners=False)).view(B, s*s, 1)
        return torch.cat([torch.zeros(B, 1, 1, device=dev), m], dim=1) if off else m

    def tv(mgrid):  # mgrid: (B,1,s,s)
        return (mgrid[..., 1:, :] - mgrid[..., :-1, :]).abs().mean() + (mgrid[..., :, 1:] - mgrid[..., :, :-1]).abs().mean()

    # learnable logits (deletion + insertion)
    md = torch.zeros((B, 1, r, r), device=dev, requires_grad=True)
    mi = torch.zeros((B, 1, r, r), device=dev, requires_grad=True)

    alphas = torch.linspace(0, 1, ig_steps, device=dev)

    def ig_dir(logits, end_fn, sign):  # IG gradient wrt logits along baseline->end_fn(mask)
        acc = 0.0
        for a in alphas:
            m = upmask(logits)
            pv_end = end_fn(m)
            pv_a = b + a * (pv_end - b)
            acc = acc + torch.autograd.grad(score(pv_a), logits, retain_graph=False, create_graph=False)[0]
        return sign * (acc / float(ig_steps))

    for _ in range(iters):
        mdf, mif = upmask(md), upmask(mi)
        mfin = (mdf * mif).clamp(0, 1)                                # (B,N,1)
        mgrid = mfin[:, off:, 0].view(B, 1, s, s)                     # ignore optional special token in regs
        reg = l1_weight * mgrid.mean() + tv_weight * tv(mgrid)
        gd_reg, gi_reg = torch.autograd.grad(reg, [md, mi], retain_graph=False, create_graph=False)

        pv_del = lambda m: x * (1 - m) + b * m
        pv_ins = lambda m: x * m + b * (1 - m)

        gd = ig_dir(md, lambda m: pv_del(m), +1.0) + gd_reg           # minimize score after deletion
        gi = ig_dir(mi, lambda m: pv_ins(m), -1.0) + gi_reg           # maximize score after insertion

        with torch.no_grad():
            md -= step_size * gd
            mi -= step_size * gi
            if clamp_logits is not None:
                md.clamp_(-clamp_logits, clamp_logits)
                mi.clamp_(-clamp_logits, clamp_logits)

    with torch.no_grad():
        mfin = (upmask(md) * upmask(mi)).clamp(0, 1)[:, off:, 0].view(s, s)  # (s,s)

    W, H = img.size
    hm = F.interpolate(mfin[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm.detach().cpu()

def get_attribution_igospp_wasserstein(
    img, inputs, model, processor,
    target_start, target_end,
    iters=10, ig_steps=4, step_size=1.0,
    baseline="mean", mask_res=None,              # mask_res=None => use full patch grid s
    topk=128, max_subset=256, cost_metric="cosine",
    sinkhorn_eps=0.05, sinkhorn_iters=20,
    l1_weight=0.05, tv_weight=0.2, clamp_logits=8.0,
):
    model.eval()
    fwd = {k: v for k, v in inputs.items() if k != "pixel_values"}
    ids = fwd["input_ids"]; dev = ids.device

    x = inputs["pixel_values"].detach().float()
    if x.dim() == 2: x = x.unsqueeze(0)                     # (1,N,D)
    B, N, D = x.shape
    b = torch.zeros_like(x) if baseline == "zero" else x.mean(1, keepdim=True).expand_as(x)

    ts, te = max(int(target_start), 1), int(target_end)
    pos = torch.arange(ts - 1, te - 1, device=dev)

    # patch grid (allow optional 1-token offset)
    off, s = 0, int(math.isqrt(N))
    if s * s != N:
        s2 = int(math.isqrt(N - 1))
        if s2 * s2 != N - 1: raise RuntimeError(f"Can't grid {N} patches (need N or N-1 square).")
        off, s = 1, s2

    r = s if mask_res is None else int(mask_res)

    def logits_at(pv):
        return model(**fwd, pixel_values=pv, use_cache=False).logits.index_select(1, pos).float()  # (B,L,V)

    def probs_sub(logits, tids):
        logZ = torch.logsumexp(logits, -1, keepdim=True)                      # (B,L,1)
        p = torch.exp(logits[:, :, tids] - logZ).mean(1)[0]                   # (K,)
        return p / p.sum().clamp_min(1e-12)

    def sinkhorn(p, q, C):
        p = (p + 1e-12) / p.sum().clamp_min(1e-12)
        q = (q + 1e-12) / q.sum().clamp_min(1e-12)
        lp, lq, lK = p.log(), q.log(), (-C / sinkhorn_eps)
        lu, lv = lp*0, lq*0
        for _ in range(sinkhorn_iters):
            lu = lp - torch.logsumexp(lK + lv[None, :], 1)
            lv = lq - torch.logsumexp(lK.t() + lu[None, :], 1)
        P = torch.exp(lu[:, None] + lK + lv[None, :])
        return (P * C).sum()

    with torch.no_grad():
        lo, lb = logits_at(x), logits_at(b)
        so, sb = lo[0].amax(0), lb[0].amax(0)
        tids = torch.unique(torch.cat([torch.topk(so, min(topk, so.numel())).indices,
                                       torch.topk(sb, min(topk, sb.numel())).indices])).to(dev)
        if tids.numel() > max_subset:
            keep = torch.topk(torch.maximum(so[tids], sb[tids]), max_subset).indices
            tids = tids[keep]
        E = model.get_input_embeddings().weight[tids].float()
        if cost_metric == "cosine":
            En = F.normalize(E, dim=-1)
            C = (1 - En @ En.t()).clamp_min(0.0)
        else:
            C = torch.cdist(E, E, p=2)
        p_ref = probs_sub(lo, tids).detach()

    Wscore = lambda pv: sinkhorn(p_ref, probs_sub(logits_at(pv), tids), C)

    # mask logits (B,1,r,r); if r==s, this is "full patch resolution"
    md = torch.zeros((B, 1, r, r), device=dev, requires_grad=True)
    mi = torch.zeros((B, 1, r, r), device=dev, requires_grad=True)
    alphas = torch.linspace(0, 1, ig_steps, device=dev)

    def upmask(L):  # (B,1,r,r)->(B,N,1)
        m = torch.sigmoid(F.interpolate(L, size=(s, s), mode="bilinear", align_corners=False)).view(B, s*s, 1)
        return torch.cat([torch.zeros(B, 1, 1, device=dev), m], 1) if off else m

    def tv(mg):  # (B,1,s,s)
        return (mg[..., 1:, :] - mg[..., :-1, :]).abs().mean() + (mg[..., :, 1:] - mg[..., :, :-1]).abs().mean()

    pv_del = lambda m: x * (1 - m) + b * m
    pv_ins = lambda m: x * m + b * (1 - m)

    def ig_dir(L, end_fn, sign):
        acc = 0.0
        for a in alphas:
            m = upmask(L)
            pv_a = b + a * (end_fn(m) - b)
            acc = acc + torch.autograd.grad(Wscore(pv_a), L, retain_graph=False, create_graph=False)[0]
        return sign * (acc / float(ig_steps))

    for _ in range(iters):
        mfin = (upmask(md) * upmask(mi)).clamp(0, 1)
        mg = mfin[:, off:, 0].view(B, 1, s, s)
        reg = l1_weight * mg.mean() + tv_weight * tv(mg)
        gd_reg, gi_reg = torch.autograd.grad(reg, [md, mi], retain_graph=False, create_graph=False)

        gd = ig_dir(md, pv_del, -1.0) + gd_reg   # deletion: maximize W
        gi = ig_dir(mi, pv_ins, +1.0) + gi_reg   # insertion: minimize W

        with torch.no_grad():
            md -= step_size * gd
            mi -= step_size * gi
            if clamp_logits is not None:
                md.clamp_(-clamp_logits, clamp_logits)
                mi.clamp_(-clamp_logits, clamp_logits)

    with torch.no_grad():
        m = (upmask(md) * upmask(mi)).clamp(0, 1)[:, off:, 0].view(s, s)

    W0, H0 = img.size
    hm = F.interpolate(m[None, None], size=(H0, W0), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm.detach().cpu()

def get_attribution_gradcam(img, inputs, model, processor, target_start, target_end, layer_idx=-1):
    tok, ids = processor.tokenizer, inputs["input_ids"]

    # image token id + positions in full seq
    itok = getattr(tok, "image_token_id", None) or getattr(tok, "img_token_id", None)
    if itok is None:
        for s in ("<|image_pad|>", "<image>", "<image_soft_token>", "<|vision_start|>"):
            t = tok.convert_tokens_to_ids(s)
            if t is not None and t != tok.unk_token_id: itok = t; break
    img_pos = (ids[0] == itok).nonzero(as_tuple=False).squeeze(-1)
    if img_pos.numel() == 0: raise RuntimeError("No image tokens found in input_ids.")

    # forward with pixel_values requiring grad so hidden states are grad-tracked
    pv = inputs["pixel_values"].detach().float().requires_grad_(True)
    fwd = dict(inputs)
    fwd["pixel_values"] = pv

    model.eval()
    out = model(**fwd, use_cache=False, return_dict=True, output_hidden_states=True)

    L = len(out.hidden_states) - 1  # decoder layers; hidden_states[0] is embeddings
    li = (layer_idx % L) + 1        # map to hidden_states index
    A = out.hidden_states[li]       # (1,S,D)

    ts, te = max(int(target_start), 1), int(target_end)
    tgt = ids[:, ts:te]
    logp = F.log_softmax(out.logits[:, ts-1:te-1, :], dim=-1)
    obj = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()

    G = torch.autograd.grad(obj, A, retain_graph=False, create_graph=False)[0]  # (1,S,D)

    # Grad-CAM on image-token positions
    Ai, Gi = A[0, img_pos, :], G[0, img_pos, :]          # (N,D)
    w = Gi.mean(0)                                       # (D,)
    t = F.relu((Ai * w).sum(-1))                         # (N,)

    # square grid (allow 1 special token)
    n = int(t.numel())
    s = int(math.isqrt(n)); off = 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1: raise RuntimeError(f"Can't reshape {n} image tokens into square (n or n-1 must be square).")
        off, s = 1, s2
    patch = t[off:off + s*s].view(s, s)

    # upsample to original image size
    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm.detach().cpu().float()

def get_attribution_grad_eclip(
    img, inputs, model, processor, target_start, target_end,
    vision_layer_idx=-1, autocast_dtype=torch.bfloat16
):
    model.eval()
    pv = inputs["pixel_values"].detach().float().requires_grad_(True)
    fwd = dict(inputs); fwd["pixel_values"] = pv

    blk = model.model.visual.blocks[vision_layer_idx]
    cache = {}

    h1 = blk.attn.qkv.register_forward_hook(lambda m, a, o: cache.__setitem__("qkv", o))
    h2 = blk.attn.register_forward_hook(lambda m, a, o: cache.__setitem__("o", o[0] if isinstance(o,(tuple,list)) else o))

    with torch.enable_grad(), torch.cuda.amp.autocast(enabled=pv.is_cuda, dtype=autocast_dtype):
        out = model(**fwd, use_cache=False, return_dict=True)

        ids = fwd["input_ids"]
        ts, te = max(int(target_start), 1), int(target_end)
        tgt = ids[:, ts:te]
        logp = F.log_softmax(out.logits[:, ts-1:te-1, :], dim=-1)
        score = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()

        qkv, o = cache["qkv"], cache["o"]
        go = torch.autograd.grad(score, o, retain_graph=True, create_graph=False)[0]

    h1.remove(); h2.remove()

    if qkv.dim() == 2: qkv = qkv.unsqueeze(0)   # (S,3D)->(1,S,3D)
    if o.dim() == 2:   o   = o.unsqueeze(0)     # (S,D)->(1,S,D)
    if go.dim() == 2:  go  = go.unsqueeze(0)

    D = qkv.shape[-1] // 3
    q, k, v = qkv[..., :D].float(), qkv[..., D:2*D].float(), qkv[..., 2*D:].float()
    o, go = o.float(), go.float()

    S = v.shape[1]
    s = int(math.isqrt(S)); has_cls = (s*s != S and int(math.isqrt(S-1))**2 == S-1)
    if has_cls:
        qg, wc = q[:, 0, :], go[:, 0, :]
        k, v = k[:, 1:, :], v[:, 1:, :]
        S = S - 1
        s = int(math.isqrt(S))
    else:
        qg, wc = q.mean(1), go.mean(1)
        k, v = k[:, :S, :], v[:, :S, :]

    wi = (k * qg[:, None, :]).sum(-1) / math.sqrt(D)             # (1,S)
    wi = (wi - wi.amin(1, keepdim=True)) / (wi.amax(1, keepdim=True) - wi.amin(1, keepdim=True) + 1e-8)

    cam = F.relu((v * wc[:, None, :]).sum(-1) * wi)[0].view(s, s) # (s,s)

    W, H = img.size
    hm = F.interpolate(cam[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm.detach().cpu()


def get_attribution_gradcampp(
    img, inputs, model, processor, target_start, target_end,
    layer_idx: int = -1, eps: float = 1e-8
):
    """
    Grad-CAM++ attribution over image-token positions using decoder hidden states
    (treat embedding dim as channels; image tokens as spatial locations).
    Returns a (H,W) heatmap normalized to [0,1] on CPU.
    """
    tok, ids = processor.tokenizer, inputs["input_ids"]

    # image token id + positions in full seq
    itok = getattr(tok, "image_token_id", None) or getattr(tok, "img_token_id", None)
    if itok is None:
        for s in ("<|image_pad|>", "<image>", "<image_soft_token>", "<|vision_start|>"):
            t = tok.convert_tokens_to_ids(s)
            if t is not None and t != tok.unk_token_id:
                itok = t
                break
    img_pos = (ids[0] == itok).nonzero(as_tuple=False).squeeze(-1)
    if img_pos.numel() == 0:
        raise RuntimeError("No image tokens found in input_ids.")

    # forward with pixel_values requiring grad so hidden states are grad-tracked
    pv = inputs["pixel_values"].detach().float().requires_grad_(True)
    fwd = dict(inputs)
    fwd["pixel_values"] = pv

    model.eval()
    out = model(**fwd, use_cache=False, return_dict=True, output_hidden_states=True)

    L = len(out.hidden_states) - 1  # hidden_states[0] is embeddings
    li = (layer_idx % L) + 1
    A = out.hidden_states[li]       # (1,S,D)

    # objective: sum log-prob of target tokens (same as your Grad-CAM version)
    ts, te = max(int(target_start), 1), int(target_end)
    tgt = ids[:, ts:te]
    logp = F.log_softmax(out.logits[:, ts-1:te-1, :], dim=-1)
    obj = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()

    # first-order grads w.r.t. activations
    G = torch.autograd.grad(obj, A, retain_graph=False, create_graph=False)[0]  # (1,S,D)

    # Grad-CAM++ on image-token positions
    Ai = A[0, img_pos, :].float()   # (N,D)
    Gi = G[0, img_pos, :].float()   # (N,D)

    relu_g = F.relu(Gi)            # (N,D)

    # Grad-CAM++ weights (token-adapted):
    # alpha_{i,d} = g^2 / (2 g^2 + (sum_j A_{j,d}) g^3)
    g2 = Gi * Gi
    g3 = g2 * Gi
    sumA = Ai.sum(dim=0, keepdim=True)                  # (1,D)
    denom = 2.0 * g2 + sumA * g3                         # (N,D)
    alpha = g2 / (denom + eps)                           # (N,D)

    # w_d = sum_i alpha_{i,d} * relu(g_{i,d})
    w = (alpha * relu_g).sum(dim=0)                      # (D,)

    # token scores and reshape to square grid (allow 1 special token)
    t = F.relu((Ai * w).sum(dim=-1))                     # (N,)
    n = int(t.numel())
    s = int(math.isqrt(n)); off = 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1:
            raise RuntimeError(f"Can't reshape {n} image tokens into square (n or n-1 must be square).")
        off, s = 1, s2
    patch = t[off:off + s*s].view(s, s)

    # upsample to original image size
    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + eps)
    return hm.detach().cpu().float()

def get_attribution_gradcam_wasserstein(
    img, inputs, model, processor, target_start, target_end,
    layer_idx=-1, baseline="zero",
    topk=256, max_subset=512, cost_metric="cosine",
    sinkhorn_eps=0.05, sinkhorn_iters=20,
):
    tok, ids = processor.tokenizer, inputs["input_ids"]

    itok = getattr(tok, "image_token_id", None) or getattr(tok, "img_token_id", None)
    if itok is None:
        for s in ("<|image_pad|>", "<image>", "<image_soft_token>", "<|vision_start|>"):
            t = tok.convert_tokens_to_ids(s)
            if t is not None and t != tok.unk_token_id: itok = t; break
    img_pos = (ids[0] == itok).nonzero(as_tuple=False).squeeze(-1)
    if img_pos.numel() == 0: raise RuntimeError("No image tokens found in input_ids.")

    ts, te = max(int(target_start), 1), int(target_end)
    pos = slice(ts - 1, te - 1)  # logits positions that predict ids[:, ts:te]

    pv = inputs["pixel_values"].detach().float().requires_grad_(True)
    out = model(**{**inputs, "pixel_values": pv}, use_cache=False, return_dict=True, output_hidden_states=True)

    hs = out.hidden_states[1:]
    A = hs[layer_idx % len(hs)]                     # (1,S,D)

    # baseline pixel_values (same shape as pv, works for 2D/3D/4D/5D)
    with torch.no_grad():
        if baseline == "zero":
            base = torch.zeros_like(pv)
        elif baseline == "mean":
            base = pv.mean(dim=tuple(range(pv.dim()-1)), keepdim=True).expand_as(pv)
        else:
            raise ValueError("baseline must be 'zero' or 'mean'")

        out_b = model(**{**inputs, "pixel_values": base}, use_cache=False, return_dict=True)

    lo = out.logits[:, pos, :]                      # (1,L,V)
    lb = out_b.logits[:, pos, :]                    # (1,L,V)

    # fixed vocab subset S = union(topk(max over positions) from orig/base), capped
    with torch.no_grad():
        so, sb = lo.detach().amax(1)[0], lb.detach().amax(1)[0]  # (V,)
        top_o = so.topk(min(topk, so.numel())).indices
        top_b = sb.topk(min(topk, sb.numel())).indices
        token_ids = torch.unique(torch.cat([top_o, top_b], 0))
        if token_ids.numel() > max_subset:
            keep = torch.maximum(so[token_ids], sb[token_ids]).topk(max_subset).indices
            token_ids = token_ids[keep]

        E = model.get_input_embeddings().weight[token_ids].float()
        if cost_metric == "cosine":
            E = F.normalize(E, dim=-1)
            C = (1.0 - (E @ E.T)).clamp_min(0.0)
        elif cost_metric == "l2":
            C = torch.cdist(E, E, p=2)
        else:
            raise ValueError("cost_metric must be 'cosine' or 'l2'")
        C = C.detach()

    def probs_mean_on_subset(logits):
        logits = logits.float()
        logZ = torch.logsumexp(logits, dim=-1, keepdim=True)          # (1,L,1)
        p = torch.exp(logits[:, :, token_ids] - logZ)                # (1,L,K) exact probs on subset
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)             # conditional on subset, per position
        p = p.mean(1)[0]                                             # (K,)
        return p / p.sum().clamp_min(1e-12)

    q = probs_mean_on_subset(lo)                                     # (K,) (has grad)
    p = probs_mean_on_subset(lb).detach()                            # (K,) constant

    def sinkhorn(p, q, C):
        p = (p + 1e-12) / p.sum().clamp_min(1e-12)
        q = (q + 1e-12) / q.sum().clamp_min(1e-12)
        log_p, log_q = p.log(), q.log()
        logK = -C / sinkhorn_eps
        log_u = torch.zeros_like(log_p)
        log_v = torch.zeros_like(log_q)
        for _ in range(sinkhorn_iters):
            log_u = log_p - torch.logsumexp(logK + log_v[None, :], dim=1)
            log_v = log_q - torch.logsumexp(logK.T + log_u[None, :], dim=1)
        P = torch.exp(log_u[:, None] + logK + log_v[None, :])
        return (P * C).sum()

    obj = sinkhorn(p, q, C)                                          # scalar distance

    G = torch.autograd.grad(obj, A, retain_graph=False, create_graph=False)[0]  # (1,S,D)

    Ai, Gi = A[0, img_pos, :], G[0, img_pos, :]                      # (N,D)
    w = Gi.mean(0)
    t = F.relu((Ai * w).sum(-1))                                     # (N,)

    n = int(t.numel())
    s = int(math.isqrt(n)); off = 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1: raise RuntimeError(f"Can't reshape {n} image tokens into square (n or n-1).")
        off, s = 1, s2
    patch = t[off:off + s*s].view(s, s)

    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm.detach().cpu().float()


# def _mask_to_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
#     """Convert HxW mask to a tight (x1,y1,x2,y2) box in pixel coords."""
#     if mask.dtype != np.bool_:
#         mask = mask > 0.0
#     ys, xs = np.where(mask)
#     if xs.size == 0 or ys.size == 0:
#         return None
#     x1, x2 = int(xs.min()), int(xs.max())
#     y1, y2 = int(ys.min()), int(ys.max())
#     return (x1, y1, x2 + 1, y2 + 1)

def _mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """mask: (H,W) bool/0-1 -> tight XYXY box (or None if empty)."""
    if mask.dtype != np.bool_:
        mask = mask > 0.0
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2 + 1, y2 + 1)

def _clip_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)

def truncate_tokens(text: str, tok: AutoTokenizer) -> str:
    ids = tok(text, truncation=True, max_length=tok.model_max_length)["input_ids"]
    return tok.decode(ids, skip_special_tokens=True)

@torch.inference_mode()
def sam2_1_boxes_from_text(
    img: Image.Image,
    descriptions: List[str],
    sam2_id: str = "facebook/sam2.1-hiera-large",
    det_id: str = "google/owlvit-base-patch32",
    det_score_threshold: float = 0.10,
    top_k_per_desc: int = 5,
    one_box_per_description: bool = False,
) -> List[Tuple[int, int, int, int]]:
    """
    Returns a list of bounding boxes (x1, y1, x2, y2) in pixel coords.

    Pipeline:
      descriptions -> OWL-ViT candidate boxes -> SAM2.1 masks (box prompt) -> tight boxes
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Text -> candidate boxes (open-vocab detector)
    if det_id in detectors:
        detector = detectors[det_id]
    else:
        detector = pipeline(
            "zero-shot-object-detection",
            model=det_id,
            device=0 if device.type == "cuda" else -1,
        )
        detectors[det_id] = detector

    # SAM2.1
    if sam2_id in detectors:
        processor = detectors[sam2_id]["processor"]
        model = detectors[sam2_id]["model"]
    else:
        from transformers import Sam2Model, Sam2Processor
        processor = Sam2Processor.from_pretrained(sam2_id)
        model = Sam2Model.from_pretrained(sam2_id).to(device).eval()
        detectors[sam2_id] = {
            "processor": processor,
            "model": model,
        }

    W, H = img.size
    all_out: List[Tuple[int, int, int, int]] = []

    for desc in descriptions:
        desc = truncate_tokens(desc, detector.tokenizer)
        dets = detector(img, candidate_labels=[desc], threshold=det_score_threshold)
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)[:top_k_per_desc]
        if not dets:
            continue

        # Build a (batch=1, num_boxes, 4) input_boxes list for SAM2.1
        boxes_xyxy: List[List[float]] = []
        det_scores: List[float] = []
        for d in dets:
            b = d["box"]  # {"xmin","ymin","xmax","ymax"} in pixel coords
            x1 = float(np.clip(b["xmin"], 0, W - 1))
            y1 = float(np.clip(b["ymin"], 0, H - 1))
            x2 = float(np.clip(b["xmax"], 0, W - 1))
            y2 = float(np.clip(b["ymax"], 0, H - 1))
            # ensure proper ordering
            x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
            y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
            boxes_xyxy.append([x1, y1, x2, y2])
            det_scores.append(float(d["score"]))

        inputs = processor(
            images=img,
            input_boxes=[boxes_xyxy],  # batch=1
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)  # outputs.pred_masks, outputs.iou_scores

        # Post-process masks back to original image size.
        # After indexing [0] (batch), expected shape is (num_boxes, num_masks, H, W)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        iou = outputs.iou_scores[0].detach().cpu()  # (num_boxes, num_masks)

        refined: List[Tuple[Tuple[int, int, int, int], float]] = []

        for i in range(masks.shape[0]):  # per input box
            # pick the best of the multiple masks SAM2.1 returns
            best_j = int(torch.argmax(iou[i]).item()) if iou.numel() else 0
            best_mask_np = masks[i, best_j].numpy()

            tight = _mask_to_box(best_mask_np)
            if tight is None:
                # fallback to detector box if SAM gives an empty mask
                x1, y1, x2, y2 = boxes_xyxy[i]
                tight = (int(x1), int(y1), int(x2), int(y2))
                quality = det_scores[i]
            else:
                quality = float(iou[i, best_j].item())

            refined.append((tight, quality))

        if one_box_per_description:
            # keep the highest-quality refined box for this description
            best_box = max(refined, key=lambda t: t[1])[0]
            all_out.append(best_box)
        else:
            all_out.extend([b for (b, _) in refined])

    return all_out

@torch.inference_mode()
def medsam_boxes_from_text(
    img: Image.Image,
    descriptions: List[str],
    # HF MedSAM (SAM v1 ViT-B fine-tuned)
    medsam_id: str = "flaviagiammarino/medsam-vit-base",
    # Text->box step (needed because MedSAM/SAM needs prompts)
    det_id: str = "google/owlvit-base-patch32",
    det_threshold: float = 0.10,
    top_k_per_desc: int = 5,
    # Mask selection/refinement
    multimask_output: bool = True,
    mask_threshold: float = 0.5,
    one_box_per_description: bool = False,
    # Optional: if you *already* have candidate boxes, pass them (one list per description)
    candidate_boxes_by_description: Optional[List[List[Tuple[int, int, int, int]]]] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    descriptions -> candidate boxes -> MedSAM masks -> tight boxes

    Returns a flat list of XYXY boxes. If one_box_per_description=True,
    returns at most 1 box per description.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    W, H = img.size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Text -> candidate boxes (only used when candidate_boxes_by_description is None)
    detector = None
    if candidate_boxes_by_description is None:
        if det_id in detectors:
            detector = detectors[det_id]
        else:
            detector = pipeline(
                "zero-shot-object-detection",
                model=det_id,
                device=0 if device == "cuda" else -1,
            )
            detectors[det_id] = detector
        # detector = pipeline(
        #     task="zero-shot-object-detection",
        #     model=det_id,
        #     device=0 if device == "cuda" else -1,
        # )

    # MedSAM (HF Transformers)
    # processor = SamProcessor.from_pretrained(medsam_id)
    # model = SamModel.from_pretrained(medsam_id).to(device).eval()

    if medsam_id in detectors:
        processor = detectors[medsam_id]["processor"]
        model = detectors[medsam_id]["model"]
    else:
        from transformers import SamModel, SamProcessor
        processor = SamProcessor.from_pretrained(medsam_id)
        model = SamModel.from_pretrained(medsam_id).to(device).eval()
        detectors[medsam_id] = {
            "processor": processor,
            "model": model,
        }

    out: List[Tuple[int, int, int, int]] = []

    for i, text in enumerate(descriptions):
        # --- get candidate boxes ---
        if candidate_boxes_by_description is not None:
            cand = candidate_boxes_by_description[i] if i < len(candidate_boxes_by_description) else []
            det_boxes = [_clip_box(b, W, H) for b in cand]
            det_scores = [1.0] * len(det_boxes)
        else:
            text = truncate_tokens(text, detector.tokenizer)
            dets = detector(img, candidate_labels=[text], threshold=det_threshold)  # type: ignore[misc]
            dets = sorted(dets, key=lambda d: float(d["score"]), reverse=True)[:top_k_per_desc]
            det_boxes = []
            det_scores = []
            for d in dets:
                b = d["box"]  # {"xmin","ymin","xmax","ymax"}
                det_boxes.append(_clip_box((int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"])), W, H))
                det_scores.append(float(d["score"]))

        if not det_boxes:
            continue

        # --- run MedSAM for all boxes in one forward pass ---
        boxes_f = [[float(x1), float(y1), float(x2), float(y2)] for (x1, y1, x2, y2) in det_boxes]
        inputs = processor(
            images=img,
            input_boxes=[boxes_f],  # batch=1, num_boxes=len(boxes_f)
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs, multimask_output=multimask_output)

        # Post-process to original image size (returns list over batch)
        # Common shape: (num_boxes, num_masks, H, W)
        masks_batched = processor.image_processor.post_process_masks(
            outputs.pred_masks.sigmoid().detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
            binarize=False,
        )[0]

        iou = outputs.iou_scores[0].detach().cpu()  # (num_boxes, num_masks) if multimask_output else (num_boxes, 1)

        refined: List[Tuple[Tuple[int, int, int, int], float]] = []
        for bi in range(masks_batched.shape[0]):
            # pick best mask for this box
            if iou.ndim == 2 and iou.shape[1] > 0:
                best_m = int(torch.argmax(iou[bi]).item())
                quality = float(iou[bi, best_m].item())
            else:
                best_m = 0
                quality = det_scores[bi] if bi < len(det_scores) else 0.0

            mask = masks_batched[bi, best_m].numpy() > mask_threshold
            tight = _mask_to_box(mask) or det_boxes[bi]
            tight = _clip_box(tight, W, H)
            refined.append((tight, quality))

        if one_box_per_description:
            out.append(max(refined, key=lambda t: t[1])[0])
        else:
            out.extend([b for (b, _) in refined])

    return out

@dataclass(frozen=True)
class MedSAM2TextBoxConfig:
    # MedSAM2 checkpoint on HF Hub
    medsam2_repo_id: str = "wanglab/MedSAM2"
    medsam2_filename: str = "MedSAM2_latest.pt"

    # Assumption: MedSAM2_latest.pt is typically a SAM2.1 "tiny" sized checkpoint (~156MB),
    # so we default to the SAM2.1 tiny config.
    # If you get state_dict key/shape mismatches, try s/b+/l configs.
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"

    # Text-to-box model (open-vocab). Swap this for a domain-specific detector if needed.
    detector_task: str = "zero-shot-object-detection"
    detector_model: str = "google/owlvit-base-patch32"

    # Thresholds
    det_threshold: float = 0.10
    max_boxes_per_description: int = 1  # set >1 if you want multiple per text

    # Mask refinement
    multimask_output: bool = False  # False gives 1 mask; True gives up to 3 masks


def medsam2_bboxes_from_descriptions(
    img: Image.Image,
    descriptions: List[str],
    cfg: MedSAM2TextBoxConfig = MedSAM2TextBoxConfig(),
    device: Optional[str] = None,
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Returns one list of boxes per description (possibly empty if nothing detected):
        [[(x0,y0,x1,y1), ...],  # for descriptions[0]
         [(x0,y0,x1,y1), ...],  # for descriptions[1]
         ...]

    Pipeline:
      text -> candidate boxes (open-vocab detector) -> MedSAM2 mask refine -> tight bbox
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1) Load detector (text -> boxes)
    # det_device = 0 if device.startswith("cuda") else -1
    # detector = pipeline(cfg.detector_task, model=cfg.detector_model, device=det_device)
    if cfg.detector_model in detectors:
        detector = detectors[cfg.detector_model]
    else:
        detector = pipeline(
            "zero-shot-object-detection",
            model=cfg.detector_model,
            device=0 if device.startswith("cuda") else -1,
        )
        detectors[cfg.detector_model] = detector

    # ---- 2) Load MedSAM2 (SAM2.1 code + MedSAM2 checkpoint from HF)
    ckpt_path = hf_hub_download(repo_id=cfg.medsam2_repo_id, filename=cfg.medsam2_filename)
    # sam2_model = build_sam2(cfg.sam2_model_cfg, ckpt_path=ckpt_path, device=device)
    # predictor = SAM2ImagePredictor(sam2_model)
    if cfg.medsam2_repo_id in detectors:
        predictor = detectors[cfg.medsam2_repo_id]["predictor"]
    else:
        sam2_model = build_sam2(cfg.sam2_model_cfg, ckpt_path=ckpt_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        detectors[cfg.medsam2_repo_id] = {
            "predictor": predictor,
        }

    # Work in RGB
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size

    # Predictor accepts PIL or np.ndarray; we’ll keep PIL for simplicity.
    # (It internally normalizes/resizes.)
    results: List[List[Tuple[int, int, int, int]]] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else torch.autocast(device_type="cpu", dtype=torch.float32)
    )

    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(img_rgb)

        for text in descriptions:
            # Candidate boxes from text
            text = truncate_tokens(text, detector.tokenizer)
            dets = detector(img_rgb, candidate_labels=[text], threshold=cfg.det_threshold)

            # Sort high->low score and keep top-K
            dets = sorted(dets, key=lambda d: float(d["score"]), reverse=True)[: cfg.max_boxes_per_description]

            refined_boxes: List[Tuple[int, int, int, int]] = []
            for d in dets:
                b = d["box"]
                raw_box = (int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"]))
                raw_box = _clip_box(raw_box, w, h)

                # MedSAM2 refine: box -> mask
                masks, ious, _ = predictor.predict(
                    box=np.array(raw_box, dtype=np.float32),
                    multimask_output=cfg.multimask_output,
                    normalize_coords=True,
                )

                # Pick best mask (if multimask_output=True, choose by predicted IoU)
                if masks.ndim != 3:  # expect (C,H,W)
                    continue
                best_i = int(np.argmax(ious)) if len(ious) else 0
                mask = masks[best_i]

                tight = _mask_to_box(mask)
                refined_boxes.append(_clip_box(tight if tight is not None else raw_box, w, h))

            results += refined_boxes

    return results


def get_attribution_attention_rollout(
    img,
    inputs,
    outputs,
    processor,
    target_start,
    target_end,
    layer_idx=-1,
    head_idx="avg",
    add_residual=True,
    eps=1e-8,
):
    """
    Attention rollout attribution heatmap for target token outputs in (decoder) Transformers / VLM LMs.

    Inputs:
      - img: PIL Image (used only for output size)
      - inputs: dict containing "input_ids" (and others)
      - outputs: model outputs with .attentions (list[L] of (1,H,S,S))
      - processor: has .tokenizer
      - target_start/target_end: token span (queries) whose output we attribute
      - layer_idx:
          * -1 (default): roll out across ALL layers
          * otherwise: roll out across layers [0..layer_idx] (supports negative indices too)
      - head_idx:
          * "avg": average over heads
          * int: specific head index

    Returns:
      - heatmap: (H_img, W_img) torch.FloatTensor on CPU in [0,1]
    """
    tok = processor.tokenizer
    ids = inputs["input_ids"][0]

    # ---- find image token id + positions (same robustness as your GradCAM) ----
    itok = getattr(tok, "image_token_id", None) or getattr(tok, "img_token_id", None)
    if itok is None:
        for s in ("<|image_pad|>", "<image>", "<image_soft_token>", "<|vision_start|>"):
            t = tok.convert_tokens_to_ids(s)
            if t is not None and t != tok.unk_token_id:
                itok = t
                break
    if itok is None:
        raise RuntimeError("Could not determine image token id from tokenizer.")

    image_pos = (ids == itok).nonzero(as_tuple=False).squeeze(-1)
    if image_pos.numel() == 0:
        raise RuntimeError("No image tokens found in input_ids.")

    # ---- attentions ----
    if not hasattr(outputs, "attentions") or outputs.attentions is None:
        raise RuntimeError("outputs.attentions is required for attention rollout.")

    attn_list = outputs.attentions
    L = len(attn_list)
    if L == 0:
        raise RuntimeError("outputs.attentions is empty.")

    # Determine how many layers to include in rollout
    if layer_idx == -1:
        last = L - 1
    else:
        last = layer_idx % L
    layers = range(0, last + 1)

    # Sequence length S
    # attn_list[l]: (1, H, S, S)
    S = attn_list[0].shape[-1]
    device = attn_list[0].device
    dtype = attn_list[0].dtype

    # Query span (target outputs) -- keep behavior similar to your attention method
    ts, te = int(target_start), int(target_end)
    ts = max(ts, 0)
    te = min(te, S)
    if te <= ts:
        raise RuntimeError(f"Invalid target span [{target_start}, {target_end}) for sequence length {S}.")

    # ---- build per-layer fused attention matrices with (optional) residual + row normalization ----
    I = torch.eye(S, device=device, dtype=dtype)

    def fuse_heads(attn_hss: torch.Tensor) -> torch.Tensor:
        # attn_hss: (H, S, S)
        if head_idx == "avg":
            return attn_hss.mean(dim=0)  # (S, S)
        else:
            h = int(head_idx)
            if h < 0 or h >= attn_hss.shape[0]:
                raise RuntimeError(f"head_idx={h} out of range for H={attn_hss.shape[0]}.")
            return attn_hss[h]  # (S, S)

    def row_normalize(M: torch.Tensor) -> torch.Tensor:
        # Make rows sum to 1 (stochastic), robust to zero rows.
        denom = M.sum(dim=-1, keepdim=True).clamp_min(eps)
        return M / denom

    # ---- rollout: joint = A0 @ A1 @ ... @ A_last ----
    joint = I.clone()
    for l in layers:
        # (1, H, S, S) -> (H, S, S)
        A = attn_list[l][0]
        A = fuse_heads(A)  # (S, S)

        if add_residual:
            A = A + I
        A = row_normalize(A)

        joint = joint @ A

    # ---- aggregate joint attention over query span -> token importance over keys ----
    # joint rows correspond to "from token" (query position), cols correspond to "to token" (key position)
    a = joint[ts:te, :].mean(dim=0)  # (S,)

    # keep only image-token keys -> (N_img,)
    img_scores = a.index_select(0, image_pos.to(device)).float()
    n = img_scores.numel()

    # ---- reshape into square grid (optionally skipping 1 special token) ----
    off = 0
    s = int(math.isqrt(n))
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != (n - 1):
            raise RuntimeError(f"Can't make square grid from {n} image tokens (need n or n-1 to be a square).")
        off, s = 1, s2

    patch_map = img_scores[off:off + s * s].view(s, s)

    # ---- upsample to original image size ----
    W, H = img.size
    heatmap = F.interpolate(
        patch_map[None, None], size=(H, W),
        mode="bilinear", align_corners=False
    )[0, 0]

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
    return heatmap.detach().cpu().float()

def get_attribution_grad_weighted_attention(
    img,
    inputs,
    model,
    processor,
    target_start,
    target_end,
    layer_idx=-1,
    head_idx="avg",
):
    """
    Gradient-Weighted Attention attribution for a target output token span.

    Returns:
        heatmap: torch.FloatTensor on CPU with shape (H, W), normalized to [0, 1]
    """
    tok, ids = processor.tokenizer, inputs["input_ids"]

    # ---- find image token id + its positions in the full seq ----
    itok = getattr(tok, "image_token_id", None) or getattr(tok, "img_token_id", None)
    if itok is None:
        for s in ("<|image_pad|>", "<image>", "<image_soft_token>", "<|vision_start|>"):
            t = tok.convert_tokens_to_ids(s)
            if t is not None and t != tok.unk_token_id:
                itok = t
                break
    if itok is None:
        raise RuntimeError("Could not determine image token id from tokenizer.")

    img_pos = (ids[0] == itok).nonzero(as_tuple=False).squeeze(-1)
    if img_pos.numel() == 0:
        raise RuntimeError("No image tokens found in input_ids.")

    # ---- forward pass with attentions kept in graph ----
    # Pixel values don't strictly need grad for this method, but keeping them requires_grad
    # helps ensure the whole graph stays differentiable in some model wrappers.
    fwd = dict(inputs)
    # if "pixel_values" in fwd and fwd["pixel_values"] is not None:
    #     pv = fwd["pixel_values"].detach().float().requires_grad_(True)
    #     fwd["pixel_values"] = pv

    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            **fwd,
            use_cache=False,
            return_dict=True,
            output_attentions=True,
        )

    if not hasattr(out, "attentions") or out.attentions is None or len(out.attentions) == 0:
        raise RuntimeError(
            "Model did not return attentions. Ensure output_attentions=True is supported, "
            "and (for some HF models) consider using eager attention implementation."
        )

    # ---- pick layer attention ----
    L = len(out.attentions)
    li = layer_idx % L
    attn = out.attentions[li]  # (B=1, H, S, S) typically

    if attn.dim() != 4:
        raise RuntimeError(f"Unexpected attention tensor shape: {tuple(attn.shape)}")

    # ---- objective: sum log-prob over target token span ----
    ts, te = max(int(target_start), 1), int(target_end)
    if te <= ts:
        raise ValueError(f"target_end must be > target_start, got {ts}, {te}")

    tgt = ids[:, ts:te]  # (1, span)
    # logits at position t-1 predict token t
    logp = F.log_softmax(out.logits[:, ts - 1 : te - 1, :], dim=-1)  # (1, span, V)
    obj = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()       # scalar

    # ---- gradients wrt attention weights ----
    grad_attn = torch.autograd.grad(
        obj, attn, retain_graph=False, create_graph=False, allow_unused=False
    )[0]  # (1, H, S, S)

    # ---- gradient-weighted attention: ReLU(grad) ⊙ attn ----
    # (Class/token-span conditional routing strength)
    gw = F.relu(grad_attn) * attn

    # ---- aggregate over query positions in the target span ----
    # We want: importance over KEY positions, specifically image-token keys.
    qs = slice(ts, te)  # query token indices that "generate" the span

    # gw[0, :, qs, :] -> (H, span, S)
    if head_idx == "avg":
        a = gw[0, :, qs, :].mean(dim=1).mean(dim=0)  # (S,)
    else:
        h = int(head_idx)
        a = gw[0, h, qs, :].mean(dim=0)               # (S,)

    # ---- keep only image-token keys ----
    img_scores = a.index_select(0, img_pos.to(a.device)).float()  # (N_img,)
    n = int(img_scores.numel())

    # ---- reshape into square grid (optionally skipping 1 special token) ----
    off = 0
    s = int(math.isqrt(n))
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1:
            raise RuntimeError(
                f"Can't reshape {n} image tokens into square grid (n or n-1 must be square)."
            )
        off, s = 1, s2
    patch_map = img_scores[off : off + s * s].view(s, s)

    # ---- upsample to original image size ----
    W, H = img.size
    heatmap = F.interpolate(
        patch_map[None, None], size=(H, W), mode="bilinear", align_corners=False
    )[0, 0]

    # ---- normalize to [0, 1] ----
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.detach().cpu().float()

def get_attribution_lrp(img, inputs, outputs, processor, target_start, target_end,
                        layer_idx=-1, head_idx="avg", add_residual=True, eps=1e-8):
    tok = processor.tokenizer
    ids = inputs["input_ids"][0]
    device = ids.device

    # --- image token id + positions ---
    itok = getattr(tok, "image_token_id", None) or getattr(tok, "img_token_id", None)
    if itok is None:
        for s in ("<|image_pad|>", "<image>", "<image_soft_token>", "<|vision_start|>"):
            t = tok.convert_tokens_to_ids(s)
            if t is not None and t != tok.unk_token_id:
                itok = t
                break
    if itok is None:
        raise RuntimeError("Couldn't determine image token id for this tokenizer/model.")

    img_pos = (ids == itok).nonzero(as_tuple=False).squeeze(-1)
    if img_pos.numel() == 0:
        raise RuntimeError("No image tokens found in input_ids.")

    attns = getattr(outputs, "attentions", None)
    if not attns:
        raise RuntimeError("outputs.attentions missing; run forward with output_attentions=True.")
    L = len(attns)
    top = layer_idx % L
    S = attns[0].shape[-1]

    # --- seed relevance on predictor positions for the target span ---
    ts, te = max(int(target_start), 1), int(target_end)
    q0, q1 = max(ts - 1, 0), min(te - 1, S)  # logits at [t-1] predict token t
    if q1 <= q0:
        raise RuntimeError("Target span maps to empty predictor positions; check indices.")

    R = torch.zeros(S, device=device, dtype=torch.float32)
    R[q0:q1] = 1.0
    R = R / R.sum().clamp_min(eps)

    eye = torch.eye(S, device=device, dtype=torch.float32)

    # --- LRP-style backward relevance propagation through attention ---
    for l in range(top, -1, -1):
        A = attns[l][0].to(torch.float32)  # (H,S,S)

        if head_idx == "avg":
            A = A.mean(dim=0)              # (S,S)
        else:
            A = A[int(head_idx)]

        if add_residual:
            A = A + eye

        # row-stochastic transition => conservative redistribution
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(eps)

        # push relevance backward (keys receive relevance from queries)
        R = A.t().matmul(R)
        R = R / R.sum().clamp_min(eps)

    # --- keep only image-token relevance and reshape to patch grid ---
    t = R.index_select(0, img_pos).float()
    n = int(t.numel())
    s = int(math.isqrt(n))
    off = 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1:
            raise RuntimeError(f"Can't reshape {n} image tokens into square (need n or n-1 square).")
        off, s = 1, s2

    patch = t[off:off + s * s].view(s, s)

    # --- upsample to image size ---
    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    hm = (hm - hm.min()) / (hm.max() - hm.min() + eps)
    return hm.detach().cpu().float()

# mask specific image tokens and observe the effect


def get_attribution_single_perturb_input(
    img,
    inputs,
    outputs,
    model,
    processor,
    target_start,
    target_end,
    layer_idx=0
):

    def perturb(module, args, img_mask, alternatives):
        x = args[0]  # hidden_states
        mask = img_mask[..., None].to(device=x.device)
        x = x * (~mask) + alternatives * mask
        return (x, *args[1:])

    img_mask_id = getattr(processor.tokenizer, "image_token_id", None) or processor.tokenizer.convert_tokens_to_ids("<|image_pad|>") or processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    img_mask = (inputs.input_ids == img_mask_id)
    start_img_token_idx = img_mask[0].nonzero(as_tuple=False)[0].item()
    end_img_token_idx = img_mask[0].nonzero(as_tuple=False)[-1].item() + 1

    removal_probs = []
    alternatives = outputs.hidden_states[layer_idx].clone()
    img_token_embeddings = outputs.hidden_states[layer_idx][img_mask]
    alternatives[img_mask] = img_token_embeddings.mean(dim=0, keepdim=True).expand_as(img_token_embeddings)

    with torch.inference_mode():
        # create a mask for each image token separately
        for img_token_idx in range(start_img_token_idx, end_img_token_idx):
            img_mask = (inputs.input_ids == img_mask_id) & (torch.arange(inputs.input_ids.shape[1], device=inputs.input_ids.device)[None, :] == img_token_idx)
            with torch.no_grad():
                handle = model.get_submodule(f'model.language_model.layers.{layer_idx}').register_forward_pre_hook(
                    partial(
                        perturb,
                        img_mask=img_mask,
                        alternatives=alternatives
                    )
                )
                outputs_perturbed = model(
                    **inputs,
                    return_dict_in_generate=True, 
                    # output_hidden_states=True, 
                    # output_attentions=True, 
                    output_scores=True
                )
                handle.remove()
            removal_probs.append(
                outputs_perturbed.logits[0, target_start-1:target_end-1].softmax(-1).gather(-1, inputs.input_ids[0, target_start:target_end].unsqueeze(-1)).squeeze(-1).prod().item()
            )

    t = 1 - ((torch.tensor(removal_probs) - torch.tensor(removal_probs).min()) / (torch.tensor(removal_probs).max() - torch.tensor(removal_probs).min()).clamp_min(1e-8))
    n = int(t.numel())
    s = int(math.isqrt(n)); off = 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1: raise RuntimeError(f"Can't reshape {n} image tokens into square (n or n-1).")
        off, s = 1, s2
    patch = t[off:off + s*s].view(s, s)

    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    heatmap = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    return heatmap.detach().cpu().float()

def get_attribution_single_perturb_output(
    img,
    inputs,
    outputs,
    model,
    processor,
    target_start,
    target_end,
    layer_idx=0
):

    def perturb(module, inputs, outputs, img_mask, alternatives):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs[:] = outputs * (~img_mask[:,:,None]) + alternatives * img_mask[:,:,None]

    img_mask_id = getattr(processor.tokenizer, "image_token_id", None) or processor.tokenizer.convert_tokens_to_ids("<|image_pad|>") or processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    img_mask = (inputs.input_ids == img_mask_id)
    start_img_token_idx = img_mask[0].nonzero(as_tuple=False)[0].item()
    end_img_token_idx = img_mask[0].nonzero(as_tuple=False)[-1].item() + 1

    removal_probs = []
    alternatives = outputs.hidden_states[layer_idx+1].clone()
    img_token_embeddings = outputs.hidden_states[layer_idx+1][img_mask]
    alternatives[img_mask] = img_token_embeddings.mean(dim=0, keepdim=True).expand_as(img_token_embeddings)

    with torch.inference_mode():
        # create a mask for each image token separately
        for img_token_idx in range(start_img_token_idx, end_img_token_idx):
            img_mask = (inputs.input_ids == img_mask_id) & (torch.arange(inputs.input_ids.shape[1], device=inputs.input_ids.device)[None, :] == img_token_idx)
            with torch.no_grad():
                handle = model.get_submodule(f'model.language_model.layers.{layer_idx}').register_forward_hook(
                    partial(
                        perturb,
                        img_mask=img_mask,
                        alternatives=alternatives
                    )
                )
                outputs_perturbed = model(
                    **inputs,
                    return_dict_in_generate=True, 
                    # output_hidden_states=True, 
                    # output_attentions=True, 
                    output_scores=True
                )
                handle.remove()
            removal_probs.append(
                outputs_perturbed.logits[0, target_start-1:target_end-1].softmax(-1).gather(-1, inputs.input_ids[0, target_start:target_end].unsqueeze(-1)).squeeze(-1).prod().item()
            )

    t = 1 - ((torch.tensor(removal_probs) - torch.tensor(removal_probs).min()) / (torch.tensor(removal_probs).max() - torch.tensor(removal_probs).min()).clamp_min(1e-8))
    n = int(t.numel())
    s = int(math.isqrt(n)); off = 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1: raise RuntimeError(f"Can't reshape {n} image tokens into square (n or n-1).")
        off, s = 1, s2
    patch = t[off:off + s*s].view(s, s)

    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    heatmap = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

    return heatmap.detach().cpu().float()

def get_attribution_single_perturb_dynamic_output(
    img,
    inputs,
    outputs,
    model,
    processor,
    target_start,
    target_end,
):

    def perturb(module, inputs, outputs, img_mask, alternatives):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs[:] = outputs * (~img_mask[:,:,None]) + alternatives * img_mask[:,:,None]

    img_mask_id = getattr(processor.tokenizer, "image_token_id", None) or processor.tokenizer.convert_tokens_to_ids("<|image_pad|>") or processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")
    img_mask = (inputs.input_ids == img_mask_id)

    removal_probs = []
    with torch.inference_mode():
        for layer_idx in range(len(outputs.hidden_states)-1):
            alternatives = outputs.hidden_states[layer_idx+1].clone()
            img_token_embeddings = outputs.hidden_states[layer_idx+1][img_mask]
            alternatives[img_mask] = img_token_embeddings.mean(dim=0, keepdim=True).expand_as(img_token_embeddings)
            with torch.no_grad():
                handle = model.get_submodule(f'model.language_model.layers.{layer_idx}').register_forward_hook(
                    partial(
                        perturb,
                        img_mask=img_mask,
                        alternatives=alternatives
                    )
                )
                outputs_perturbed = model(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                handle.remove()
            removal_probs.append(
                outputs_perturbed.logits[0, target_start-1:target_end-1].softmax(-1).gather(-1, inputs.input_ids[0, target_start:target_end].unsqueeze(-1)).squeeze(-1).prod().item()
            )
    max_layer_idx = torch.tensor(removal_probs).argmin().item()
    return get_attribution_single_perturb_output(
        img,
        inputs,
        outputs,
        model,
        processor,
        target_start,
        target_end,
        layer_idx=max_layer_idx
    )

def get_attribution_pixel_perturb_output(
    img,
    inputs,
    outputs,
    model,
    processor,
    target_start,
    target_end,
    baseline='zero',
    patches=[],
    batch_size=32
):

    if img.mode != "L":
        img = img.convert("L")

    baseline_img = None

    # removal_probs = []
    if baseline == 'zero':
        baseline_img = Image.new('L', img.size, 0)
    elif baseline == 'mean':
        img_array = np.array(img)
        mean_value = int(img_array.mean())
        baseline_img = Image.new('L', img.size, mean_value)
    elif baseline == 'blur':
        blur_radius = max(1, int(0.2 * min(img.size)))
        baseline_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    with torch.inference_mode():
        # create a perturbed img for each patch (x1, y1, x2, y2) separately with the patch part replaced by baseline
        # Prepare all perturbed images first
        perturbed_imgs = []
        for patch in patches:
            perturbed_img = img.copy()
            x1, y1, x2, y2 = patch
            perturbed_img.paste(baseline_img.crop((x1, y1, x2, y2)), (x1, y1))
            perturbed_imgs.append(perturbed_img)
        
        # Batch process all images at once
        if 'gemma' in model.config.architectures[0].lower():
            batch_processed = processor(
                text="<start_of_image><image_soft_token><end_of_image>" * len(patches),
                images=perturbed_imgs,
                return_tensors="pt"
            )
        else:
            batch_processed = processor(
                text="",
                images=perturbed_imgs,
                return_tensors="pt"
            )
        
        perturbed_pixel_values = batch_processed['pixel_values'].to(inputs['pixel_values'].dtype).to(inputs['pixel_values'].device)
        ori_len = inputs['pixel_values'].shape[0]
        perturbed_logits = []
        # for i in range(0, len(perturbed_pixel_values), batch_size):
        for i in range(0, len(patches), batch_size):
            batch_pixel_values = perturbed_pixel_values[i * ori_len : (i + batch_size) * ori_len]
            perturbed_inputs = inputs.copy()
            perturbed_inputs['pixel_values'] = batch_pixel_values
            for k in perturbed_inputs:
                if k != 'pixel_values':
                    perturbed_inputs[k] = perturbed_inputs[k].repeat(len(batch_pixel_values) // ori_len, *([1] * (perturbed_inputs[k].dim() - 1)))
            outputs_perturbed = model(
                **perturbed_inputs,
                return_dict_in_generate=True,
                output_scores=True
            )
            perturbed_logits.append(outputs_perturbed.logits)
        
        perturbed_logits = torch.cat(perturbed_logits, dim=0)
        # for logits in perturbed_logits:
        #     removal_probs.append(
        #         logits[target_start-1:target_end-1].softmax(-1).gather(-1, inputs.input_ids[0, target_start:target_end].unsqueeze(-1)).squeeze(-1).prod().item()
        #     )
        target_logits = perturbed_logits[:, target_start-1:target_end-1]  # shape: (num_patches, target_len, vocab_size)
        target_probs = target_logits.softmax(-1)  # shape: (num_patches, target_len, vocab_size)
        target_ids = inputs.input_ids[0, target_start:target_end].unsqueeze(0).unsqueeze(-1)  # shape: (1, target_len, 1)
        selected_probs = target_probs.gather(-1, target_ids.expand(len(patches), -1, -1)).squeeze(-1)  # shape: (num_patches, target_len)
        removal_probs = selected_probs.prod(dim=-1)  # shape: (num_patches,)

    t = 1 - ((removal_probs - removal_probs.min()) / (removal_probs.max() - removal_probs.min()).clamp_min(1e-8))
    n = int(t.numel())
    s = int(math.isqrt(n)); off = 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1: raise RuntimeError(f"Can't reshape {n} image tokens into square (n or n-1).")
        off, s = 1, s2
    patch = t[off:off + s*s].view(s, s)

    W, H = img.size
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    heatmap = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return heatmap.detach().cpu().float()


def get_attribution_rise(
    img, inputs, outputs, model, processor, target_start, target_end,
    baseline="zero", patches=[], batch_size=32, num_masks=64, p_keep=0.5
):
    if img.mode != "L": img = img.convert("L")
    W, H = img.size

    if baseline == "zero":
        base = Image.new("L", img.size, 0)
    elif baseline == "mean":
        base = Image.new("L", img.size, int(np.array(img).mean()))
    elif baseline == "blur":
        base = img.filter(ImageFilter.GaussianBlur(radius=max(1, int(0.2 * min(img.size)))))
    else:
        raise ValueError(baseline)

    n = len(patches)
    s, off = int(math.isqrt(n)), 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1: raise RuntimeError(f"Can't reshape {n} tokens into square (n or n-1).")
        s, off = s2, 1
    grid = patches[off:off + s * s]
    K = s * s

    # Pre-crop original patch content + store paste positions (fast PIL path)
    crops = [img.crop((x1, y1, x2, y2)) for (x1, y1, x2, y2) in grid]
    pos   = [(x1, y1) for (x1, y1, _, _) in grid]

    # Pre-sample all masks once
    masks = (torch.rand(num_masks, K) < p_keep)  # bool on CPU

    # Build all perturbed images once (like your pixel_perturb function)
    perturbed_imgs = []
    for m in masks:
        im = base.copy()
        for j in m.nonzero(as_tuple=True)[0].tolist():
            im.paste(crops[j], pos[j])
        perturbed_imgs.append(im)

    is_gemma = "gemma" in model.config.architectures[0].lower()
    proc = processor(
        text=("<start_of_image><image_soft_token><end_of_image>" * num_masks) if is_gemma else "",
        images=perturbed_imgs, return_tensors="pt"
    )

    dev, dtype = inputs["pixel_values"].device, inputs["pixel_values"].dtype
    pv = proc["pixel_values"].to(dtype).to(dev)
    ori_len = inputs["pixel_values"].shape[0]

    ids = inputs.input_ids[0, target_start:target_end][None, :, None]  # (1,T,1)
    masks_dev = masks.to(dev).float()                                  # (N,K)

    sal_grid = torch.zeros(K, device=dev)

    with torch.inference_mode():
        for i in range(0, num_masks, batch_size):
            bs = min(batch_size, num_masks - i)
            batch_pv = pv[i * ori_len : (i + bs) * ori_len]

            pert = inputs.copy()
            pert["pixel_values"] = batch_pv
            rep = len(batch_pv) // ori_len
            for k in pert:
                if k != "pixel_values":
                    pert[k] = pert[k].repeat(rep, *([1] * (pert[k].dim() - 1)))

            logits = model(**pert, return_dict_in_generate=True, output_scores=True).logits
            probs  = logits[:, target_start-1:target_end-1].softmax(-1)
            # scores = probs.gather(-1, ids.expand(rep, -1, -1)).squeeze(-1).prod(-1).float()
            scores = probs.gather(-1, ids.expand(rep, -1, -1)).squeeze(-1).log().mean(dim=-1).exp().float()
            sal_grid += (scores[:, None] * masks_dev[i:i + bs]).sum(0)

    sal = torch.zeros(n, device=dev)
    sal[off:off + K] = sal_grid / (num_masks * max(p_keep, 1e-8))
    if off: sal[0] = sal.max()  # harmless, keeps downstream normalization stable

    t = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    patch = t[off:off + K].view(s, s)
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8)).detach().cpu().float()


def sobol_masks(K, num_masks, keep_prob=0.5, seed=0, device="cpu"):
    engine = torch.quasirandom.SobolEngine(dimension=K, scramble=True, seed=seed)
    return (engine.draw(num_masks).to(device) < keep_prob)  # [num_masks, K] bool

def get_attribution_rise_sobol(
    img, inputs, outputs, model, processor, target_start, target_end,
    baseline="zero", patches=[], batch_size=32, num_masks=64, p_keep=0.5, seed=0
):
    if img.mode != "L": img = img.convert("L")
    W, H = img.size

    if baseline == "zero":
        base = Image.new("L", img.size, 0)
    elif baseline == "mean":
        base = Image.new("L", img.size, int(np.array(img).mean()))
    elif baseline == "blur":
        base = img.filter(ImageFilter.GaussianBlur(radius=max(1, int(0.2 * min(img.size)))))
    else:
        raise ValueError(baseline)

    n = len(patches)
    s, off = int(math.isqrt(n)), 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1: raise RuntimeError(f"Can't reshape {n} tokens into square (n or n-1).")
        s, off = s2, 1
    grid = patches[off:off + s * s]
    K = s * s

    masks = sobol_masks(K, num_masks, keep_prob=p_keep, seed=seed, device="cpu")  # bool [N,K]
    crops = [img.crop((x1, y1, x2, y2)) for (x1, y1, x2, y2) in grid]
    topleft = [(x1, y1) for (x1, y1, _, _) in grid]

    perturbed_imgs = []
    for m in masks:
        im = base.copy()
        for j in m.nonzero(as_tuple=True)[0].tolist():
            im.paste(crops[j], topleft[j])
        perturbed_imgs.append(im)

    is_gemma = "gemma" in model.config.architectures[0].lower()
    proc = processor(
        text=("<start_of_image><image_soft_token><end_of_image>" * num_masks) if is_gemma else "",
        images=perturbed_imgs, return_tensors="pt"
    )

    dev, dtype = inputs["pixel_values"].device, inputs["pixel_values"].dtype
    pv = proc["pixel_values"].to(dtype).to(dev)
    ori_len = inputs["pixel_values"].shape[0]
    target_ids = inputs.input_ids[0, target_start:target_end][None, :, None]  # (1,T,1)
    masks_dev = masks.to(dev).float()

    sal_grid = torch.zeros(K, device=dev)

    with torch.inference_mode():
        for i in range(0, num_masks, batch_size):
            bs = min(batch_size, num_masks - i)
            batch_pv = pv[i * ori_len : (i + bs) * ori_len]
            pert = inputs.copy()
            pert["pixel_values"] = batch_pv
            rep = len(batch_pv) // ori_len
            for k in pert:
                if k != "pixel_values":
                    pert[k] = pert[k].repeat(rep, *([1] * (pert[k].dim() - 1)))

            logits = model(**pert, return_dict_in_generate=True, output_scores=True).logits
            probs = logits[:, target_start-1:target_end-1].softmax(-1)
            scores = probs.gather(-1, target_ids.expand(rep, -1, -1)).squeeze(-1).prod(-1).float()
            sal_grid += (scores[:, None] * masks_dev[i:i + bs]).sum(0)

    sal = torch.zeros(n, device=dev)
    sal[off:off + K] = sal_grid / (num_masks * max(p_keep, 1e-8))

    t = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    patch = t[off:off + K].view(s, s)
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8)).detach().cpu().float()

def get_probs_mask_perturb_output(
    img,
    inputs,
    outputs,
    model,
    processor,
    target_start,
    target_end,
    baseline='zero',
    masks=None,
    batch_size=32,
    output_token_probs=False
):

    if img.mode != "L":
        img = img.convert("L")

    if type(baseline) == list:
        perturbed_imgs = baseline
    else:
        # removal_probs = []
        if baseline == 'zero':
            baseline_img = Image.new('L', img.size, 0)
        elif baseline == 'mean':
            img_array = np.array(img)
            mean_value = int(img_array.mean())
            baseline_img = Image.new('L', img.size, mean_value)
        elif baseline == 'blur':
            blur_radius = max(1, int(0.2 * min(img.size)))
            baseline_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # create a perturbed img for each patch (x1, y1, x2, y2) separately with the patch part replaced by baseline
        # Prepare all perturbed images first
        perturbed_imgs = []
        # for patch in patches:
        #     perturbed_img = img.copy()
        #     x1, y1, x2, y2 = patch
        #     perturbed_img.paste(baseline_img.crop((x1, y1, x2, y2)), (x1, y1))
        #     perturbed_imgs.append(perturbed_img)
        for mask in masks:
            perturbed_img = img.copy()
            perturbed_img_np = np.array(perturbed_img)
            baseline_img_np = np.array(baseline_img)
            perturbed_img_np[mask > 0] = baseline_img_np[mask > 0]
            perturbed_img = Image.fromarray(perturbed_img_np)
            perturbed_imgs.append(perturbed_img)

    with torch.inference_mode():

        # Batch process all images at once
        if 'gemma' in model.config.architectures[0].lower():
            batch_processed = processor(
                text="<start_of_image><image_soft_token><end_of_image>" * len(masks),
                images=perturbed_imgs,
                return_tensors="pt"
            )
        else:
            batch_processed = processor(
                text="",
                images=perturbed_imgs,
                return_tensors="pt"
            )
        
        perturbed_pixel_values = batch_processed['pixel_values'].to(inputs['pixel_values'].dtype).to(inputs['pixel_values'].device)
        ori_len = inputs['pixel_values'].shape[0]
        perturbed_logits = []
        # for i in range(0, len(perturbed_pixel_values), batch_size):
        for i in range(0, len(masks), batch_size):
            batch_pixel_values = perturbed_pixel_values[i * ori_len : (i + batch_size) * ori_len]
            perturbed_inputs = inputs.copy()
            perturbed_inputs['pixel_values'] = batch_pixel_values
            for k in perturbed_inputs:
                if k != 'pixel_values':
                    perturbed_inputs[k] = perturbed_inputs[k].repeat(len(batch_pixel_values) // ori_len, *([1] * (perturbed_inputs[k].dim() - 1)))
            outputs_perturbed = model(
                **perturbed_inputs,
                return_dict_in_generate=True,
                output_scores=True
            )
            perturbed_logits.append(outputs_perturbed.logits)
        
        perturbed_logits = torch.cat(perturbed_logits, dim=0)
        # for logits in perturbed_logits:
        #     removal_probs.append(
        #         logits[target_start-1:target_end-1].softmax(-1).gather(-1, inputs.input_ids[0, target_start:target_end].unsqueeze(-1)).squeeze(-1).prod().item()
        #     )
        target_logits = perturbed_logits[:, target_start-1:target_end-1]  # shape: (num_patches, target_len, vocab_size)
        target_probs = target_logits.softmax(-1)  # shape: (num_patches, target_len, vocab_size)
        target_ids = inputs.input_ids[0, target_start:target_end].unsqueeze(0).unsqueeze(-1)  # shape: (1, target_len, 1)
        selected_probs = target_probs.gather(-1, target_ids.expand(len(masks), -1, -1)).squeeze(-1)  # shape: (num_patches, target_len)
        # removal_probs = selected_probs.prod(dim=-1)  # shape: (num_patches,)
        removal_probs = selected_probs.log().mean(dim=-1).exp()  # shape:(num_patches,)
        if output_token_probs:
            return removal_probs.cpu(), selected_probs.cpu()
    return removal_probs.cpu()

def get_attribution_occlusion(
    img, inputs, outputs, model, processor, target_start, target_end,
    baseline="zero", patches=[], batch_size=32, positive_only=True
):
    """
    Occlusion sensitivity over the same patch grid used by your RISE code.

    - Builds one perturbed image per patch by *occluding exactly that patch* (replace with baseline patch).
    - Scores each occluded image with the same token-prob-product scoring you use in RISE.
    - Attribution per patch = score(original) - score(occluded). (Optionally clamp to >=0)

    Returns: (H,W) heatmap tensor on CPU in [0,1].
    """
    if img.mode != "L":
        img = img.convert("L")
    W, H = img.size

    # Baseline image (used only as a source of "replacement patch" content)
    if baseline == "zero":
        base = Image.new("L", img.size, 0)
    elif baseline == "mean":
        base = Image.new("L", img.size, int(np.array(img).mean()))
    elif baseline == "blur":
        base = img.filter(ImageFilter.GaussianBlur(radius=max(1, int(0.2 * min(img.size)))))
    else:
        raise ValueError(baseline)

    # Same "square grid (n or n-1)" logic as your RISE
    n = len(patches)
    s, off = int(math.isqrt(n)), 0
    if s * s != n:
        s2 = int(math.isqrt(n - 1))
        if s2 * s2 != n - 1:
            raise RuntimeError(f"Can't reshape {n} tokens into square (n or n-1).")
        s, off = s2, 1

    grid = patches[off:off + s * s]
    K = s * s

    # Precompute baseline crops to paste in (one patch occluded at a time)
    base_crops = [base.crop((x1, y1, x2, y2)) for (x1, y1, x2, y2) in grid]
    pos        = [(x1, y1) for (x1, y1, _, _) in grid]

    occluded_imgs = []
    for j in range(K):
        im = img.copy()
        im.paste(base_crops[j], pos[j])  # occlude just patch j
        occluded_imgs.append(im)

    # Processor call mirrors your RISE path
    is_gemma = "gemma" in model.config.architectures[0].lower()
    proc = processor(
        text=("<start_of_image><image_soft_token><end_of_image>" * K) if is_gemma else "",
        images=occluded_imgs, return_tensors="pt"
    )

    dev, dtype = inputs["pixel_values"].device, inputs["pixel_values"].dtype
    pv = proc["pixel_values"].to(dtype).to(dev)
    ori_len = inputs["pixel_values"].shape[0]

    # Target token IDs and scoring (same as your RISE)
    ids = inputs.input_ids[0, target_start:target_end][None, :, None]  # (1,T,1)

    # Compute original score once
    with torch.inference_mode():
        logits0 = model(**inputs, return_dict_in_generate=True, output_scores=True).logits
        probs0  = logits0[:, target_start-1:target_end-1].softmax(-1)
        # score0  = probs0.gather(-1, ids).squeeze(-1).prod(-1).float()  # (B,) usually B=1
        score0  = probs0.gather(-1, ids).squeeze(-1).log().mean(dim=-1).exp().float()  # (B,) usually B=1

    sal_grid = torch.zeros(K, device=dev)

    # Score each single-patch-occluded image (batched)
    with torch.inference_mode():
        for i in range(0, K, batch_size):
            bs = min(batch_size, K - i)

            batch_pv = pv[i * ori_len : (i + bs) * ori_len]

            pert = inputs.copy()
            pert["pixel_values"] = batch_pv

            rep = len(batch_pv) // ori_len  # should equal bs
            for k in pert:
                if k != "pixel_values":
                    pert[k] = pert[k].repeat(rep, *([1] * (pert[k].dim() - 1)))

            logits = model(**pert, return_dict_in_generate=True, output_scores=True).logits
            probs  = logits[:, target_start-1:target_end-1].softmax(-1)
            scores = probs.gather(-1, ids.expand(rep, -1, -1)).squeeze(-1).prod(-1).float()

            deltas = score0.expand_as(scores) - scores  # drop when occluded
            if positive_only:
                deltas = deltas.clamp_min(0)

            sal_grid[i:i + bs] = deltas

    # Pack back into token-length vector, normalize, and upsample to (H,W)
    sal = torch.zeros(n, device=dev)
    sal[off:off + K] = sal_grid
    if off:
        sal[0] = sal.max()  # keep downstream normalization stable

    t = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    patch = t[off:off + K].view(s, s)
    hm = F.interpolate(patch[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8)).detach().cpu().float()