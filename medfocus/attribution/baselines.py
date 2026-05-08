"""Ten attribution baselines used in the paper.

All functions return either:
    - a (H, W) torch.FloatTensor heatmap on CPU in [0, 1]; or
    - for the prompting-based methods, a list of [x1, y1, x2, y2] integer boxes.

The two output forms are unified downstream by `heatmap_to_bboxes_quantile`.

Group → method mapping:
    attention-based   : attention_head, attention_rollout, lrp, grad_weighted_attention
    gradient-based    : gradcam, gradcampp, integrated_gradients
    perturbation-based: occlusion, rise
    prompting-based   : prompting (LVLM-only), prompting_medsam (LVLM + MedSAM refine)

Most attribution methods need attentions / hidden states; load the LVLM with
`attn_implementation="eager"` to make `output_attentions=True` work reliably.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from medfocus.lvlm.adapters import LVLMAdapter
from medfocus.lvlm.generation import teacher_forced_forward


# ============================================================================
# Shared helpers
# ============================================================================

def _patch_grid_shape(n: int) -> tuple[int, int, int]:
    """Return (s, s, off) such that the (n-off) image tokens form an s×s grid."""
    s = int(math.isqrt(n))
    if s * s == n:
        return s, s, 0
    s2 = int(math.isqrt(n - 1))
    if s2 * s2 == n - 1:
        return s2, s2, 1
    raise RuntimeError(f"Cannot reshape {n} image tokens into a square grid (n or n-1).")


def _upsample_to_image(patch_2d: torch.Tensor, img: Image.Image) -> torch.Tensor:
    W, H = img.size
    hm = F.interpolate(
        patch_2d[None, None].float(), size=(H, W), mode="bilinear", align_corners=False
    )[0, 0]
    return ((hm - hm.min()) / (hm.max() - hm.min() + 1e-8)).detach().cpu().float()


def _target_logp_sum(out, ts: int, te: int, target_ids: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(out.logits[:, ts - 1 : te - 1, :], dim=-1)
    return logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).sum()


# ============================================================================
# Attention-based
# ============================================================================

def attention_head(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    layer_idx: int = -1,
    head_idx: int | str = "avg",
) -> torch.Tensor:
    """Raw attention from a single layer/head, aggregated over the answer span."""
    user_inputs, full_inputs, out = teacher_forced_forward(adapter, img, question, answer)
    ts, te = adapter.target_span(user_inputs, full_inputs)
    image_pos = adapter.image_token_positions(full_inputs["input_ids"])
    attn = out.attentions[layer_idx][0]  # (H, S, S)
    qs = slice(int(ts), int(te))
    if head_idx == "avg":
        a = attn[:, qs, :].mean(dim=1).mean(dim=0)
    else:
        a = attn[int(head_idx), qs, :].mean(dim=0)
    img_attn = a.index_select(0, image_pos.to(a.device)).float()
    s, _, off = _patch_grid_shape(int(img_attn.numel()))
    patch = img_attn[off: off + s * s].view(s, s)
    return _upsample_to_image(patch, img)


def attention_rollout(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    layer_idx: int = -1,
    head_idx: int | str = "avg",
    add_residual: bool = True,
) -> torch.Tensor:
    """Abnar & Zuidema (2020) attention rollout."""
    user_inputs, full_inputs, out = teacher_forced_forward(adapter, img, question, answer)
    ts, te = adapter.target_span(user_inputs, full_inputs)
    image_pos = adapter.image_token_positions(full_inputs["input_ids"])

    attn_list = out.attentions
    L = len(attn_list)
    last = L - 1 if layer_idx == -1 else (layer_idx % L)
    S = attn_list[0].shape[-1]
    device = attn_list[0].device
    dtype = attn_list[0].dtype
    eye = torch.eye(S, device=device, dtype=dtype)

    def fuse(A):  # (H,S,S)
        return A.mean(dim=0) if head_idx == "avg" else A[int(head_idx)]

    joint = eye.clone()
    for l in range(0, last + 1):
        A = fuse(attn_list[l][0])
        if add_residual:
            A = A + eye
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint = joint @ A

    a = joint[ts:te, :].mean(dim=0)
    img_scores = a.index_select(0, image_pos.to(device)).float()
    s, _, off = _patch_grid_shape(int(img_scores.numel()))
    patch = img_scores[off: off + s * s].view(s, s)
    return _upsample_to_image(patch, img)


def lrp(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    layer_idx: int = -1,
    head_idx: int | str = "avg",
    add_residual: bool = True,
) -> torch.Tensor:
    """Layer-wise relevance propagation through attention (Bach et al., 2015)."""
    user_inputs, full_inputs, out = teacher_forced_forward(adapter, img, question, answer)
    ts, te = adapter.target_span(user_inputs, full_inputs)
    image_pos = adapter.image_token_positions(full_inputs["input_ids"])

    attns = out.attentions
    L = len(attns)
    top = layer_idx % L
    S = attns[0].shape[-1]
    device = attns[0].device

    R = torch.zeros(S, device=device, dtype=torch.float32)
    R[max(ts - 1, 0): min(te - 1, S)] = 1.0
    R = R / R.sum().clamp_min(1e-8)
    eye = torch.eye(S, device=device, dtype=torch.float32)

    for l in range(top, -1, -1):
        A = attns[l][0].to(torch.float32)
        A = A.mean(dim=0) if head_idx == "avg" else A[int(head_idx)]
        if add_residual:
            A = A + eye
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        R = A.t().matmul(R)
        R = R / R.sum().clamp_min(1e-8)

    t = R.index_select(0, image_pos).float()
    s, _, off = _patch_grid_shape(int(t.numel()))
    patch = t[off: off + s * s].view(s, s)
    return _upsample_to_image(patch, img)


def grad_weighted_attention(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    layer_idx: int = -1,
    head_idx: int | str = "avg",
) -> torch.Tensor:
    """Chefer et al. (2021): ReLU(grad) ⊙ attn aggregated over the answer span."""
    user_inputs, full_inputs = adapter.build_teacher_forced_inputs(img, question, answer)
    image_pos = adapter.image_token_positions(full_inputs["input_ids"])
    ts, te = adapter.target_span(user_inputs, full_inputs)

    adapter.model.eval()
    out = adapter.model(**full_inputs, return_dict=True, output_attentions=True, use_cache=False)
    attn = out.attentions[layer_idx % len(out.attentions)]
    target_ids = full_inputs["input_ids"][:, ts:te]
    obj = _target_logp_sum(out, ts, te, target_ids)
    grad_attn = torch.autograd.grad(obj, attn, retain_graph=False)[0]
    gw = F.relu(grad_attn) * attn

    qs = slice(ts, te)
    if head_idx == "avg":
        a = gw[0, :, qs, :].mean(dim=1).mean(dim=0)
    else:
        a = gw[0, int(head_idx), qs, :].mean(dim=0)
    img_scores = a.index_select(0, image_pos.to(a.device)).float()
    s, _, off = _patch_grid_shape(int(img_scores.numel()))
    patch = img_scores[off: off + s * s].view(s, s)
    return _upsample_to_image(patch, img)


# ============================================================================
# Gradient-based
# ============================================================================

def gradcam(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    layer_idx: int = -1,
) -> torch.Tensor:
    """Selvaraju et al. (2017): Grad-CAM on decoder hidden states (image-token positions)."""
    user_inputs, full_inputs = adapter.build_teacher_forced_inputs(img, question, answer)
    ts, te = adapter.target_span(user_inputs, full_inputs)
    image_pos = adapter.image_token_positions(full_inputs["input_ids"])

    pv = full_inputs["pixel_values"].detach().float().requires_grad_(True)
    fwd = dict(full_inputs)
    fwd["pixel_values"] = pv
    adapter.model.eval()
    out = adapter.model(**fwd, return_dict=True, output_hidden_states=True, use_cache=False)
    L = len(out.hidden_states) - 1
    A = out.hidden_states[(layer_idx % L) + 1]  # (1, S, D)
    target_ids = full_inputs["input_ids"][:, ts:te]
    obj = _target_logp_sum(out, ts, te, target_ids)
    G = torch.autograd.grad(obj, A, retain_graph=False)[0]

    Ai, Gi = A[0, image_pos, :], G[0, image_pos, :]
    w = Gi.mean(0)
    t = F.relu((Ai * w).sum(-1))
    s, _, off = _patch_grid_shape(int(t.numel()))
    patch = t[off: off + s * s].view(s, s)
    return _upsample_to_image(patch, img)


def gradcampp(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    layer_idx: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Chattopadhay et al. (2018): Grad-CAM++ on decoder hidden states."""
    user_inputs, full_inputs = adapter.build_teacher_forced_inputs(img, question, answer)
    ts, te = adapter.target_span(user_inputs, full_inputs)
    image_pos = adapter.image_token_positions(full_inputs["input_ids"])

    pv = full_inputs["pixel_values"].detach().float().requires_grad_(True)
    fwd = dict(full_inputs)
    fwd["pixel_values"] = pv
    adapter.model.eval()
    out = adapter.model(**fwd, return_dict=True, output_hidden_states=True, use_cache=False)
    L = len(out.hidden_states) - 1
    A = out.hidden_states[(layer_idx % L) + 1]
    target_ids = full_inputs["input_ids"][:, ts:te]
    obj = _target_logp_sum(out, ts, te, target_ids)
    G = torch.autograd.grad(obj, A, retain_graph=False)[0]

    Ai = A[0, image_pos, :].float()
    Gi = G[0, image_pos, :].float()
    g2 = Gi * Gi
    g3 = g2 * Gi
    sumA = Ai.sum(dim=0, keepdim=True)
    alpha = g2 / (2.0 * g2 + sumA * g3 + eps)
    w = (alpha * F.relu(Gi)).sum(dim=0)
    t = F.relu((Ai * w).sum(dim=-1))
    s, _, off = _patch_grid_shape(int(t.numel()))
    patch = t[off: off + s * s].view(s, s)
    return _upsample_to_image(patch, img)


def integrated_gradients(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    steps: int = 24,
    baseline: str = "zero",
) -> torch.Tensor:
    """Sundararajan et al. (2017): IG over `pixel_values`."""
    user_inputs, full_inputs = adapter.build_teacher_forced_inputs(img, question, answer)
    ts, te = adapter.target_span(user_inputs, full_inputs)
    target_ids = full_inputs["input_ids"][:, ts:te]
    pos = torch.arange(ts - 1, te - 1, device=full_inputs["input_ids"].device)
    fwd = {k: v for k, v in full_inputs.items() if k != "pixel_values"}

    x = full_inputs["pixel_values"].detach().float()
    if baseline == "zero":
        b = torch.zeros_like(x)
    elif baseline == "mean":
        if x.dim() == 4:
            b = x.mean(dim=(2, 3), keepdim=True).expand_as(x)
        else:
            b = x.mean().expand_as(x)
    else:
        raise ValueError(f"Unknown baseline {baseline!r}")
    d = x - b

    alphas = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)
    total = torch.zeros_like(x)
    adapter.model.eval()
    for a in alphas:
        pv = (b + a * d).detach().requires_grad_(True)
        out = adapter.model(**fwd, pixel_values=pv, use_cache=False)
        logp = F.log_softmax(out.logits.index_select(1, pos), dim=-1)
        obj = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).sum()
        grad = torch.autograd.grad(obj, pv, retain_graph=False)[0]
        total += grad

    ig = d * (total / float(steps))
    if ig.dim() == 4:
        ig = ig.permute(0, 2, 3, 1)[0]
    elif ig.dim() == 3 and ig.shape[0] == 1:
        ig = ig[0]
    if ig.dim() == 2:
        h = math.isqrt(img.size[1] * ig.shape[0] // img.size[0])
        ig = ig.reshape(h, ig.shape[0] // h, -1)
    patch = ig.abs().sum(dim=-1)
    return _upsample_to_image(patch, img)


# ============================================================================
# Perturbation-based
# ============================================================================

def _patch_grid_for_image(img: Image.Image, n_patches: int) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Build n_patches × n_patches axis-aligned tiles covering the image."""
    W, H = img.size
    pw, ph = W / n_patches, H / n_patches
    boxes: list[tuple[int, int, int, int]] = []
    for y in range(n_patches):
        for x in range(n_patches):
            x1 = int(round(x * pw))
            y1 = int(round(y * ph))
            x2 = int(round((x + 1) * pw))
            y2 = int(round((y + 1) * ph))
            boxes.append((x1, y1, x2, y2))
    return boxes, n_patches, n_patches


def _build_baseline(img: Image.Image, baseline: str) -> Image.Image:
    if baseline == "zero":
        return Image.new(img.mode, img.size, 0)
    if baseline == "mean":
        arr = np.array(img.convert("L"))
        return Image.new(img.mode, img.size, int(arr.mean()))
    if baseline == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=max(1, int(0.2 * min(img.size)))))
    raise ValueError(f"Unknown baseline {baseline!r}")


@torch.inference_mode()
def occlusion(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    n_patches: int = 8,
    baseline: str = "zero",
    batch_size: int = 16,
    positive_only: bool = True,
) -> torch.Tensor:
    """Zeiler & Fergus (2014): single-patch occlusion sensitivity."""
    if img.mode != "L":
        img = img.convert("L")
    base = _build_baseline(img, baseline)
    boxes, sx, sy = _patch_grid_for_image(img, n_patches)
    K = len(boxes)

    # Score the original.
    user_inputs, full_inputs, out0 = teacher_forced_forward(
        adapter, img, question, answer, output_attentions=False, output_hidden_states=False
    )
    ts, te = adapter.target_span(user_inputs, full_inputs)
    target_ids = full_inputs["input_ids"][0, ts:te][None, :, None]

    def _score(images_chunk: list[Image.Image]) -> torch.Tensor:
        from medfocus.concepts.intervention import batched_teacher_forced_logprobs
        lp = batched_teacher_forced_logprobs(adapter, images_chunk, question, answer, batch_size=batch_size)
        return lp.exp().mean(dim=-1)  # geometric-style score per image

    base_arr = np.array(base)
    img_arr = np.array(img)

    chunks: list[Image.Image] = []
    for x1, y1, x2, y2 in boxes:
        cur = img_arr.copy()
        cur[y1:y2, x1:x2] = base_arr[y1:y2, x1:x2]
        chunks.append(Image.fromarray(cur))

    scores = _score(chunks)
    score0 = _score([img])[0]
    deltas = (score0 - scores).clamp_min(0) if positive_only else (score0 - scores)

    grid = deltas.view(sy, sx)
    return _upsample_to_image(grid.float(), img)


@torch.inference_mode()
def rise(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    n_patches: int = 8,
    num_masks: int = 64,
    p_keep: float = 0.5,
    baseline: str = "zero",
    batch_size: int = 16,
) -> torch.Tensor:
    """Petsiuk et al. (2018): Random Input Sampling for Explanation."""
    if img.mode != "L":
        img = img.convert("L")
    boxes, sx, sy = _patch_grid_for_image(img, n_patches)
    K = len(boxes)
    base = _build_baseline(img, baseline)
    base_arr = np.array(base)
    img_arr = np.array(img)
    crops = [(b, img_arr[b[1]:b[3], b[0]:b[2]].copy()) for b in boxes]

    masks = (torch.rand(num_masks, K) < p_keep)
    perturbed: list[Image.Image] = []
    for m in masks:
        cur = base_arr.copy()
        for j in m.nonzero(as_tuple=True)[0].tolist():
            (x1, y1, x2, y2), patch = crops[j]
            cur[y1:y2, x1:x2] = patch
        perturbed.append(Image.fromarray(cur))

    from medfocus.concepts.intervention import batched_teacher_forced_logprobs
    lp = batched_teacher_forced_logprobs(adapter, perturbed, question, answer, batch_size=batch_size)
    scores = lp.exp().mean(dim=-1).float()  # (num_masks,)
    sal = (scores[:, None] * masks.float()).sum(0) / (num_masks * max(p_keep, 1e-8))

    grid = sal.view(sy, sx)
    return _upsample_to_image(grid, img)


# ============================================================================
# Prompting-based
# ============================================================================

_PROMPT_BBOX = (
    "Identify the local evidence in the image that supports the answer, and output "
    "the bounding box coordinates. Provide your answer as a list of bounding boxes "
    "in the format [[x1, y1, x2, y2], ...], where (x1, y1) is the top-left corner "
    "and (x2, y2) is the bottom-right corner of each bounding box."
)
_PROMPT_DESCR = (
    "Identify the local evidence in the image that supports the answer, and output "
    "descriptions of the target objects/regions. Provide your answer as a list of "
    "words or phrases [\"region1\", \"region2\", ...] that concisely describe the "
    "target regions in the image."
)


def prompting(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    max_new_tokens: int = 256,
) -> list[tuple[int, int, int, int]]:
    """LVLM directly emits bounding boxes for the supporting evidence."""
    from medfocus.lvlm.generation import generate_answer
    out = generate_answer(adapter, img, question + "\n" + _PROMPT_BBOX, max_new_tokens=max_new_tokens)
    boxes: list[tuple[int, int, int, int]] = []
    try:
        parsed = eval(out, {"__builtins__": {}})  # numbers-only expression
    except Exception:
        return boxes
    if isinstance(parsed, list):
        for b in parsed:
            try:
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
            except Exception:
                continue
    return boxes


def prompting_medsam(
    adapter: LVLMAdapter,
    img: Image.Image,
    question: str,
    answer: str,
    *,
    medsam,
    detector_model: str = "google/owlvit-base-patch32",
    det_threshold: float = 0.10,
    top_k_per_desc: int = 5,
    max_new_tokens: int = 256,
) -> list[tuple[int, int, int, int]]:
    """LVLM emits region descriptions; an open-vocab detector + MedSAM refine them."""
    from transformers import pipeline as hf_pipeline
    from medfocus.lvlm.generation import generate_answer

    descriptions_str = generate_answer(
        adapter, img, question + "\n" + _PROMPT_DESCR, max_new_tokens=max_new_tokens
    )
    try:
        descriptions = eval(descriptions_str, {"__builtins__": {}})
    except Exception:
        descriptions = []
    if not isinstance(descriptions, list):
        return []

    detector = hf_pipeline(
        "zero-shot-object-detection",
        model=detector_model,
        device=0 if torch.cuda.is_available() else -1,
    )
    img_rgb = img.convert("RGB")
    W, H = img_rgb.size

    candidates: list[tuple[int, int, int, int]] = []
    for desc in descriptions:
        if not isinstance(desc, str) or not desc.strip():
            continue
        dets = detector(img_rgb, candidate_labels=[desc], threshold=det_threshold)
        dets = sorted(dets, key=lambda d: float(d["score"]), reverse=True)[:top_k_per_desc]
        for d in dets:
            b = d["box"]
            x1 = int(np.clip(b["xmin"], 0, W - 1))
            y1 = int(np.clip(b["ymin"], 0, H - 1))
            x2 = int(np.clip(b["xmax"], 0, W - 1))
            y2 = int(np.clip(b["ymax"], 0, H - 1))
            if x2 > x1 and y2 > y1:
                candidates.append((x1, y1, x2, y2))
    return medsam.refine_boxes(img_rgb, candidates)


# ============================================================================
# Method registry
# ============================================================================

#: Maps method-name -> (callable, returns_heatmap_bool).
METHODS: dict[str, tuple] = {
    "attention_head": (attention_head, True),
    "attention_rollout": (attention_rollout, True),
    "lrp": (lrp, True),
    "grad_weighted_attention": (grad_weighted_attention, True),
    "gradcam": (gradcam, True),
    "gradcampp": (gradcampp, True),
    "integrated_gradients": (integrated_gradients, True),
    "occlusion": (occlusion, True),
    "rise": (rise, True),
    "prompting": (prompting, False),
    "prompting_medsam": (prompting_medsam, False),
}
