import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torch
import numpy as np
import pydicom

# def visualize_matrix(mtx, source_tokens, target_tokens, flip_xy=True):
#     if not flip_xy:
#         source_tokens, target_tokens = target_tokens, source_tokens
#         mtx = mtx.T
    
#     fig_width = max(10, len(source_tokens) * 0.25)
#     fig_height = max(10, len(target_tokens) * 0.8)

#     plt.figure(figsize=(fig_width, fig_height))

#     # Create heatmap - transposed to flip x and y
#     # We transpose mtx to switch rows and columns
#     sns.heatmap(
#         mtx.T, 
#         xticklabels=source_tokens, 
#         yticklabels=target_tokens, 
#         cmap="viridis", 
#         cbar_kws={'label': 'Transport Mass'}
#     )

#     if not flip_xy:
#         plt.xlabel("Target Tokens", fontsize=12)
#         plt.ylabel("Source Tokens", fontsize=12)
#     else:
#         plt.xlabel("Source Tokens", fontsize=12)
#         plt.ylabel("Target Tokens", fontsize=12)
#     # plt.title("Unbalanced Optimal Transport Plan", fontsize=14)
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.show()


# MAX_SIDE = 2048
# def load_rgb(path, max_side=MAX_SIDE):
#     with Image.open(path) as img:
#         img = img.convert("RGB")
#         w, h = img.size
#         s = max(w, h)
#         if s > max_side:
#             scale = max_side / s
#             img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
#         return img.copy()
def safe_open_image(path):
    try:
        if path.lower().endswith(".dcm") or path.lower().endswith(".dicom"):
            return dicom_to_pil(path, force_8bit=True)
        with Image.open(path) as img:

            mode = img.mode

            # Common case: already 8-bit grayscale
            if mode == "L":
                out = img
            # 16-bit grayscale (very common for medical exports)
            elif mode in ("I;16", "I;16B", "I;16L"):
                arr = np.array(img, dtype=np.uint16)
                # Map to 0..255 using min-max scaling (display-oriented)
                lo, hi = int(arr.min()), int(arr.max())
                if hi == lo:
                    arr8 = np.zeros(arr.shape, dtype=np.uint8)
                else:
                    arr8 = ((arr - lo) * 255.0 / (hi - lo)).astype(np.uint8)
                out = Image.fromarray(arr8, mode="L")
            # 32-bit int or float
            elif mode in ("I", "F"):
                arr = np.array(img, dtype=np.float32)
                lo, hi = float(np.min(arr)), float(np.max(arr))
                if hi == lo:
                    arr8 = np.zeros(arr.shape, dtype=np.uint8)
                else:
                    arr8 = ((arr - lo) * 255.0 / (hi - lo)).astype(np.uint8)
                out = Image.fromarray(arr8, mode="L")
            # Palette, RGB, RGBA, etc. -> convert to L
            else:
                out = img.convert("L")
            
            return out.copy()
        
    except Exception:
        return None

# def generate_answer(model, processor, img, question, max_new_tokens=256, question_suffix=" Answer with a letter, word, phrase, or sentence."):
def generate_answer(model, processor, img, question, max_new_tokens=256, question_suffix=""):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question + question_suffix},
            ],
        }
    ]
    
    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to("cuda")
    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True
        )


    prompt_len = inputs.input_ids.shape[1]
    new_ids = gen.sequences[0, prompt_len:]  # generated token ids only
    generated_text = processor.batch_decode([new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return generated_text

def mask_image_regions(img, boxes):
    img_masked = img.copy()
    img_array = np.array(img_masked)

    for box in boxes:
        x1, y1, x2, y2 = box
        img_array[y1:y2, x1:x2] = 0

    img_masked = Image.fromarray(img_array)
    return img_masked

def keep_image_regions(img, boxes, masked_value=0):
    img_array = np.array(img)
    kept = np.zeros_like(img_array) + masked_value
    for x1, y1, x2, y2 in boxes:
        kept[y1:y2, x1:x2] = img_array[y1:y2, x1:x2]
    return Image.fromarray(kept)

def dicom_to_pil(path, force_8bit=True):
    ds = pydicom.dcmread(path)

    # Get pixel data as float for transforms
    arr = ds.pixel_array.astype(np.float32)

    # Apply rescale (CT, etc.) if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    # Handle MONOCHROME1 (inverted grayscale)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    # Apply windowing if available (common for CT/MR viewers)
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        # WindowCenter/Width can be stored as lists
        if isinstance(wc, (list, pydicom.multival.MultiValue)):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if isinstance(ww, (list, pydicom.multival.MultiValue)):
            ww = float(ww[0])
        else:
            ww = float(ww)

        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
        arr = np.clip(arr, lo, hi)
    else:
        # Otherwise clip to min/max
        arr = np.clip(arr, arr.min(), arr.max())

    # Normalize to 0..255
    arr -= arr.min()
    denom = arr.max() if arr.max() != 0 else 1.0
    arr = arr / denom

    if force_8bit:
        out = (arr * 255.0).round().astype(np.uint8)
        return Image.fromarray(out, mode="L")
    else:
        out = (arr * 65535.0).round().astype(np.uint16)
        return Image.fromarray(out, mode="I;16")



def resize_pad(img: Image.Image, width: int, fill: int = 0) -> Image.Image:
    """
    Resize a PIL image so its longest side becomes `width`, then pad to (width, width)
    with black pixels, keeping aspect ratio.
    """
    if width <= 0:
        raise ValueError("width must be a positive integer")

    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("image has invalid size")

    # Scale so the longest side == width (aspect preserved)
    scale = width / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # Pad to square (width x width), centered
    delta_w = width - new_w
    delta_h = width - new_h
    left = delta_w // 2
    right = delta_w - left
    top = delta_h // 2
    bottom = delta_h - top

    padded = ImageOps.expand(resized, border=(left, top, right, bottom), fill=fill)
    return padded


def visualize_heatmap(img, heatmap, alpha=0.45):

    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()

    img = np.array(img)
    heatmap = np.array(heatmap)

    # ---- normalize image ----
    if img.ndim == 2:  # grayscale -> RGB
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    # ---- normalize / resize heatmap ----
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be (H,W), got {heatmap.shape}")
    hm = heatmap.astype(np.float32)
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)

    if hm.shape[:2] != img.shape[:2]:
        raise ValueError(f"heatmap shape {hm.shape} must match image shape {img.shape[:2]}")

    # ---- plot ----
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(hm, alpha=alpha)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def heatmap_to_bboxes_quantile(
    heatmap,
    q=0.9,
    min_area=16,
    max_boxes=10,
    connectivity=8,
    remove_zeros=True
):
    """
    Convert a heatmap (H,W) into bounding boxes by thresholding at a quantile
    and extracting connected components.

    Args:
      heatmap: np.ndarray | torch.Tensor, shape (H,W)
      q: quantile in (0,1); threshold = np.quantile(heatmap, q)
      min_area: discard components with fewer than this many pixels
      max_boxes: keep top-K boxes by component mean heat
      connectivity: 4 or 8 for connected-component neighborhood

    Returns:
      boxes: list of dicts with:
        - bbox: (x1,y1,x2,y2) inclusive pixel coords
        - score: mean heat inside component
        - area: pixel count
        - thr: threshold used
    """
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be (H,W), got {hm.shape}")

    # normalize to [0,1] for stable quantiles
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)

    # only consider non-zero values for quantile computation
    if remove_zeros:
        vals = hm[hm > 0]
    else:
        vals = hm.flatten()
    if len(vals) == 0:
        thr = 0.0
    else:
        thr = float(np.quantile(vals, q))
    mask = hm >= thr
    H, W = hm.shape

    # connected components (simple BFS/stack)
    if connectivity == 8:
        nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    elif connectivity == 4:
        nbrs = [(-1,0),(0,-1),(0,1),(1,0)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    visited = np.zeros((H, W), dtype=bool)
    comps = []

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            xs, ys = [], []
            ssum = 0.0
            cnt = 0

            while stack:
                cy, cx = stack.pop()
                xs.append(cx); ys.append(cy)
                v = float(hm[cy, cx])
                ssum += v
                cnt += 1
                for dy, dx in nbrs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if cnt < int(min_area):
                continue

            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            comps.append({
                "bbox": (x1, y1, x2, y2),
                "score": ssum / cnt,
                "area": cnt,
                "thr": thr,
            })

    # rank by score (mean heat), keep top-K
    comps.sort(key=lambda d: d["score"], reverse=True)
    return comps[: int(max_boxes)]

def bboxes_rescale(bboxes, orig_size, heatmap_size):
    orig_w, orig_h = orig_size
    heatmap_w, heatmap_h = heatmap_size
    
    scale_ = max(orig_w / heatmap_w, orig_h / heatmap_h)
    delta_w = (heatmap_w - orig_w / scale_) / 2
    delta_h = (heatmap_h - orig_h / scale_) / 2
    rescaled_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1_rescaled = min(max(0, round((x1 - delta_w) * scale_)), orig_w - 1)
        y1_rescaled = min(max(0, round((y1 - delta_h) * scale_)), orig_h - 1)
        x2_rescaled = min(max(0, round((x2 - delta_w) * scale_)), orig_w - 1)
        y2_rescaled = min(max(0, round((y2 - delta_h) * scale_)), orig_h - 1)
        if x2_rescaled <= x1_rescaled or y2_rescaled <= y1_rescaled:
            continue
        rescaled_bboxes.append([x1_rescaled, y1_rescaled, x2_rescaled, y2_rescaled])
    return rescaled_bboxes

def eval_union_boxes(preds, gts, inclusive_xyxy=False):
    """
    preds, gts: list[list[(x1,y1,x2,y2)]], evaluated per item by UNION region overlap.
    Returns mean IoU / precision / recall / F1.
    """
    def norm(bs):
        if not bs: return []
        out = []
        for x1,y1,x2,y2 in bs:
            x1,y1,x2,y2 = map(float,(x1,y1,x2,y2))
            if inclusive_xyxy: x2,y2 = x2+1, y2+1
            if x2>x1 and y2>y1: out.append((x1,y1,x2,y2))
        return out

    def uarea(rs):  # exact union area via coord compression
        if not rs: return 0.0
        xs = sorted({x for a in rs for x in (a[0],a[2])})
        ys = sorted({y for a in rs for y in (a[1],a[3])})
        xi = {x:i for i,x in enumerate(xs)}; yi = {y:i for i,y in enumerate(ys)}
        m = np.zeros((len(xs)-1, len(ys)-1), bool)
        for x1,y1,x2,y2 in rs: m[xi[x1]:xi[x2], yi[y1]:yi[y2]] = True
        return float((m * np.diff(xs)[:,None] * np.diff(ys)[None,:]).sum())

    def iarea(a,b):
        return uarea([(max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2))
                      for ax1,ay1,ax2,ay2 in a for bx1,by1,bx2,by2 in b
                      if min(ax2,bx2) > max(ax1,bx1) and min(ay2,by2) > max(ay1,by1)])

    ious = []; precs = []; recs = []; f1s = []
    for p,g in zip(preds, gts):
        p,g = norm(p), norm(g)
        ap, ag = uarea(p), uarea(g)
        ai = iarea(p,g)
        au = ap + ag - ai
        iou  = 1.0 if au==0 else ai/au
        prec = 1.0 if ap==0 and ag==0 else (0.0 if ap==0 else ai/ap)
        rec  = 1.0 if ap==0 and ag==0 else (0.0 if ag==0 else ai/ag)
        f1   = 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)
        ious.append(iou); precs.append(prec); recs.append(rec); f1s.append(f1)

    return {
        "iou": ious,
        "precision": precs,
        "recall": recs,
        "f1": f1s
    }

def get_inputs_outputs(model, processor, image, input, output):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": input},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": output},
            ]
        }
    ]
    user_inputs = processor.apply_chat_template(
        messages[:1], 
        tokenize=True, 
        add_generation_prompt=True, 
        return_dict=True, 
        return_tensors="pt"
    )
    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=False, 
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            return_dict=True,
            output_attentions=True,
            output_scores=True,
            output_hidden_states=True,
        )
    return user_inputs, inputs, outputs

def normalize_piece(t):
    return t.replace("▁", "").replace("Ġ", "").strip().lower()