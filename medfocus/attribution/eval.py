"""Spatial attribution evaluation: heatmap → bbox conversion + IoU/F1/Prec/Recall.

Pixel-level saliency maps are converted to bounding boxes via a uniform
quantile-thresholding procedure (paper Sec. 3.3); per-sample metrics are then
computed against expert-annotated ground-truth boxes via union-region overlap.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ----------------------------- heatmap → boxes ------------------------------

def heatmap_to_bboxes_quantile(
    heatmap,
    *,
    q: float = 0.9,
    min_area: int = 16,
    max_boxes: int = 10,
    connectivity: int = 8,
    remove_zeros: bool = True,
) -> list[dict]:
    """Threshold + connected-components → list of bbox dicts ranked by mean heat."""
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach().cpu().numpy()
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be (H,W), got {hm.shape}")
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)

    vals = hm[hm > 0] if remove_zeros else hm.flatten()
    thr = float(np.quantile(vals, q)) if len(vals) else 0.0
    mask = hm >= thr
    H, W = hm.shape

    if connectivity == 8:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    elif connectivity == 4:
        nbrs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    visited = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            xs, ys = [], []
            ssum, cnt = 0.0, 0
            while stack:
                cy, cx = stack.pop()
                xs.append(cx)
                ys.append(cy)
                ssum += float(hm[cy, cx])
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
            comps.append({"bbox": (x1, y1, x2, y2), "score": ssum / cnt, "area": cnt, "thr": thr})

    comps.sort(key=lambda d: d["score"], reverse=True)
    return comps[: int(max_boxes)]


def bboxes_rescale(
    bboxes: Iterable[Sequence[int]],
    orig_size: tuple[int, int],
    heatmap_size: tuple[int, int],
) -> list[list[int]]:
    """Map boxes from (heatmap_w, heatmap_h) coordinates back to (orig_w, orig_h).

    Inverts the resize-and-pad transform applied to the input image.
    """
    orig_w, orig_h = orig_size
    heatmap_w, heatmap_h = heatmap_size
    scale_ = max(orig_w / heatmap_w, orig_h / heatmap_h)
    delta_w = (heatmap_w - orig_w / scale_) / 2
    delta_h = (heatmap_h - orig_h / scale_) / 2
    out = []
    for x1, y1, x2, y2 in bboxes:
        x1r = min(max(0, round((x1 - delta_w) * scale_)), orig_w - 1)
        y1r = min(max(0, round((y1 - delta_h) * scale_)), orig_h - 1)
        x2r = min(max(0, round((x2 - delta_w) * scale_)), orig_w - 1)
        y2r = min(max(0, round((y2 - delta_h) * scale_)), orig_h - 1)
        if x2r > x1r and y2r > y1r:
            out.append([x1r, y1r, x2r, y2r])
    return out


# ----------------------------- IoU / F1 / Prec / Recall ---------------------

def eval_union_boxes(
    preds: list[list[Sequence[float]]],
    gts: list[list[Sequence[float]]],
    *,
    inclusive_xyxy: bool = False,
) -> dict[str, list[float]]:
    """Per-sample union-region IoU/precision/recall/F1 (matches paper protocol)."""
    def norm(bs):
        out = []
        for x1, y1, x2, y2 in bs:
            x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
            if inclusive_xyxy:
                x2, y2 = x2 + 1, y2 + 1
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
        return out

    def uarea(rs):
        if not rs:
            return 0.0
        xs = sorted({x for a in rs for x in (a[0], a[2])})
        ys = sorted({y for a in rs for y in (a[1], a[3])})
        xi = {x: i for i, x in enumerate(xs)}
        yi = {y: i for i, y in enumerate(ys)}
        m = np.zeros((len(xs) - 1, len(ys) - 1), bool)
        for x1, y1, x2, y2 in rs:
            m[xi[x1]: xi[x2], yi[y1]: yi[y2]] = True
        return float((m * np.diff(xs)[:, None] * np.diff(ys)[None, :]).sum())

    def iarea(a, b):
        return uarea([
            (max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2))
            for ax1, ay1, ax2, ay2 in a for bx1, by1, bx2, by2 in b
            if min(ax2, bx2) > max(ax1, bx1) and min(ay2, by2) > max(ay1, by1)
        ])

    ious, precs, recs, f1s = [], [], [], []
    for p, g in zip(preds, gts):
        p, g = norm(p), norm(g)
        ap, ag = uarea(p), uarea(g)
        ai = iarea(p, g)
        au = ap + ag - ai
        iou = 1.0 if au == 0 else ai / au
        prec = 1.0 if ap == 0 and ag == 0 else (0.0 if ap == 0 else ai / ap)
        rec = 1.0 if ap == 0 and ag == 0 else (0.0 if ag == 0 else ai / ag)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        ious.append(iou)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    return {"iou": ious, "precision": precs, "recall": recs, "f1": f1s}


def compute_metrics_table(
    results: list[dict],
    *,
    pred_key: str = "pred_boxes",
    gt_key: str = "gt_boxes",
    method_key: str = "method",
    dataset_key: str = "dataset",
) -> pd.DataFrame:
    """Aggregate per-sample results into a (method, dataset) -> mean-metrics table."""
    rows: list[dict] = []
    for (method, dataset), grp in pd.DataFrame(results).groupby([method_key, dataset_key]):
        m = eval_union_boxes(grp[pred_key].tolist(), grp[gt_key].tolist())
        rows.append({
            "method": method,
            "dataset": dataset,
            "iou": np.mean(m["iou"]) * 100,
            "f1": np.mean(m["f1"]) * 100,
            "precision": np.mean(m["precision"]) * 100,
            "recall": np.mean(m["recall"]) * 100,
            "n": len(grp),
        })
    return pd.DataFrame(rows)
