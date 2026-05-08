"""Image-to-image UOT mapping + concept-mask transfer."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom

from medfocus.ot.sinkhorn import sinkhorn_uot


# ----------------------------- helpers --------------------------------------

def _to_gray_float(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float32)
    m = float(img.min())
    if m < 0:
        img = img - m
    return img


def _norm01(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = _to_gray_float(img)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def _downsample(img: np.ndarray, factor: int, order: int = 1) -> np.ndarray:
    if factor <= 1:
        return img
    return zoom(img, zoom=(1.0 / factor, 1.0 / factor), order=order)


def _build_support(img: np.ndarray, threshold: float = 0.02, max_points: int = 4000):
    coords = np.argwhere(img > threshold).astype(np.float64)
    if coords.size == 0:
        raise ValueError("No support points found; lower `threshold`.")
    w = img[coords[:, 0].astype(int), coords[:, 1].astype(int)].astype(np.float64)
    if max_points is not None and coords.shape[0] > max_points:
        idx = np.argpartition(w, -max_points)[-max_points:]
        coords, w = coords[idx], w[idx]
    s = w.sum()
    if s <= 0:
        raise ValueError("Support weights sum to zero.")
    return coords, w / s


def _pdist2(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X2 = (X * X).sum(1, keepdims=True)
    Y2 = (Y * Y).sum(1, keepdims=True).T
    C = X2 + Y2 - 2.0 * (X @ Y.T)
    np.maximum(C, 0.0, out=C)
    return C


def _rasterize(coords: np.ndarray, mass: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    out = np.zeros((H, W), dtype=np.float64)
    if coords.size == 0:
        return out
    yy = np.rint(coords[:, 0]).astype(np.int64)
    xx = np.rint(coords[:, 1]).astype(np.int64)
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    np.add.at(out, (yy[ok], xx[ok]), mass[ok])
    return out


# ------------------------------ public API ----------------------------------

def compute_ot_mapping_uot(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    downsample: int = 4,
    threshold: float = 0.02,
    max_points: int = 4000,
    epsilon: float = 0.05,
    lambda_marginal: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Compute UOT mapping from `img1` -> `img2` and a barycentric map."""
    x1, x2 = _norm01(img1), _norm01(img2)
    x1_ds = _downsample(x1, downsample, order=1)
    x2_ds = _downsample(x2, downsample, order=1)

    Xs_ds, a = _build_support(x1_ds, threshold=threshold, max_points=max_points)
    Xt_ds, b = _build_support(x2_ds, threshold=threshold, max_points=max_points)

    C = _pdist2(Xs_ds, Xt_ds)
    scale = np.percentile(C, 90)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(C.max()), 1.0)
    Cn = C / scale

    P = sinkhorn_uot(
        a, b, Cn,
        epsilon=epsilon,
        lambda_marginal=lambda_marginal,
        max_iter=max_iter,
        tol=tol,
    )

    row_mass = P.sum(axis=1, keepdims=True)
    bary_ds = np.empty_like(Xs_ds)
    valid = row_mass[:, 0] > 1e-18
    bary_ds[valid] = (P[valid] @ Xt_ds) / row_mass[valid]
    bary_ds[~valid] = Xs_ds[~valid]

    cost_ds = float((P * C).sum())
    return {
        "source_coords_ds": Xs_ds,
        "target_coords_ds": Xt_ds,
        "source_coords": Xs_ds * float(downsample),
        "barycentric_map_ds": bary_ds,
        "barycentric_map": bary_ds * float(downsample),
        "source_weights": a,
        "target_weights": b,
        "transport_plan": P,
        "img1_ds_shape": x1_ds.shape,
        "img2_ds_shape": x2_ds.shape,
        "img1_shape": x1.shape,
        "img2_shape": x2.shape,
        "downsample": int(downsample),
        "epsilon": float(epsilon),
        "lambda_marginal": float(lambda_marginal),
        "cost_scale": float(scale),
        "transported_mass_total": float(P.sum()),
        "cost_ds": cost_ds,
        "cost": cost_ds * (float(downsample) ** 2),
    }


def map_mask_through_ot(
    mask_img1: np.ndarray,
    ot_result: dict[str, Any],
    *,
    output_binary: bool = True,
    mass_quantile: float = 0.75,
) -> dict[str, Any]:
    """Push a binary mask on `img1` through the UOT plan into `img2` space.

    Returns a dict with `target_heatmap` (continuous mass map) and, when
    `output_binary=True`, `mapped_binary_mask` (the smallest pixel set whose
    cumulative mass covers `mass_quantile` of the total).
    """
    P = ot_result["transport_plan"]
    Xs_ds = ot_result["source_coords_ds"]
    Xs = ot_result["source_coords"]
    Xt_ds = ot_result["target_coords_ds"]
    bary = ot_result["barycentric_map"]
    a = ot_result["source_weights"]
    ds = int(ot_result["downsample"])
    img2_ds_shape = tuple(ot_result["img2_ds_shape"])
    img2_shape = tuple(ot_result["img2_shape"])

    mask = np.asarray(mask_img1)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0
    mask_ds = _downsample(mask.astype(np.float32), ds, order=0) > 0.5

    yy = np.rint(Xs_ds[:, 0]).astype(np.int64)
    xx = np.rint(Xs_ds[:, 1]).astype(np.int64)
    H1, W1 = mask_ds.shape
    valid = (yy >= 0) & (yy < H1) & (xx >= 0) & (xx < W1)
    in_mask = np.zeros(Xs_ds.shape[0], dtype=bool)
    in_mask[valid] = mask_ds[yy[valid], xx[valid]]
    idx = np.where(in_mask)[0]

    if idx.size == 0:
        out = {
            "masked_source_indices": idx,
            "masked_source_coords": np.zeros((0, 2)),
            "masked_mapped_coords": np.zeros((0, 2)),
            "masked_source_weights": np.zeros((0,)),
            "target_support_mass_from_mask": np.zeros((P.shape[1],)),
            "target_heatmap_ds": np.zeros(img2_ds_shape),
            "target_heatmap": np.zeros(img2_shape),
        }
        if output_binary:
            out["mapped_binary_mask"] = np.zeros(img2_shape, dtype=np.uint8)
        return out

    target_mass = P[idx].sum(axis=0)
    heat_ds = _rasterize(Xt_ds, target_mass, img2_ds_shape)

    if ds > 1:
        zy = img2_shape[0] / heat_ds.shape[0]
        zx = img2_shape[1] / heat_ds.shape[1]
        heat = zoom(heat_ds, zoom=(zy, zx), order=1)
        heat = heat[: img2_shape[0], : img2_shape[1]]
        if heat.shape != img2_shape:
            tmp = np.zeros(img2_shape, dtype=np.float64)
            h = min(img2_shape[0], heat.shape[0])
            w = min(img2_shape[1], heat.shape[1])
            tmp[:h, :w] = heat[:h, :w]
            heat = tmp
    else:
        heat = heat_ds.copy()

    out = {
        "masked_source_indices": idx,
        "masked_source_coords": Xs[idx],
        "masked_mapped_coords": bary[idx],
        "masked_source_weights": a[idx],
        "target_support_mass_from_mask": target_mass,
        "target_heatmap_ds": heat_ds,
        "target_heatmap": heat,
    }

    if output_binary:
        mapped = np.zeros(img2_shape, dtype=np.uint8)
        total = float(target_mass.sum())
        if total > 0:
            order = np.argsort(target_mass)[::-1]
            csum = np.cumsum(target_mass[order])
            k = np.searchsorted(csum, float(mass_quantile) * total, side="left") + 1
            keep = order[:k]
            Xt = Xt_ds * float(ds)
            pts = Xt[keep]
            yy2 = np.rint(pts[:, 0]).astype(np.int64)
            xx2 = np.rint(pts[:, 1]).astype(np.int64)
            ok2 = (yy2 >= 0) & (yy2 < img2_shape[0]) & (xx2 >= 0) & (xx2 < img2_shape[1])
            mapped[yy2[ok2], xx2[ok2]] = 1
        out["mapped_binary_mask"] = mapped

    return out
