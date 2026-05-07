import numpy as np
from scipy.ndimage import zoom


def _to_gray_float(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    img = img.astype(np.float32)
    m = float(img.min())
    if m < 0:
        img = img - m
    return img


def _norm01(img, eps=1e-12):
    x = _to_gray_float(img)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def _downsample(img, factor, order=1):
    if factor <= 1:
        return img
    return zoom(img, zoom=(1.0 / factor, 1.0 / factor), order=order)


def _build_support(img, threshold=0.02, max_points=4000):
    coords = np.argwhere(img > threshold).astype(np.float64)  # (N,2): y,x
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


def _pdist2(X, Y):
    X2 = (X * X).sum(1, keepdims=True)
    Y2 = (Y * Y).sum(1, keepdims=True).T
    C = X2 + Y2 - 2.0 * (X @ Y.T)
    np.maximum(C, 0.0, out=C)
    return C


def _sinkhorn_uot(a, b, C, epsilon=0.03, reg_m=1.0, max_iter=500, tol=1e-6):
    """
    Entropic UOT with KL-relaxed marginals (generalized Sinkhorn).
    - epsilon: entropy regularization (on normalized cost)
    - reg_m:   marginal relaxation strength (larger => closer to balanced OT)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    a = a / (a.sum() + 1e-18)
    b = b / (b.sum() + 1e-18)

    eps = max(float(epsilon), 1e-12)
    reg_m = float(reg_m)
    if not np.isfinite(reg_m) or reg_m <= 0:
        raise ValueError("`reg_m` must be a positive finite number for UOT.")

    K = np.exp(-C / eps)
    K = np.maximum(K, 1e-300)

    # KL-relaxed scaling exponent
    fi = reg_m / (reg_m + eps)  # (0,1)
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(max_iter):
        u_prev = u.copy()
        u = (a / (K @ v + 1e-18)) ** fi
        v = (b / (K.T @ u + 1e-18)) ** fi
        if np.linalg.norm(u - u_prev, ord=1) < tol:
            break

    P = (u[:, None] * K) * v[None, :]
    return P  # NOTE: not normalized (total transported mass is meaningful in UOT)


def compute_ot_mapping_uot(
    img1, img2, *,
    downsample=4,
    threshold=0.02,
    max_points=4000,
    epsilon=0.03,
    reg_m=1.0,
    max_iter=500,
    tol=1e-6,
):
    """
    Compute UOT mapping from img1 -> img2.
    Returns barycentric mapping and transport plan.
    """
    x1 = _norm01(img1)
    x2 = _norm01(img2)

    x1_ds = _downsample(x1, downsample, order=1)
    x2_ds = _downsample(x2, downsample, order=1)

    Xs_ds, a = _build_support(x1_ds, threshold=threshold, max_points=max_points)
    Xt_ds, b = _build_support(x2_ds, threshold=threshold, max_points=max_points)

    C = _pdist2(Xs_ds, Xt_ds)
    scale = np.percentile(C, 90)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(C.max()), 1.0)
    Cn = C / scale

    P = _sinkhorn_uot(a, b, Cn, epsilon=epsilon, reg_m=reg_m, max_iter=max_iter, tol=tol)

    row_mass = P.sum(axis=1, keepdims=True)
    bary_ds = np.empty_like(Xs_ds)
    valid = row_mass[:, 0] > 1e-18
    bary_ds[valid] = (P[valid] @ Xt_ds) / row_mass[valid]
    bary_ds[~valid] = Xs_ds[~valid]  # fallback if no mass transported

    cost_ds = float((P * C).sum())
    cost = cost_ds * (float(downsample) ** 2)
    return {
        "source_coords_ds": Xs_ds,                 # (N,2) in downsampled img1
        "target_coords_ds": Xt_ds,                 # (M,2) in downsampled img2
        "source_coords": Xs_ds * float(downsample),
        "barycentric_map_ds": bary_ds,             # (N,2) in downsampled img2
        "barycentric_map": bary_ds * float(downsample),
        "source_weights": a,
        "target_weights": b,
        "transport_plan": P,                       # UOT plan (not normalized)
        "img1_ds_shape": x1_ds.shape,
        "img2_ds_shape": x2_ds.shape,
        "img1_shape": x1.shape,
        "img2_shape": x2.shape,
        "downsample": int(downsample),
        "threshold": float(threshold),
        "epsilon": float(epsilon),
        "reg_m": float(reg_m),
        "cost_scale": float(scale),
        "transported_mass_total": float(P.sum()),
        "cost_ds": cost_ds,
        "cost": cost,
    }


def _rasterize(coords, mass, shape):
    H, W = shape
    out = np.zeros((H, W), dtype=np.float64)
    if coords.size == 0:
        return out
    yy = np.rint(coords[:, 0]).astype(np.int64)
    xx = np.rint(coords[:, 1]).astype(np.int64)
    ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    np.add.at(out, (yy[ok], xx[ok]), mass[ok])
    return out


def map_mask_through_ot(mask_img1, ot_result, *, output_binary=True, mass_quantile=0.95):
    """
    Map a mask on img1 into img2 using the UOT/OT plan in `ot_result`.
    """
    P = ot_result["transport_plan"]          # (N,M)
    Xs_ds = ot_result["source_coords_ds"]    # (N,2)
    Xs = ot_result["source_coords"]          # (N,2)
    Xt_ds = ot_result["target_coords_ds"]    # (M,2)
    bary = ot_result["barycentric_map"]      # (N,2)
    a = ot_result["source_weights"]
    ds = int(ot_result["downsample"])
    img2_ds_shape = tuple(ot_result["img2_ds_shape"])
    img2_shape = tuple(ot_result["img2_shape"])

    mask = np.asarray(mask_img1)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0)

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
            "masked_source_coords": np.zeros((0, 2), dtype=np.float64),
            "masked_mapped_coords": np.zeros((0, 2), dtype=np.float64),
            "masked_source_weights": np.zeros((0,), dtype=np.float64),
            "target_support_mass_from_mask": np.zeros((P.shape[1],), dtype=np.float64),
            "target_heatmap_ds": np.zeros(img2_ds_shape, dtype=np.float64),
            "target_heatmap": np.zeros(img2_shape, dtype=np.float64),
        }
        if output_binary:
            out["mapped_binary_mask"] = np.zeros(img2_shape, dtype=np.uint8)
        return out

    # For UOT this is the actual transported mass from masked source support to each target support point
    target_mass = P[idx].sum(axis=0)
    heat_ds = _rasterize(Xt_ds, target_mass, img2_ds_shape)

    if ds > 1:
        zy = img2_shape[0] / heat_ds.shape[0]
        zx = img2_shape[1] / heat_ds.shape[1]
        heat = zoom(heat_ds, zoom=(zy, zx), order=1)
        heat = heat[:img2_shape[0], :img2_shape[1]]
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
        "masked_source_coords": Xs[idx],        # original img1 coords (y,x)
        "masked_mapped_coords": bary[idx],      # mapped coords in original img2 (y,x)
        "masked_source_weights": a[idx],
        "target_support_mass_from_mask": target_mass,
        "target_heatmap_ds": heat_ds,
        "target_heatmap": heat,
    }

    if output_binary:
        mapped_binary_mask = np.zeros(img2_shape, dtype=np.uint8)
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
            mapped_binary_mask[yy2[ok2], xx2[ok2]] = 1

        out["mapped_binary_mask"] = mapped_binary_mask

    return out