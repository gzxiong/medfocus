"""Entropic Unbalanced OT via the generalized Sinkhorn algorithm.

The KL-relaxed marginal formulation follows Chizat et al. (2018). We optimize:

    T* = argmin_{T >= 0} <C, T>
                       + lambda_marginal * KL(T 1 || mu_ref)
                       + lambda_marginal * KL(T^T 1 || mu_tgt)
                       + epsilon * H(T)

where H is the entropic regularizer. The marginal-relaxation exponent
`fi = lambda / (lambda + eps)` collapses to balanced OT as lambda -> infinity
and to fully unconstrained marginals as lambda -> 0.
"""

from __future__ import annotations

import numpy as np


def sinkhorn_uot(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    *,
    epsilon: float = 0.05,
    lambda_marginal: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve entropic UOT and return the (unnormalized) transport plan.

    Parameters
    ----------
    a, b : (N,), (M,) source and target marginals.
    C : (N, M) cost matrix.
    epsilon : entropic regularization strength.
    lambda_marginal : KL-relaxation weight on both marginals.
    max_iter, tol : Sinkhorn loop limits.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    a = a / (a.sum() + 1e-18)
    b = b / (b.sum() + 1e-18)

    eps = max(float(epsilon), 1e-12)
    if not np.isfinite(lambda_marginal) or lambda_marginal <= 0:
        raise ValueError("`lambda_marginal` must be a positive finite number.")

    K = np.exp(-C / eps)
    K = np.maximum(K, 1e-300)
    fi = lambda_marginal / (lambda_marginal + eps)  # KL-relaxed scaling exponent in (0, 1)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(max_iter):
        u_prev = u.copy()
        u = (a / (K @ v + 1e-18)) ** fi
        v = (b / (K.T @ u + 1e-18)) ** fi
        if np.linalg.norm(u - u_prev, ord=1) < tol:
            break

    # Total transported mass is meaningful in UOT; do not normalize.
    return (u[:, None] * K) * v[None, :]
