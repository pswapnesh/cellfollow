"""A cell as a Gaussian mixture: one Gaussian on its centre and on each eigen point, and the outline of that mixture."""

import cv2
import numpy as np


def point_covariances(pts, k_sigma=0.6, k_width=0.25, n_near=2, min_sigma=1.5):
    """Points (..., P, 2) -> covariance (..., P, 2, 2) from the scatter of each point's nearest points."""
    P = pts.shape[-2]
    d = np.linalg.norm(pts[..., :, None, :] - pts[..., None, :, :], axis=-1)
    d[..., np.arange(P), np.arange(P)] = np.inf
    near = np.argsort(d, -1)[..., :n_near]
    v = np.take_along_axis(pts[..., None, :, :], near[..., None], -2) - pts[..., None, :]
    C = np.einsum("...ni,...nj->...ij", v, v) / n_near
    width = np.maximum(k_width * d.min(-1), min_sigma)
    return k_sigma ** 2 * C + width[..., None, None] ** 2 * np.eye(2)


def outline(points, shape, level=0.5, step=2):
    """Closed outline (N, 2) x, y of sum_i exp(-0.5 d_i^2) = level for points (P, 2) y, x, shape (P, 2, 2); None if empty."""
    r = 3 * np.sqrt(np.linalg.eigvalsh(shape)[:, -1])
    lo = np.floor((points - r[:, None]).min(0)).astype(int)
    hi = np.ceil((points + r[:, None]).max(0)).astype(int)
    ys, xs = np.arange(lo[0], hi[0] + 1, step), np.arange(lo[1], hi[1] + 1, step)
    inv = np.linalg.inv(shape)
    dy, dx = ys[None, :, None] - points[:, 0, None, None], xs[None, None, :] - points[:, 1, None, None]
    d2 = inv[:, 0, 0, None, None] * dy ** 2 + 2 * inv[:, 0, 1, None, None] * dy * dx + inv[:, 1, 1, None, None] * dx ** 2
    cs, _ = cv2.findContours((np.exp(-0.5 * d2).sum(0) >= level).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea)[:, 0] * step + [lo[1], lo[0]] if cs else None
