"""NumPy post-processing of the exported network's outputs (export_fast.FastNet), the same for every backend: the
detections, fg masks and P(real) that CellTracker.detect computes with torch, made cheap enough for a CPU.

What changed against the torch originals (results agree, see check_fast.py):
    percentiles      exact, from the uint8 histogram instead of a sort
    flow covariance  the sigma 5 blur of fg and the flow tensor is only evaluated at the points that read it, not over
                     the whole frame
    fill_cuts        a cut pixel takes the nearest labelled pixel in its (2 px + 1)^2 window instead of a distance
                     transform over the whole frame
"""

import math

import cv2
import numpy as np

from model import POINT_STRIDE
from .cell_probability import logit_score
from .gmm import point_covariances


def percentiles_u8(gray, q=(1.0, 99.0)):
    """np.percentile(gray, q) (linear interpolation) of a uint8 image, from its histogram."""
    c = np.cumsum(cv2.calcHist([gray], [0], None, [256], [0, 256])[:, 0].astype(np.int64))
    out = []
    for qi in q:
        pos = qi / 100 * (c[-1] - 1)
        i = int(math.floor(pos))
        lo, hi = np.searchsorted(c, i, "right"), np.searchsorted(c, min(i + 1, c[-1] - 1), "right")
        out.append(lo + (pos - i) * (hi - lo))
    return out


def lohi(gray, invert=False):
    """Bounds for FastNet: x = clip((gray - lo) / (hi - lo), 0, 1), as cell_tracker.normalize (all zeros if flat)."""
    lo, hi = percentiles_u8(gray)
    if hi <= lo:
        lo, hi = (-2.0, -1.0) if invert else (256.0, 257.0)   # all ones / all zeros, as 1 - normalize / normalize
        return np.array([lo, hi], np.float32)
    return np.array([hi, lo] if invert else [lo, hi], np.float32)


def tril_cholesky(raw, d):
    """raw (N, d (d + 1) / 2) -> L (N, d, d): exp on the diagonal, as model.err_covariance."""
    rows, cols = np.tril_indices(d)
    L = np.zeros((len(raw), d, d), np.float32)
    L[:, rows, cols] = raw
    i = np.arange(d)
    L[:, i, i] = np.exp(np.clip(raw[:, rows == cols], -4, 4))
    return L * POINT_STRIDE


def decode(grid, nc, k, thr=0.05):
    """grid (C, h, w) -> as model.decode_points, in numpy float32."""
    gy, gx = np.nonzero(grid[0] > thr)
    g = grid[:, gy, gx].T.astype(np.float32)                        # (N, C)
    i = 1
    cls_prob, i = g[:, i:i + nc], i + nc
    off, i = g[:, i:i + 2], i + 2
    kpt, i = g[:, i:i + 2 * k], i + 2 * k
    P = k + 1
    sh, i = g[:, i:i + 3 * P].reshape(-1, P, 3), i + 3 * P
    err = g[:, i:]
    yx = (np.stack([gy, gx], 1) + off) * POINT_STRIDE
    kpts = kpt.reshape(-1, k, 2) * POINT_STRIDE + yx[:, None]
    L = np.zeros((len(g), P, 2, 2), np.float32)
    L[..., 0, 0], L[..., 1, 0], L[..., 1, 1] = np.exp(np.clip(sh[..., 0], -4, 4)), sh[..., 1], np.exp(np.clip(sh[..., 2], -4, 4))
    L *= POINT_STRIDE
    Le = tril_cholesky(err, 2 * P)
    return {"yx": yx, "score": g[:, 0], "cls_prob": cls_prob, "kpts": kpts, "shape": L @ np.swapaxes(L, -1, -2),
            "R": Le @ np.swapaxes(Le, -1, -2)}


def gauss1d(sigma):
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float32) / sigma) ** 2)
    return k / k.sum(), r


def flow_covariances(pts, flow, fg, sigma=5.0, k_sigma=0.6, k_width=0.25, min_sigma=1.5, min_coherence=0.3,
                     min_mass=0.2):
    """As cell_probability.flow_covariances, numpy: pts (N, P, 2), flow (2, H, W), fg (H, W) -> (N, P, 2, 2)."""
    N, P = pts.shape[:2]
    S0 = point_covariances(pts.astype(np.float64), k_sigma, k_width).astype(np.float32)
    H, W = fg.shape
    y = np.clip(np.round(pts[..., 0]).astype(np.int64), 0, H - 1).ravel()
    x = np.clip(np.round(pts[..., 1]).astype(np.int64), 0, W - 1).ravel()
    k, r = gauss1d(sigma)
    # windows of fg and the unit flow around each point, then the blur of fg, fg f0^2, fg f0 f1, fg f1^2 there
    dy = np.arange(-r, r + 1)
    yy, xx = y[:, None] + dy, x[:, None] + dy
    wy, wx = k * ((yy >= 0) & (yy < H)), k * ((xx >= 0) & (xx < W))
    idx = np.clip(yy, 0, H - 1)[:, :, None] * W + np.clip(xx, 0, W - 1)[:, None, :]
    g, f0, f1 = np.take(fg.ravel(), idx), np.take(flow[0].ravel(), idx), np.take(flow[1].ravel(), idx)
    gw = g * (wy[:, :, None] * wx[:, None, :])
    mass = gw.sum((1, 2))
    gw = (gw / np.maximum(f0 * f0 + f1 * f1, 1e-12) / np.maximum(mass, 1e-6)[:, None, None]).reshape(len(y), -1)
    f0, f1 = f0.reshape(len(y), -1), f1.reshape(len(y), -1)
    a, b, c = (np.einsum("ni,ni->n", gw, f0 * f0), np.einsum("ni,ni->n", gw, f0 * f1),
               np.einsum("ni,ni->n", gw, f1 * f1))
    theta = 0.5 * np.arctan2(2 * b, a - c)
    e = np.stack([-np.sin(theta), np.cos(theta)], -1).reshape(N, P, 2)
    coherence = (np.sqrt((a - c) ** 2 + 4 * b ** 2) / np.maximum(a + c, 1e-9)).reshape(N, P)
    d = np.linalg.norm(pts[:, :, None] - pts[:, None], axis=-1) + np.eye(P, dtype=np.float32) * 1e9
    w2 = np.maximum(k_width * d.min(-1), min_sigma) ** 2
    l2 = np.maximum(S0[..., 0, 0] + S0[..., 1, 1] - 2 * w2, 0)
    S = l2[..., None, None] * (e[..., :, None] * e[..., None, :]) + w2[..., None, None] * np.eye(2, dtype=np.float32)
    used = (coherence >= min_coherence) & (mass.reshape(N, P) >= min_mass)
    return np.where(used[..., None, None], S, S0)


def candidate_scores(p, S, score, fg, rho=1.0, n=16, tau=0.5, stride=4):
    """As cell_probability.candidate_scores, numpy."""
    H, W = fg.shape
    i = np.arange(n, dtype=np.float32)
    r, t = rho * np.sqrt((i + 0.5) / n), i * 2.399963
    z = np.stack([r * np.sin(t), r * np.cos(t)], 1)                     # (n, 2)
    x = p[:, :, None] + np.einsum("npij,sj->npsi", np.linalg.cholesky(S), z)
    yi = np.clip(np.round(x[..., 0]).astype(np.int64), 0, H - 1)
    xi = np.clip(np.round(x[..., 1]).astype(np.int64), 0, W - 1)
    vals = fg[yi, xi]
    ok = vals.mean((1, 2)) >= tau
    cell = (yi // stride) * (W // stride + 1) + xi // stride
    rank = np.argsort(np.argsort(score, kind="stable"), kind="stable") + 1
    top = np.zeros(int(cell.max()) + 1, np.int64)
    if ok.any():
        np.maximum.at(top, cell[ok].ravel(), np.broadcast_to(rank[ok][:, None, None], cell[ok].shape).ravel())
    return (vals * (top[cell] == rank[:, None, None])).mean((1, 2))


def fg_probability(points, score, fg, flow, params):
    """As cell_probability.fg_probability, numpy."""
    covered = candidate_scores(points, flow_covariances(points, flow, fg), score, fg, params["rho"],
                               stride=params["stride"])
    z = (np.c_[logit_score(score.astype(np.float64)), covered.astype(np.float64)] - np.array(params["mu"])) / np.array(params["sd"])
    w = np.array(params["w"])
    return 1 / (1 + np.exp(-(z @ w[:2] + w[2])))


def window_offsets(px):
    """Offsets of the (2 px + 1)^2 window, nearest first."""
    d = np.arange(-px, px + 1)
    off = np.stack(np.meshgrid(d, d, indexing="ij"), -1).reshape(-1, 2)
    off = off[(off != 0).any(1)]
    return off[np.argsort((off ** 2).sum(1), kind="stable")]


def fill_cuts(labels, px):
    """As masks.fill_cuts: unlabelled pixels with two different masks within px take the nearest mask."""
    if px <= 0:
        return labels
    has = cv2.compare(labels, 0, cv2.CMP_GT)                                # uint8 0 / 255
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    lab = labels.astype(np.float32)
    big = np.float32(2 ** 24)
    hi = cv2.dilate(lab, kernel)                                            # largest label in the window
    lo = big - cv2.dilate(np.where(has > 0, big - lab, np.float32(0)), kernel)   # smallest, big if none
    cut = cv2.findNonZero(cv2.bitwise_and(cv2.compare(lo, hi, cv2.CMP_LT), cv2.bitwise_not(has)))
    if cut is None:
        return labels
    H, W = labels.shape
    cx, cy = cut[:, 0, 0], cut[:, 0, 1]
    fill = np.zeros(len(cy), labels.dtype)
    for dy, dx in window_offsets(px):
        y, x = cy + dy, cx + dx
        inside = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        v = np.where(inside, labels[np.clip(y, 0, H - 1), np.clip(x, 0, W - 1)], 0)
        fill = np.where(fill == 0, v, fill)
    labels = labels.copy()
    labels[cy, cx] = fill
    return labels


def fg_masks(fg, thr=0.7, cut_px=2, min_size=15):
    """As masks.fg_masks: fg (H, W) -> labels (H, W) int32, masks numbered 1..K."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((fg > thr).astype(np.uint8), connectivity=4,
                                                           ltype=cv2.CV_32S)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_size
    keep[0] = False
    if not keep[1:].all():
        labels = np.take((np.cumsum(keep) * keep).astype(np.int32), labels)
    return fill_cuts(labels, cut_px)


# Hybrid flow masks: cellpose-style flow following, with the endpoints given to the detections' Gaussian mixtures,
# so each mask belongs to a detection (and through the tracker to a track) instead of to an anonymous cluster.

def smooth_unit_flow(flow, sigma=1.5, scale=2):
    """flow (2, H, W) -> scale x downsampled, Gaussian-smoothed (sigma in full-res px) unit vectors (h, w, 2) [dy, dx]."""
    H, W = flow.shape[1:]
    c = [np.ascontiguousarray(f) for f in flow]
    if scale > 1:
        c = [cv2.resize(f, (W // scale, H // scale), interpolation=cv2.INTER_AREA) for f in c]
    if sigma > 0:
        c = [cv2.GaussianBlur(f, (0, 0), sigma / scale) for f in c]
    m = np.maximum(cv2.magnitude(c[0], c[1]), 1e-6)
    return cv2.merge([c[0] / m, c[1] / m])


def follow_flow(f, y, x, steps=20, momentum=0.9):
    """Move positions y, x (n,) along the unit flow f (h, w, 2) with Nesterov momentum, as masks.follow_flow;
    bilinear sampling by cv2.remap."""
    h, w = f.shape[:2]
    n = len(y)
    cols = 1024                                   # cv2.remap wants maps under 32767 px per side
    rows = -(-n // cols)
    py, px = np.zeros(rows * cols, np.float32), np.zeros(rows * cols, np.float32)
    py[:n], px[:n] = y, x
    vy, vx = np.zeros_like(py), np.zeros_like(px)
    for _ in range(steps):
        g = cv2.remap(f, px.reshape(rows, cols), py.reshape(rows, cols), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)
        vy, vx = momentum * vy + g[:, 0], momentum * vx + g[:, 1]
        py = np.clip(py + g[:, 0] + momentum * vy, 0, h - 1)
        px = np.clip(px + g[:, 1] + momentum * vx, 0, w - 1)
    return py[:n], px[:n]


def gmm_assign(q, points, shape, prob, k_near=4, min_level=0.1):
    """Detection (0..M-1, or -1) of each query point q (n, 2): the one whose mixture of Gaussians - points (M, P, 2),
    shape (M, P, 2, 2) - weighted by its prob (M,), explains it best; -1 where no mixture reaches min_level
    (sum_i exp(-d_i^2 / 2), the outline level of gmm.outline)."""
    from scipy.spatial import cKDTree
    M = len(points)
    if not M or not len(q):
        return np.full(len(q), -1, np.int64)
    k = min(k_near, M)
    nb = cKDTree(points[:, 0]).query(q, k=k)[1].reshape(len(q), k)          # (n, k) nearest centres
    inv = np.linalg.inv(shape.astype(np.float64)).astype(np.float32)
    d = q[:, None, None, :] - points[nb]                                    # (n, k, P, 2)
    m = np.einsum("nkpi,nkpij,nkpj->nkp", d, inv[nb], d)
    level = np.exp(-0.5 * m).sum(-1)                                        # (n, k)
    best = np.argmax(np.log(np.maximum(level, 1e-30)) + np.log(np.maximum(prob[nb], 1e-6)), 1)
    r = np.arange(len(q))
    return np.where(level[r, best] >= min_level, nb[r, best], -1)


def flow_gmm_masks(fg, flow, points, shape, prob, thr=0.7, sigma=1.5, scale=2, steps=20, min_prob=0.1,
                   min_level=0.1, cut_px=2, min_size=15):
    """fg (H, W), flow (2, H, W), the detections' points (M, P, 2), shape (M, P, 2, 2) and P(real) prob (M,) ->
    labels (H, W) int32: j + 1 for the pixels of detection j, M + 1.. for fg regions no detection claims.
    fg pixels (on a scale x downsampled grid) follow the smoothed flow to their cell's medial axis, where the
    detection's Gaussians sit; each endpoint goes to the best mixture among detections with prob >= min_prob.
    Masks under min_size px (counted on the downsampled grid) are dropped."""
    H, W = fg.shape
    binary = (fg > thr).astype(np.uint8)
    small = cv2.resize(binary, (W // scale, H // scale), interpolation=cv2.INTER_AREA) if scale > 1 else binary
    ys, xs = np.nonzero(small)
    M = len(points)
    lab = np.zeros(small.shape, np.int32)
    if not len(ys):
        return np.zeros((H, W), np.int32)
    ey, ex = follow_flow(smooth_unit_flow(flow, sigma, scale), ys.astype(np.float32), xs.astype(np.float32), steps)
    end = np.round(np.stack([ey, ex], 1) * scale + (scale - 1) / 2).astype(np.int64)   # full-res endpoint pixels
    key, inv = np.unique(end[:, 0] * W + end[:, 1], return_inverse=True)   # endpoints pile up: assign each once
    seeds = np.nonzero(prob >= min_prob)[0]
    who = gmm_assign(np.stack([key // W, key % W], 1).astype(np.float32), points[seeds].astype(np.float32),
                     shape[seeds], prob[seeds], min_level=min_level)
    lab[ys, xs] = np.where(who >= 0, seeds[np.maximum(who, 0)] + 1, 0)[inv.ravel()]
    rest = (small > 0) & (lab == 0)                                         # fg that no detection claimed
    if rest.any():
        cc = cv2.connectedComponents(rest.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)[1]
        lab[rest] = cc[rest] + M
    counts = np.bincount(lab.ravel())
    drop = np.nonzero(counts[1:] * scale * scale < min_size)[0] + 1
    if len(drop):
        lut = np.arange(len(counts), dtype=np.int32)
        lut[drop] = 0
        lab = np.take(lut, lab)
    if scale > 1:
        lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    return fill_cuts(lab * binary, cut_px)
