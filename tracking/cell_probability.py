"""P(real) of a detection: is it a cell, and not a duplicate or a false point?

Points only: the point score, calibrated. With the dense head: a logistic of that score and the fraction of its Gaussians
that lies on foreground not already claimed by a better detection (candidate_scores), which removes most duplicates.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .gmm import point_covariances


def logit_score(score):
    p = np.clip(score, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def calibrated_score(score, a, b):
    """P(real) = sigmoid(a logit(score) + b)."""
    return 1 / (1 + np.exp(-(a * logit_score(score) + b)))


def blur(img, sigma):
    """Gaussian blur of (C, H, W), zero outside the image."""
    r = int(3 * sigma + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=img.device, dtype=img.dtype) / sigma) ** 2)
    k = (k / k.sum()).expand(img.shape[0], -1)[:, None]
    x = F.conv2d(img[None], k[..., None], padding=(r, 0), groups=img.shape[0])
    return F.conv2d(x, k[:, :, None], padding=(0, r), groups=img.shape[0])[0]


def flow_covariances(pts, flow, fg, sigma=5.0, k_sigma=0.6, k_width=0.25, min_sigma=1.5, min_coherence=0.3, min_mass=0.2):
    """Covariances (N, P, 2, 2) of the points (N, P, 2), directed by the flow. f is the gradient of the cell's potential and
    points mostly across the cell, so the cell runs along the minor axis of the local E[f f^T] under fg. Length: the
    scatter of the neighbouring points. Width: k_width * nearest point distance. Points with low coherence or little fg
    keep the covariance from their neighbours."""
    S0 = torch.as_tensor(point_covariances(pts.cpu().numpy(), k_sigma, k_width), dtype=pts.dtype, device=pts.device)
    f = flow / flow.norm(dim=0, keepdim=True).clamp(min=1e-6)
    mass = blur(fg[None], sigma)
    T = blur(torch.stack([fg * f[0] ** 2, fg * f[0] * f[1], fg * f[1] ** 2]), sigma) / mass.clamp(min=1e-6)
    H, W = fg.shape
    y, x = pts[..., 0].round().long().clamp(0, H - 1), pts[..., 1].round().long().clamp(0, W - 1)
    a, b, c = T[0][y, x], T[1][y, x], T[2][y, x]
    theta = 0.5 * torch.atan2(2 * b, a - c)
    e = torch.stack([-theta.sin(), theta.cos()], -1)
    coherence = ((a - c) ** 2 + 4 * b ** 2).sqrt() / (a + c).clamp(min=1e-9)
    d = torch.cdist(pts, pts) + torch.eye(pts.shape[1], device=pts.device) * 1e9
    w2 = (k_width * d.amin(-1)).clamp(min=min_sigma) ** 2
    l2 = (S0[..., 0, 0] + S0[..., 1, 1] - 2 * w2).clamp(min=0)
    S = l2[..., None, None] * (e[..., None] * e[..., None, :]) + w2[..., None, None] * torch.eye(2, device=pts.device)
    used = (coherence >= min_coherence) & (mass[0][y, x] >= min_mass)
    return torch.where(used[..., None, None], S, S0)


def candidate_scores(p, S, score, fg, rho=1.0, n=16, tau=0.5, stride=4):
    """Mean fg (N,) under n fixed samples per Gaussian out to rho sigma, one batch for all detections. A stride px map cell
    belongs to the best-scored detection landing on it among those with mean fg >= tau, and only owned samples count,
    so a duplicate of a kept cell scores low."""
    H, W = fg.shape
    i = torch.arange(n, device=p.device, dtype=p.dtype)
    r, t = rho * ((i + 0.5) / n).sqrt(), i * 2.399963
    z = torch.stack([r * t.sin(), r * t.cos()], 1)
    x = p[:, :, None] + (torch.linalg.cholesky(S)[:, :, None] @ z[..., None])[..., 0]
    yi, xi = x[..., 0].round().long().clamp(0, H - 1), x[..., 1].round().long().clamp(0, W - 1)
    vals = fg[yi, xi]
    ok = vals.mean((1, 2)) >= tau
    cell = (yi // stride) * (W // stride + 1) + xi // stride
    rank = score.argsort().argsort() + 1
    top = torch.zeros(int(cell.max()) + 1, dtype=torch.long, device=p.device)
    if ok.any():
        top.scatter_reduce_(0, cell[ok].ravel(), rank[ok][:, None, None].expand_as(cell[ok]).ravel(), "amax")
    return (vals * (top[cell] == rank[:, None, None])).mean((1, 2))


@torch.no_grad()
def fg_probability(points, score, fg, flow, params):
    """P(real) (N,) from the score and the fg the detection covers. points (N, P, 2), score (N,), fg (H, W), flow (2, H, W)."""
    points, score, fg, flow = points.float(), score.float(), fg.float(), flow.float()
    covered = candidate_scores(points, flow_covariances(points, flow, fg), score, fg, params["rho"], stride=params["stride"])
    s = score.cpu().numpy().astype(np.float64)
    z = (np.c_[logit_score(s), covered.cpu().numpy().astype(np.float64)] - np.array(params["mu"])) / np.array(params["sd"])
    w = np.array(params["w"])
    return 1 / (1 + np.exp(-(z @ w[:2] + w[2])))
