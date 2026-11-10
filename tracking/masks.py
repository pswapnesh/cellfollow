"""Instance masks from the foreground output, which has touching cells cut apart.

    fg_masks    connected components of fg > thr, cuts filled (fastest)
    flow_masks  fg > thr, pixels follow the flow and their endpoints are clustered, cuts filled
    with_track_ids  gives masks the id of the track whose centre lies inside
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

UNTRACKED = 2_000_000  # ids of masks without a track: UNTRACKED + mask number


def fill_cuts(labels, px):
    """Unlabelled pixels with two different masks within px (the cut between touching cells) take the nearest mask.
    The outer outline of the masks does not grow."""
    has = labels > 0
    if px <= 0 or not has.any():
        return labels
    kernel = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    lab = labels.astype(np.float32)
    hi = cv2.dilate(lab, kernel)
    lo = cv2.erode(np.where(has, lab, np.inf).astype(np.float32), kernel)
    cut = ~has & (hi > 0) & (lo < hi)
    if not cut.any():
        return labels
    _, near = cv2.distanceTransformWithLabels((~has).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    at = np.zeros(int(near.max()) + 1, np.int64)
    at[near[has]] = labels[has]
    labels = labels.copy()
    labels[cut] = at[near[cut]]
    return labels


def drop_small(labels, min_size):
    """Drop masks smaller than min_size px, renumber 1..K."""
    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False
    return (np.cumsum(keep) * keep)[labels]


def fg_masks(fg, thr=0.7, cut_px=2, min_size=15):
    """fg (H, W) tensor -> labels (H, W) int64."""
    binary = (fg.float() > thr).cpu().numpy().astype(np.uint8)
    labels = cv2.connectedComponents(binary, connectivity=4, ltype=cv2.CV_32S)[1].astype(np.int64)
    return fill_cuts(drop_small(labels, min_size), cut_px)


def follow_flow(flow, pos, steps=20, momentum=0.9):
    """Move positions (N, 2) along the flow (2, H, W) with Nesterov momentum."""
    H, W = flow.shape[1:]
    lim = torch.tensor([H - 1, W - 1], device=pos.device, dtype=pos.dtype)
    p, v = pos.clone(), torch.zeros_like(pos)
    for _ in range(steps):
        grid = torch.stack([p[:, 1] / (W - 1) * 2 - 1, p[:, 0] / (H - 1) * 2 - 1], -1)
        g = F.grid_sample(flow[None], grid[None, None], align_corners=True)[0, :, 0].T
        v = momentum * v + g
        p = torch.minimum((p + g + momentum * v).clamp(min=0), lim)
    return p


def cluster_endpoints(end, shape, min_count=2, max_dist=1.5):
    """Endpoints (N, 2) -> label per endpoint (0 = noise): dense endpoint regions are seeds, endpoints join the seed
    within max_dist px."""
    H, W = shape
    iy, ix = end[:, 0].round().long().clamp(0, H - 1), end[:, 1].round().long().clamp(0, W - 1)
    hist = torch.bincount(iy * W + ix, minlength=H * W).view(1, 1, H, W).float()
    seeds = (F.avg_pool2d(hist, 3, 1, 1)[0, 0] * 9 >= min_count).cpu().numpy()
    if not seeds.any():
        return torch.zeros(len(end), dtype=torch.long, device=end.device)
    dist, near = cv2.distanceTransformWithLabels((~seeds).astype(np.uint8), cv2.DIST_L2, 3,
                                                 labelType=cv2.DIST_LABEL_CCOMP)
    near[dist > max_dist] = 0
    return torch.from_numpy(near.astype(np.int64)).to(end.device)[iy, ix]


def flow_masks(fg, flow, thr=0.7, steps=20, scale=2, cut_px=2, min_size=15):
    """fg (H, W), flow (2, H, W) tensors -> labels (H, W) int64; following runs on a scale x downsampled grid."""
    H, W = fg.shape
    h, w = H // scale, W // scale
    binary = fg.float() > thr
    small = F.adaptive_avg_pool2d(binary[None, None].float(), (h, w))[0, 0] > 0.5
    f = F.adaptive_avg_pool2d(flow[None].float(), (h, w))[0]
    if not small.any():
        return np.zeros((H, W), np.int64)
    ys, xs = small.nonzero(as_tuple=True)
    end = follow_flow(f / f.norm(dim=0, keepdim=True).clamp(min=1e-6), torch.stack([ys, xs], 1).float(), steps)
    lab = torch.zeros(h, w, dtype=torch.long, device=fg.device)
    lab[ys, xs] = cluster_endpoints(end, (h, w))
    lab = F.interpolate(lab[None, None].float(), size=(H, W), mode="nearest")[0, 0].long() * binary
    return fill_cuts(drop_small(lab.cpu().numpy(), min_size), cut_px)


def with_track_ids(labels, ids, yx, p):
    """Each mask takes the id of the most probable track whose centre lies inside; others get UNTRACKED + number."""
    out = np.where(labels > 0, labels + UNTRACKED, 0)
    if not len(ids) or not labels.any():
        return out
    H, W = labels.shape
    lab = labels[np.clip(np.round(yx[:, 0]).astype(int), 0, H - 1), np.clip(np.round(yx[:, 1]).astype(int), 0, W - 1)]
    lut = np.where(np.arange(int(labels.max()) + 1) > 0, np.arange(int(labels.max()) + 1) + UNTRACKED, 0)
    for j in np.argsort(p):                                                  # most probable written last
        if lab[j] > 0:
            lut[lab[j]] = ids[j]
    return lut[labels]
