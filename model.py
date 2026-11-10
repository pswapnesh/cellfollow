"""Inference network: YOLO26n backbone + neck, a dense decoder (flow + foreground), a point head and a covariance head on P3.

Input (B, 2, H, W) [main, second channel, zeros if absent]. Outputs:
    cls (B, nc, H/8, W/8) logits, off (B, 2, H/8, W/8) centre offset in its grid cell (0-1),
    kpt (B, 2k, H/8, W/8) the k eigen points relative to the centre, in grid units
    shape (B, 3P, H/8, W/8), err (B, E, H/8, W/8) Cholesky factors of the covariances of the P = k + 1 points, grid units
    dense (B, 5, H, W) [u logit, d logit, dy, dx, fg logit]; fg has touching cells cut apart
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, DWConv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import fuse_conv_and_bn

HERE = os.path.dirname(os.path.abspath(__file__))
# A vendored copy (third_party/ultralytics_cfg, see its LICENSE), not whatever's bundled with the installed
# ultralytics: older pins may lack YOLO26, and DetectionModel's own scale-suffix resolution finds this file even
# though "yolo26n.yaml" itself doesn't literally exist (only yolo26.yaml, with the n/s/m/l/x scales inside it).
YOLO26_YAML = os.path.join(HERE, "third_party", "ultralytics_cfg", "26", "yolo26n.yaml")

TAPS = (0, 2, 16, 19, 22)  # layers of yolo26n.yaml: P1, P2 (backbone), P3, P4, P5 (neck)
P3 = 2                     # index of P3 in TAPS
STRIDE, POINT_STRIDE = 32, 8


def dw_block(c):
    return nn.Sequential(DWConv(c, c, 3), Conv(c, c, 1))


class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out, n=2):
        super().__init__()
        self.proj = Conv(c_in, c_out, 1)
        self.skip = Conv(c_skip, c_out, 1)
        self.refine = nn.Sequential(*[dw_block(c_out) for _ in range(n)])

    def forward(self, x, skip):
        x = F.interpolate(self.proj(x), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(x + self.skip(skip))


class DenseDecoder(nn.Module):
    def __init__(self, in_chs, chs=(96, 64, 48, 32), out_ch=5, up=2):
        super().__init__()
        p1, p2, p3, p4, p5 = in_chs
        self.up4 = UpBlock(p5, p4, chs[0])
        self.up3 = UpBlock(chs[0], p3, chs[1])
        self.up2 = UpBlock(chs[1], p2, chs[2])
        self.up1 = UpBlock(chs[2], p1, chs[3])
        self.head = nn.Sequential(nn.Conv2d(chs[3], out_ch * up * up, 1), nn.PixelShuffle(up))

    def forward(self, feats):
        p1, p2, p3, p4, p5 = feats
        return self.head(self.up1(self.up2(self.up3(self.up4(p5, p4), p3), p2), p1))


class PointHead(nn.Module):
    def __init__(self, c_in, nc, k=4, hidden=64):
        super().__init__()
        branch = lambda c_out: nn.Sequential(Conv(c_in, hidden, 1), dw_block(hidden), dw_block(hidden),
                                             nn.Conv2d(hidden, c_out, 1))
        self.cls = branch(nc)
        self.reg = branch(2 + 2 * k)

    def forward(self, x):
        reg = self.reg(x)
        return {"cls": self.cls(x), "off": reg[:, :2].sigmoid(), "kpt": reg[:, 2:]}


class CovHead(nn.Module):
    """Decomposed, so the covariances stay positive definite. shape: how far the cell extends around each point.
    err: the joint error of all the points' positions."""

    def __init__(self, c_in, points, hidden=64):
        super().__init__()
        d = 2 * points
        branch = lambda c_out: nn.Sequential(Conv(c_in, hidden, 1), dw_block(hidden), dw_block(hidden),
                                             nn.Conv2d(hidden, c_out, 1))
        self.shape, self.err = branch(3 * points), branch(d * (d + 1) // 2)

    def forward(self, x):
        return {"shape": self.shape(x), "err": self.err(x)}


def shape_covariance(raw):
    """raw (N, 3P) -> (N, P, 2, 2) px^2."""
    raw = raw.float().view(len(raw), -1, 3)
    L = torch.zeros(*raw.shape[:-1], 2, 2, device=raw.device)
    L[..., 0, 0], L[..., 1, 0], L[..., 1, 1] = raw[..., 0].clamp(-4, 4).exp(), raw[..., 1], raw[..., 2].clamp(-4, 4).exp()
    L = L * POINT_STRIDE
    return L @ L.transpose(-1, -2)


def err_covariance(raw):
    """raw (N, E) -> (N, D, D) px^2, with D (D + 1) / 2 = E."""
    raw = raw.float()
    d = int((math.sqrt(8 * raw.shape[-1] + 1) - 1) / 2)
    rows, cols = torch.tril_indices(d, d, device=raw.device)
    L = torch.zeros(len(raw), d, d, device=raw.device)
    L[:, rows, cols] = raw
    idx = torch.arange(d, device=raw.device)
    L[:, idx, idx] = raw[:, rows == cols].clamp(-4, 4).exp()
    L = L * POINT_STRIDE
    return L @ L.transpose(-1, -2)


class InputAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 1)

    def forward(self, x):
        return self.conv(x)


class CellNet(nn.Module):
    def __init__(self, nc=4, k=4):
        super().__init__()
        self.adapter = InputAdapter()
        self.backbone = DetectionModel(YOLO26_YAML, ch=3, verbose=False).model[:-1]  # without the Detect head
        with torch.no_grad():
            chs = [f.shape[1] for f in self.features(torch.zeros(1, 3, 64, 64))]
        self.decoder = DenseDecoder(chs)
        self.points = PointHead(chs[P3], nc, k)
        self.cov = CovHead(chs[P3], k + 1)

    def features(self, x, taps=TAPS):
        y = []
        for m in self.backbone[:max(taps) + 1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)
        return [y[j] for j in taps]

    def forward(self, img, dense=True):
        img = self.adapter(img)
        if not dense:
            p3 = self.features(img, (TAPS[P3],))[0]
            return {**self.points(p3), **self.cov(p3)}
        feats = self.features(img)
        return {**self.points(feats[P3]), **self.cov(feats[P3]), "dense": self.decoder(feats)}

    @torch.no_grad()
    def predict(self, img, thr=0.05, dense=True):
        """Any image size: pad to the stride, run, crop back. Returns the detections (decode_points) and, with dense,
        fg (B, 1, H, W) probability and flow (B, 2, H, W) [dy, dx]."""
        H, W = img.shape[-2:]
        out = self(F.pad(img, (0, -W % STRIDE, 0, -H % STRIDE), mode="reflect"), dense=dense)
        res = decode_points(out, thr)
        if dense:
            d = out["dense"][..., :H, :W]
            res.update(fg=d[:, 4:5].sigmoid(), flow=d[:, 2:4])
        return res

    def fuse(self):
        """Fold BatchNorm into the convolutions (outputs unchanged, faster)."""
        for mod in self.modules():
            if isinstance(mod, (Conv, DWConv)) and hasattr(mod, "bn"):
                mod.conv = fuse_conv_and_bn(mod.conv, mod.bn)
                delattr(mod, "bn")
                mod.forward = mod.forward_fuse
        return self

    def points_only(self):
        """Drop everything that only feeds the dense output."""
        self.decoder = None
        self.backbone = self.backbone[:TAPS[P3] + 1]
        return self


def decode_points(out, thr=0.05):
    """Local maxima of the class-agnostic centre probability of the first image -> yx (N, 2) px, score (N,),
    cls_prob (N, nc), kpts (N, k, 2) px, shape (N, k + 1, 2, 2) and R (N, 2k + 2, 2k + 2) covariances px^2."""
    heat = out["cls"][0].float().sigmoid()
    best = heat.max(0).values
    gy, gx = ((best == F.max_pool2d(best[None], 3, 1, 1)[0]) & (best > thr)).nonzero(as_tuple=True)
    off = out["off"][0, :, gy, gx].float()
    yx = torch.stack([gy + off[0], gx + off[1]], 1) * POINT_STRIDE
    kpt = out["kpt"][0, :, gy, gx].float().T.view(len(gy), out["kpt"].shape[1] // 2, 2) * POINT_STRIDE + yx[:, None]
    return {"yx": yx, "score": best[gy, gx], "cls_prob": heat[:, gy, gx].T, "kpts": kpt,
            "shape": shape_covariance(out["shape"][0, :, gy, gx].T), "R": err_covariance(out["err"][0, :, gy, gx].T)}
