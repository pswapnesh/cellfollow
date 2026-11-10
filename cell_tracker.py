"""CellTracker: frame in, tracked cells out.

    tracker = CellTracker("weights/cellpoint.pt")               # masks: None (points only, fastest), "fg" or "flow"
    result = tracker.update(frame)                              # frame: (H, W) grayscale or (H, W, 3) BGR
    result.tracks                                               # [Track(id, y, x, vy, vx, p_exist, cls, cls_prob, points, shape)]
    result.labels                                               # (H, W) int64, a mask holds its track's id; None if
                                                                # masks=None (tracking.gmm.outline gives a track's outline)
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from model import CellNet
from tracking.bayes_tracker import BayesTracker
from tracking.cell_probability import calibrated_score, fg_probability
from tracking.masks import flow_masks, fg_masks, with_track_ids


def normalize(channel, lo_pct=1.0, hi_pct=99.0):
    """Percentile range -> [0, 1], as in training."""
    lo, hi = np.percentile(channel, (lo_pct, hi_pct))
    return np.clip((channel - lo) / (hi - lo), 0, 1).astype(np.float32) if hi > lo else np.zeros_like(channel, np.float32)


@dataclass
class Result:
    tracks: list
    labels: np.ndarray = None


class CellTracker:
    def __init__(self, weights, masks=None, device=None, min_score=0.05, fg_thr=0.7, invert=False, keep_dense=False,
                 **tracker_kw):
        """keep_dense: keep the decoder with masks=None, so set_masks can turn it on later."""
        self.check_masks(masks)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.masks, self.min_score, self.fg_thr, self.invert = masks, min_score, fg_thr, invert
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.names, self.k, self.params = ck["names"], ck["k"], ck["tracking"]
        net = CellNet(len(self.names), ck["k"])
        net.load_state_dict(ck["model"])
        net = net.eval().fuse()
        self.half = self.device.type == "cuda"
        self.net = (net if masks or keep_dense else net.points_only()).to(self.device).to(memory_format=torch.channels_last)
        self.net = self.net.half() if self.half else self.net
        make = lambda key: BayesTracker(self.params["Q"], nc=len(self.names), k=ck["k"], **{**self.params[key], **tracker_kw})
        self.trackers = {"points": make("points_tracker"), "dense": make("fg_tracker")}

    @staticmethod
    def check_masks(masks):
        if masks not in ("fg", "flow", None):
            raise ValueError(f"unknown masks {masks!r}: use 'fg', 'flow' or None")

    @property
    def tracker(self):
        return self.trackers["dense" if self.masks else "points"]

    def set_masks(self, masks):
        """Change the mask mode; tracks start again."""
        self.check_masks(masks)
        if masks and self.net.decoder is None:
            raise ValueError("no decoder: create the CellTracker with keep_dense=True")
        self.masks = masks
        self.reset()

    def reset(self):
        """Forget all tracks (new video)."""
        for t in self.trackers.values():
            t.reset()

    def to_input(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        main = normalize(gray.astype(np.float32))
        main = 1 - main if self.invert else main
        x = torch.from_numpy(np.stack([main, np.zeros_like(main)]))[None].to(self.device)
        return (x.half() if self.half else x).to(memory_format=torch.channels_last)

    def detect(self, frame):
        """Network and P(real) of one frame: what BayesTracker.update takes."""
        res = self.net.predict(self.to_input(frame), thr=self.min_score, dense=self.masks is not None)
        yx, kpts, score = res["yx"].float(), res["kpts"].float(), res["score"].float()
        points = torch.cat([yx[:, None], kpts], 1)
        labels = None
        if self.masks:
            fg, flow = res["fg"][0, 0], res["flow"][0]
            labels = flow_masks(fg, flow, self.fg_thr) if self.masks == "flow" else fg_masks(fg, self.fg_thr)
        if not len(yx):
            prob = np.zeros(0)
        elif self.masks:
            prob = fg_probability(points, score, fg, flow, self.params["logistic"])
        else:
            prob = calibrated_score(score.cpu().numpy().astype(np.float64), *self.params["score_calibration"])
        as_np = lambda t: t.cpu().numpy().astype(np.float64)
        return dict(mu=as_np(points.flatten(1)), R=as_np(res["R"]) + 1e-3 * np.eye(2 * (self.k + 1)), shape=as_np(res["shape"]),
                    prob=prob, cls_prob=as_np(res["cls_prob"].float()), labels=labels)

    def track(self, det):
        """Tracks and masks from the output of detect."""
        tracks = self.tracker.update(**det)
        return Result(tracks, None if det["labels"] is None else with_track_ids(det["labels"], *self.tracker.current()))

    def update(self, frame):
        return self.track(self.detect(frame))
