"""Check an exported model (export_fast.py) against CellTracker(masks="fg") on the same frames, and time both.

    python check_fast.py movie.mp4                                   # every export found for 640, and the torch net
    python check_fast.py stack.tif --model weights/cellpoint_640_fp16.engine --frames 100
    python check_fast.py image.tif --pan                             # one image, panned to make a moving video

Reports, per model, how many detections agree (same centre within 1 px), the largest P(real) difference among them,
the fraction of pixels whose mask (cell or background) agrees, and how often the same track ids are reported.
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np

from cell_tracker import CellTracker
from fast_tracker import WEIGHTS, FastCellTracker
from test_live import center_crop
from test_on_video import read_tif, read_video, to_bgr8

HERE = os.path.dirname(os.path.abspath(__file__))


def load_frames(path, size, n, pan):
    frames, _, _ = read_tif(path) if path.lower().endswith((".tif", ".tiff")) else read_video(path)
    frames = [to_bgr8(f) for _, f in zip(range(1 if pan else n), frames)]
    if pan:
        img, s = frames[0], 2 * size
        h, w = img.shape[:2]
        if min(h, w) < s + 3 * n:
            raise SystemExit(f"--pan needs an image of at least {s + 3 * n} px")
        y0, x0 = (h - s - 3 * n) // 2, (w - s - 2 * n) // 2
        frames = [img[y0 + 3 * i:y0 + 3 * i + s, x0 + 2 * i:x0 + 2 * i + s] for i in range(n)]
    return [cv2.cvtColor(center_crop(f, size), cv2.COLOR_BGR2GRAY) for f in frames]


def run(tracker, frames):
    out, ms = [], []
    for f in frames:
        t0 = time.perf_counter()
        det = tracker.detect(f)
        res = tracker.track(det)
        ms.append((time.perf_counter() - t0) * 1e3)
        out.append((det, res))
    return out, float(np.median(ms[3:] if len(ms) > 6 else ms))


def compare(ref, new):
    matched = total = 0
    dprob, mask_eq, same_ids = [], [], []
    for (d0, r0), (d1, r1) in zip(ref, new):
        a, b = d0["mu"][:, :2], d1["mu"][:, :2]
        total += max(len(a), len(b))
        if len(a) and len(b):
            dist = np.linalg.norm(a[:, None] - b[None], axis=-1)
            i, j = np.nonzero((dist <= 1.0) & (dist == dist.min(1, keepdims=True)))
            matched += len(i)
            if len(i):
                dprob.append(np.abs(d0["prob"][i] - d1["prob"][j]).max())
        if d0["labels"] is not None:
            mask_eq.append(((d0["labels"] > 0) == (d1["labels"] > 0)).mean())   # cell or background
        same_ids.append({t.id for t in r0.tracks} == {t.id for t in r1.tracks})
    return dict(dets=f"{matched}/{total}", max_dprob=f"{max(dprob, default=0):.3f}",
                mask_agree=f"{np.mean(mask_eq):.4f}" if mask_eq else "n/a", same_track_ids=f"{np.mean(same_ids):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="video or TIFF stack (or one image with --pan)")
    p.add_argument("--model", action="append", help="export(s) to check; default: all found for --size, and torch")
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--pan", action="store_true", help="make a video by panning across the first frame")
    p.add_argument("--masks", default="fg", choices=("fg", "flow", "none"),
                   help="masks of the fast tracker. fg / flow: the full exports against CellTracker with fg masks (only "
                        "cell vs background is compared); none: the points-only exports against CellTracker without")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--ref-device", default="cpu", help="device of the reference CellTracker (cpu: fp32, cuda: fp16)")
    args = p.parse_args()

    frames = load_frames(args.input, args.size, args.frames, args.pan)
    masks = None if args.masks == "none" else args.masks
    found = glob.glob(os.path.join(HERE, "weights", f"cellpoint_{args.size}*"))
    found = [m for m in found if ("_points" in m) == (masks is None) and not m.endswith("_fp16.onnx")]
    models = args.model or [args.weights] + sorted(found)
    ref_tracker = CellTracker(args.weights, masks=masks and "fg", device=args.ref_device, invert=args.invert)
    ref, ref_ms = run(ref_tracker, frames)
    print(f"reference CellTracker {masks and 'fg' or 'points only'} on {args.ref_device}: {ref_ms:.1f} ms/frame, "
          f"{np.mean([len(d['prob']) for d, _ in ref]):.0f} detections/frame")
    for m in models:
        tr = FastCellTracker(None if masks is None else m, points_model=m if masks is None else None,
                             weights=args.weights, size=args.size, invert=args.invert, masks=masks, show_untracked=True)
        new, ms = run(tr, frames)
        stats = compare(ref, new)
        if args.masks == "flow":          # flow masks start tracks per detection mask: ids are numbered differently
            stats["same_track_ids"] = "n/a"
        print(f"{tr.backend:32s} {os.path.basename(m):28s} {ms:6.1f} ms/frame  {stats}")


if __name__ == "__main__":
    main()
