"""Run CellTracker on a video or a TIFF stack and save a video with masks and tracks, drawn as in test_live.py.

    python test_on_video.py movie.mp4
    python test_on_video.py stack.tif --fps 10 --out result.mp4
    python test_on_video.py stack.tif --channel 1 --masks fg
    python test_on_video.py movie.mp4 --masks fg --contours
    python test_on_video.py movie.mp4 --scale 0.5 --min-score 0.25 --gray --invert
"""

import argparse
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import tifffile

from cell_tracker import CellTracker, normalize
from test_live import HERE, draw


def read_video(path):
    """Frames of a video as BGR; returns the frame generator, its length and fps."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)

    def frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()
    return frames(), n, fps


def read_tif(path, channel=0):
    """Frames of a TIFF stack (any time / z / position axes in order, one channel, RGB as BGR); no fps stored."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        stack, axes = series.asarray(), series.axes
    if "C" in axes:
        stack = np.take(stack, channel, axis=axes.index("C"))
        axes = axes.replace("C", "")
    if axes.endswith("S"):
        stack = stack[..., ::-1]
        stack = stack.reshape(-1, *stack.shape[-3:])
    else:
        stack = stack.reshape(-1, *stack.shape[-2:])
    return iter(stack), len(stack), None


def to_bgr8(frame):
    """Frame of any bit depth -> 8-bit BGR for drawing."""
    if frame.dtype != np.uint8:
        frame = (normalize(frame.astype(np.float32), 0.1, 99.9) * 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="video (mp4, avi, ...) or TIFF stack")
    p.add_argument("--out", default=None, help="output mp4 (default: <input>_cellpoint.mp4)")
    p.add_argument("--weights", default=os.path.join(HERE, "weights", "cellpoint.pt"))
    p.add_argument("--masks", default="none", choices=("none", "fg", "flow"), help="none: points only, the decoder does not run")
    p.add_argument("--contours", action="store_true", help="draw each cell's outline instead of its mask")
    p.add_argument("--level", type=float, default=0.5, help="outline level; lower: larger outline")
    p.add_argument("--fps", type=float, default=None, help="output frame rate (default: the video's; TIFF stacks: 10)")
    p.add_argument("--channel", type=int, default=0, help="TIFF stacks with several channels: the one to use")
    p.add_argument("--scale", type=float, default=1.0, help="resize frames before processing, e.g. 0.5")
    p.add_argument("--min-score", type=float, default=0.05, help="minimum detection probability of a point")
    p.add_argument("--gray", action="store_true", help="show grayscale (the network always sees grayscale)")
    p.add_argument("--invert", action="store_true", help="invert the image for the network and in the saved video")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    is_tif = args.input.lower().endswith((".tif", ".tiff"))
    frames, n, fps = read_tif(args.input, args.channel) if is_tif else read_video(args.input)
    fps = args.fps or fps or 10.0
    out_path = args.out or os.path.splitext(args.input)[0] + "_cellpoint.mp4"
    masks = None if args.masks == "none" else args.masks
    tracker = CellTracker(args.weights, masks=masks, device="cpu" if args.cpu else None, invert=args.invert,
                          min_score=args.min_score)

    trails, writer, done, spent, ids = defaultdict(lambda: deque(maxlen=15)), None, 0, 0.0, set()
    for frame in frames:
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        if args.gray and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        result = tracker.update(frame)
        ms = (time.perf_counter() - t0) * 1000
        spent += ms
        shown = to_bgr8(frame)
        vis = draw(255 - shown if args.invert else shown, result, trails, tracker.names, args.contours, args.level)
        H, W = vis.shape[:2]
        hud = f"frame {done} | {tracker.device.type} {ms:5.1f} ms | {len(result.tracks)} tracks | masks {tracker.masks}"
        cv2.rectangle(vis, (0, 0), (W, 18), (0, 0, 0), -1)
        cv2.putText(vis, hud, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            if not writer.isOpened():
                raise SystemExit(f"cannot write {out_path}")
        writer.write(vis)
        ids.update(t.id for t in result.tracks)
        done += 1
        if done % 50 == 0:
            print(f"{done}/{n} frames", flush=True)
    if writer is None:
        raise SystemExit(f"no frames in {args.input}")
    writer.release()
    print(f"{done} frames, {spent / done:.1f} ms per frame (without drawing), {len(ids)} tracks -> {out_path}")


if __name__ == "__main__":
    main()
