"""Live webcam view of CellTracker on a square centre crop: points only by default, outlines from each cell's Gaussians.

Keys: q/Esc quit | m masks (off / fg / flow) | i invert | r reset tracks | s save frame
      [ / ] outline level down / up (lower level: larger outline)

    python test_live.py
    python test_live.py --cam 0 --masks fg --cpu --level 0.3
"""

import argparse
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch

from camera import Camera
from cell_tracker import CellTracker
from tracking.gmm import outline

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = (None, "fg", "flow")


def open_camera(cam):
    """A USB camera if one is present, otherwise the Pi's CSI camera via picamera2 (see camera.py)."""
    return Camera(cam)


def center_crop(frame, size):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = frame[y0:y0 + s, x0:x0 + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_LINEAR)


def id_colors(ids):
    """BGR per id, stable over time."""
    ids = np.asarray(ids, np.int64)
    return np.stack([(ids * 37) % 196 + 60, (ids * 101) % 196 + 60, (ids * 211) % 196 + 60], -1).astype(np.uint8)


def draw(frame, result, trails, names, contours=False, level=0.5):
    """contours: outlines from each cell's Gaussians, also when masks exist; level: mixture level of the outline."""
    vis = frame.copy()
    if not contours and result.labels is not None and result.labels.any():
        cells = result.labels > 0
        vis[cells] = (0.55 * vis[cells] + 0.45 * id_colors(result.labels[cells])).astype(np.uint8)
    elif (contours or result.labels is None) and result.tracks:
        fill, lines = vis.copy(), []
        for t in result.tracks:
            c = outline(t.points, t.shape, level)
            if c is not None:
                col = tuple(int(v) for v in id_colors(t.id))
                cv2.fillPoly(fill, [c.astype(np.int32)], col)
                lines.append((c.astype(np.int32), col))
        vis = cv2.addWeighted(vis, 0.6, fill, 0.4, 0)
        for c, col in lines:
            cv2.polylines(vis, [c], True, col, 1, cv2.LINE_AA)
    for t in result.tracks:
        col = tuple(int(c) for c in id_colors(t.id))
        trails[t.id].append((int(t.x), int(t.y)))
        if len(trails[t.id]) > 1:
            cv2.polylines(vis, [np.array(trails[t.id], np.int32)], False, col, 1, cv2.LINE_AA)
        for py, px in t.points[1:]:
            cv2.circle(vis, (int(px), int(py)), 2, col, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(t.x), int(t.y)), 4, (255, 255, 0), -1, cv2.LINE_AA)  # centre: cyan
        cv2.circle(vis, (int(t.x), int(t.y)), 5, col, 1, cv2.LINE_AA)
        if len(result.tracks) < 80:
            cv2.putText(vis, f"{t.id} {names.get(t.cls, t.cls)} {t.p_exist:.2f}", (int(t.x) + 5, int(t.y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    return vis


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=os.path.join(HERE, "weights", "cellpoint.pt"))
    p.add_argument("--cam", type=int, default=-1, help="camera index; -1: USB camera if present, else integrated")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--masks", default="none", choices=("none", "fg", "flow"), help="none: points only, the decoder does not run")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--keep-dense", action="store_true",
                    help="keep the dense decoder loaded even with --masks none, so 'm' can switch it on later")
    p.add_argument("--level", type=float, default=0.5, help="outline at sum_i exp(-d_i^2 / 2) = level; lower: larger")
    args = p.parse_args()
    level = args.level

    device = "cpu" if args.cpu else None
    masks = None if args.masks == "none" else args.masks
    tracker = CellTracker(args.weights, masks=masks, device=device, invert=args.invert, keep_dense=args.keep_dense)
    print(f"device: {tracker.device} ({'GPU' if tracker.device.type == 'cuda' else 'CPU'})"
          + (f" - {torch.cuda.get_device_name(tracker.device)}, half={tracker.half}" if tracker.device.type == "cuda" else ""))
    print(f"dense decoder loaded: {tracker.net.decoder is not None}")
    cap = open_camera(args.cam)
    trails, fps = defaultdict(lambda: deque(maxlen=15)), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = center_crop(frame, args.size)
        t0 = time.perf_counter()
        result = tracker.update(frame)
        ms = (time.perf_counter() - t0) * 1000
        fps = 0.9 * fps + 0.1 * 1000 / max(ms, 1e-3)
        vis = draw(frame, result, trails, tracker.names, level=level)
        hud = f"{tracker.device.type} {ms:5.1f} ms | {fps:4.1f} fps | {len(result.tracks)} tracks | masks {tracker.masks} | lv {level:.2f}"
        cv2.rectangle(vis, (0, 0), (args.size, 18), (0, 0, 0), -1)
        cv2.putText(vis, hud, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("cellpoint (q to quit)", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("m"):
            try:
                tracker.set_masks(MODES[(MODES.index(tracker.masks) + 1) % len(MODES)])
                trails.clear()
            except ValueError as e:
                print(e, "(pass --keep-dense to allow switching masks on)")
        elif key == ord("i"):
            tracker.invert = not tracker.invert
        elif key == ord("r"):
            tracker.reset()
            trails.clear()
        elif key in (ord("["), ord("]")):
            level = float(np.clip(level + (0.05 if key == ord("]") else -0.05), 0.05, 2.0))
        elif key == ord("s"):
            path = os.path.join(HERE, f"capture_{time.strftime('%H%M%S')}.png")
            cv2.imwrite(path, vis)
            print("saved", path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
