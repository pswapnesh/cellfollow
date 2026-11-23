"""Live webcam view of FastCellTracker on the networks exported by export_fast.py: the full network with masks and
tracking, or the points-only network (dense decoder cut off) with tracking and outlines. The camera is read on its
own thread, so the loop always works on the newest frame and never waits on the camera's frame interval.

Keys: q/Esc quit | m masks fg / flow / none (points only) | c masks / outlines | u untracked masks on / off | i invert
      r reset tracks | s save frame
      [ / ] outline level down / up (lower level: larger outline)

    python export_fast.py                       # once: builds the GPU engine and the CPU model
    python fast_live.py                         # TensorRT on the GPU if there is one, else OpenVINO on the CPU
    python fast_live.py --cpu --threads 4
    python fast_live.py --masks none            # points only: detections and tracks, outlines, no dense head
    python fast_live.py --model weights/cellpoint_640.onnx
    python fast_live.py --cpu --pipeline        # network of the next frame runs while this one is post-processed

--pipeline overlaps the network with post-processing, tracking and drawing (worth it on the CPU, where the network
is the slow part), at the cost of showing results one frame late.
"""

import argparse
import os
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from fast_tracker import MASKS, WEIGHTS, FastCellTracker
from test_live import center_crop, draw, open_camera

HERE = os.path.dirname(os.path.abspath(__file__))


class LatestFrame:
    """Reads the camera continuously on a thread and keeps only the newest frame. alive turns False only when the
    camera itself gives up (camera.Camera retries dropped frames and stalls first)."""

    def __init__(self, cap):
        self.cap, self.frame, self.n, self.alive = cap, None, 0, True
        self.cond = threading.Condition()
        self.fps, self._t = 0.0, None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.alive:
            try:
                ok, frame = self.cap.read()
            except Exception as e:
                print(f"camera thread: {e}")
                ok, frame = False, None
            t = time.perf_counter()
            with self.cond:
                if ok:
                    if self._t is not None:
                        self.fps = 0.9 * self.fps + 0.1 / max(t - self._t, 1e-6)
                    self._t = t
                    self.frame, self.n = frame, self.n + 1
                else:
                    self.alive = False
                self.cond.notify_all()

    def read(self, last=0, timeout=0.25):
        """The newest frame after frame number last: (new, frame, n); new is False if none came within timeout
        (check alive: a stall is not the end)."""
        with self.cond:
            self.cond.wait_for(lambda: self.n > last or not self.alive, timeout)
            return self.n > last, self.frame, self.n

    def release(self):
        self.alive = False
        self.thread.join(timeout=1)
        self.cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="export_fast.py output; default: the GPU engine if there is a GPU, else OpenVINO")
    p.add_argument("--weights", default=WEIGHTS, help="checkpoint, for class names and tracking parameters")
    p.add_argument("--cam", type=int, default=-1, help="camera index; -1: USB camera if present, else integrated")
    p.add_argument("--size", type=int, default=640, help="frame size of the default model")
    p.add_argument("--masks", default="fg", choices=("fg", "flow", "none"),
                   help="none: points only, the network without its dense decoder (fastest)")
    p.add_argument("--points-model", help="export_fast.py points-only output; default: as --model")
    p.add_argument("--contours", action="store_true", help="draw outlines from the Gaussians instead of masks")
    p.add_argument("--show-untracked", action="store_true",
                   help="also show masks without a reported track (by default only tracked cells' masks)")
    p.add_argument("--invert", action="store_true")
    p.add_argument("--cpu", action="store_true", help="use the CPU model even if there is a GPU")
    p.add_argument("--threads", type=int, default=None, help="CPU inference threads (default: the runtime's choice)")
    p.add_argument("--level", type=float, default=0.5, help="outline at sum_i exp(-d_i^2 / 2) = level; lower: larger")
    p.add_argument("--pipeline", action="store_true",
                   help="run the network on the next frame during post-processing (faster, one frame late)")
    args = p.parse_args()
    level, contours = args.level, args.contours

    tracker = FastCellTracker(args.model, weights=args.weights, masks=None if args.masks == "none" else args.masks,
                              points_model=args.points_model, invert=args.invert,
                              threads=args.threads, device="cpu" if args.cpu else None, size=args.size,
                              show_untracked=args.show_untracked)
    size = tracker.size
    print(f"model: {tracker.model} ({tracker.backend}), {size} x {size}")
    cam = LatestFrame(open_camera(args.cam))
    trails, fps, n, t_prev = defaultdict(lambda: deque(maxlen=15)), 0.0, 0, time.perf_counter()
    pool = ThreadPoolExecutor(1) if args.pipeline else None    # one worker: network calls never overlap each other
    pending = None                                              # (frame, its network future), pipeline only
    try:
        while True:
            ok, frame, n = cam.read(n)
            if not ok:                            # no new frame yet: keep the window alive, quit only if the camera is gone
                if not cam.alive:
                    print("camera gone, stopping")
                    break
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue
            frame = center_crop(frame, size)
            t0 = time.perf_counter()
            if pool:
                job = (frame, pool.submit(tracker.infer, frame, True))
                if pending is None:
                    pending = job
                    continue
                (frame, net_out), pending = pending, job
                result = tracker.track(tracker.detect(frame, net_out.result()))
            else:
                result = tracker.update(frame)
            ms = (time.perf_counter() - t0) * 1e3
            vis = draw(frame, result, trails, tracker.names, contours, level)
            now = time.perf_counter()
            fps, t_prev = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6), now
            t = tracker.ms
            hud = (f"{tracker.device} {ms:4.1f} ms (net {t['net']:.1f} post {t['post']:.1f} track {t['track']:.1f}) | "
                   f"{fps:4.1f} fps (cam {cam.fps:4.1f}) | {len(result.tracks)} tracks | {tracker.masks or 'points'}")
            cv2.rectangle(vis, (0, 0), (size, 18), (0, 0, 0), -1)
            cv2.putText(vis, hud, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("cellpoint fast (q to quit)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                tracker.set_masks(MASKS[(MASKS.index(tracker.masks) + 1) % len(MASKS)])
                print(f"masks {tracker.masks or 'none (points only)'}: {tracker.model}")
                trails.clear()
            elif key == ord("c"):
                contours = not contours
            elif key == ord("u"):
                tracker.show_untracked = not tracker.show_untracked
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
    finally:
        if pool:
            pool.shutdown(wait=True)
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
