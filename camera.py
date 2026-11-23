"""A camera that's a USB/V4L2 device if one is present, otherwise falls back to the Pi's CSI camera via
picamera2 (not installed on every machine, so only imported if actually needed). Exposes the same
read()/release()/isOpened() interface as cv2.VideoCapture, so it drops in wherever that was used.

read() rides out dropped frames and short stalls: a failed read is retried (a USB device reopened after repeated
failures) for up to retry_s seconds before it reports the camera gone, so one bad frame never ends a live view.
"""

import os
import time

import cv2


def usb_candidates():
    """USB/V4L2 capture devices (index, name, kind): plugged-in USB cameras first, then unknown, integrated last.
    Built-in laptop webcams are USB devices too; the kernel marks their port removable=fixed, which is the main
    signal, with "integrated" in the name as a fallback."""
    root = "/sys/class/video4linux"
    if not os.path.isdir(root):
        return []

    def read(*parts):
        path = os.path.join(root, *parts)
        return open(path).read().strip() if os.path.exists(path) else ""

    def kind(d):
        removable = read(d, "device", "..", "removable")       # the USB port of the camera's interface
        if removable == "removable":
            return "usb"
        if removable == "fixed" or "integrated" in read(d, "name").lower():
            return "integrated"
        return "unknown"
    rank = {"usb": 0, "unknown": 1, "integrated": 2}
    cams = [(int(d[5:]), read(d, "name"), kind(d)) for d in os.listdir(root)
            if d[5:].isdigit() and read(d, "index") in ("", "0")]  # index > 0: metadata node
    return sorted(cams, key=lambda c: (rank[c[2]], c[0]))


MJPG = cv2.VideoWriter_fourcc(*"MJPG")


def open_usb(index):
    """cv2.VideoCapture of a USB camera asking for MJPG, or None if it gives no frame. Uncompressed YUYV at 640 x 480
    and above can need more than a USB 2 port gives: the camera then streams (its light is on) but no frame ever
    arrives. Cameras without MJPG keep their default format."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2) if os.path.isdir("/sys/class/video4linux") else cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, MJPG)
    if cap.read()[0]:
        return cap
    cap.release()
    return None


class Camera:
    """cam=-1: the first USB camera found (external before integrated), or the Pi's CSI camera via picamera2 if
    none is found; cam>=0: that USB index specifically, no fallback. retry_s: how long read() keeps retrying a
    failing camera before returning (False, None)."""
    retry_s, index, kind = 5.0, None, None

    def __init__(self, cam=-1, retry_s=5.0):
        self.backend = self.cap = self.picam2 = None
        self.retry_s = retry_s
        for i, name, kind in ([(cam, f"camera {cam}", "chosen")] if cam >= 0 else usb_candidates()):
            cap = open_usb(i)
            if cap is not None:
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace")
                print(f"camera {i}: {name} ({kind}), {fourcc} {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} x "
                      f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
                self.backend, self.cap, self.index, self.kind = "usb", cap, i, kind
                return
            print(f"camera {i}: {name} ({kind}) did not give a frame, trying the next")
        if cam >= 0:
            raise SystemExit(f"camera {cam} not found")
        print("no USB camera found, falling back to the Pi camera (picamera2)")
        self._open_picamera2()

    def _open_picamera2(self):
        from picamera2 import Picamera2  # Pi-only; not a hard dependency elsewhere
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888"}))
        self.picam2.start()
        self.backend, self.kind = "picamera2", "picamera2"

    def isOpened(self):
        return self.backend is not None

    def _read_once(self):
        try:
            if self.backend == "usb":
                ok, frame = self.cap.read()
                return ok and frame is not None, frame
            frame = self.picam2.capture_array()  # RGB888
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:                   # a driver hiccup is a failed frame, not the end of the program
            print(f"camera read failed: {e}")
            return False, None

    def _reopen(self):
        if self.backend == "usb" and self.index is not None:
            print(f"camera {self.index}: reopening")
            self.cap.release()
            self.cap = open_usb(self.index) or cv2.VideoCapture(self.index)

    def read(self):
        """(ok, frame): retries a failing camera for up to retry_s s; (False, None) only if it stays gone."""
        t0, fails = time.monotonic(), 0
        while True:
            ok, frame = self._read_once()
            if ok:
                if fails:
                    print(f"camera back after {fails} failed reads")
                return True, frame
            fails += 1
            if time.monotonic() - t0 >= self.retry_s:
                print(f"camera gone: no frame for {self.retry_s:.0f} s")
                return False, None
            if fails % 20 == 0:
                self._reopen()
            time.sleep(0.05)

    def release(self):
        if self.backend == "usb":
            self.cap.release()
        elif self.backend == "picamera2":
            self.picam2.stop()
