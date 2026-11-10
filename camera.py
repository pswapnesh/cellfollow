"""A camera that's a USB/V4L2 device if one is present, otherwise falls back to the Pi's CSI camera via
picamera2 (not installed on every machine, so only imported if actually needed). Exposes the same
read()/release()/isOpened() interface as cv2.VideoCapture, so it drops in wherever that was used.
"""

import os

import cv2


def usb_candidates():
    """USB/V4L2 capture devices, external cameras first, integrated last."""
    root = "/sys/class/video4linux"
    if not os.path.isdir(root):
        return []

    def read(d, f):
        path = os.path.join(root, d, f)
        return open(path).read().strip() if os.path.exists(path) else ""
    cams = [(int(d[5:]), read(d, "name")) for d in os.listdir(root)
            if d[5:].isdigit() and read(d, "index") in ("", "0")]  # index > 0: metadata node
    return sorted(sorted(cams), key=lambda c: "integrated" in c[1].lower())


class Camera:
    """cam=-1: the first USB camera found (external before integrated), or the Pi's CSI camera via picamera2 if
    none is found; cam>=0: that USB index specifically, no fallback."""

    def __init__(self, cam=-1):
        self.backend = self.cap = self.picam2 = None
        for i, name in ([(cam, f"camera {cam}")] if cam >= 0 else usb_candidates()):
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                print(f"camera {i}: {name}")
                self.backend, self.cap = "usb", cap
                return
            cap.release()
        if cam >= 0:
            raise SystemExit(f"camera {cam} not found")
        print("no USB camera found, falling back to the Pi camera (picamera2)")
        self._open_picamera2()

    def _open_picamera2(self):
        from picamera2 import Picamera2  # Pi-only; not a hard dependency elsewhere
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888"}))
        self.picam2.start()
        self.backend = "picamera2"

    def isOpened(self):
        return self.backend is not None

    def read(self):
        if self.backend == "usb":
            return self.cap.read()
        frame = self.picam2.capture_array()  # RGB888
        return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        if self.backend == "usb":
            self.cap.release()
        elif self.backend == "picamera2":
            self.picam2.stop()
