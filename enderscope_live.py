"""Live CellTracker view driven by an Enderscope (https://github.com/mutterer/enderscopy): jog and home the stage
while tracking, on the scope's own camera. Frames come from OpenCV, same as test_live.py. The enderscope package
itself lives unmodified under third_party/enderscopy (a plain downloaded copy, pinned commit, not pip-installed:
its pyproject.toml fails to build on current setuptools, see README) and is imported from there.

Keys: q/Esc quit | m masks (off / fg / flow) | i invert | r reset tracks | s save frame
      [ / ] outline level down / up (lower level: larger outline)
      a/d jog west/east (x) | w/x jog south/north (y) | e/c jog up/down (z) - only the two axes in --axes jog
      h home | l toggle light
      Left/Right (or Up/Down) select a tracked cell | f toggle following it (switches to the nearest-to-centre
      track if it's lost, rather than stopping; stops only on f again or if no tracks remain)
      +/- increase/decrease follow gain (speed of the follow correction)

    python enderscope_live.py --stage-port /dev/ttyUSB0
    python enderscope_live.py --stage-port /dev/ttyUSB0 --axes xy
    python enderscope_live.py --no-stage   # camera + tracker only, no hardware (e.g. to test the vision side first)

Only two axes are ever driven, whichever --axes names (default xz): jog and follow both only touch those two,
leaving the third alone entirely - set this to whichever pair your camera actually moves with.

Follow (f): follows the cell selected with the arrow keys (white ring; the lowest id is selected by default).
Every tracking frame, a dedicated single-point Kalman filter (position + velocity, separate from the main
BayesTracker's own 5-point one - that one is tuned for matching/shape, this one only for smoothing the centre
point for stage control) is fed the cell's centre; the filtered position and velocity, led by --follow-lookahead
frames, become the error for a PID controller (one per axis, --follow-gain/--follow-ki/--follow-kd, output
capped at --follow-max-step mm) whose output is what actually moves the stage. The PID/Kalman step and the move
itself run on an independent mover thread: the camera/tracking loop just publishes the latest filtered
target into a one-slot mailbox, and a fresher one simply replaces a stale one the mover hasn't gotten to yet -
the stage's G-code handshake is the slow part, and it must never stall the camera loop. f toggles following on
and off; losing the followed cell switches to whichever other track is nearest the frame centre instead of
stopping, so an ordinary track swap doesn't quietly turn follow off - it only stops on f again, or if no tracks
remain at all, or if the stage has moved more than --follow-limit mm from where following began (defense in
depth beyond the per-move cap). image x always drives --axes's first letter, image y its second; which physical
direction is positive on each is a guess: if the followed cell drifts further from centre instead of closer,
stop (f) and retry with --flip-x/--flip-y.
"""

import argparse
import os
import sys
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import torch

from cell_tracker import CellTracker
from test_live import MODES, center_crop, draw, open_camera
from utils import PID, ConstantVelocityKalman

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "enderscopy"))
from enderscope import EnderLights, SerialDevice, Stage  # noqa: E402

AXIS_DIRS = {"x": ("west", "east"), "y": ("south", "north"), "z": ("down", "up")}  # (negative, positive)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}  # position into stage.get_position()'s (X, Y, Z)
AXIS_JOG_KEYS = {"x": ("a", "d"), "y": ("x", "w"), "z": ("c", "e")}  # (negative key, positive key)
AXIS_STEP_ARG = {"x": "x_step", "y": "y_step", "z": "z_step"}
PREV_KEYS, NEXT_KEYS = (81, 82), (83, 84)  # Left/Up, Right/Down (X11 keysym & 0xFF; may differ on other backends)


def parse_axes(axes):
    axes = axes.lower()
    if len(axes) != 2 or set(axes) - set("xyz") or axes[0] == axes[1]:
        raise SystemExit(f"--axes must be two distinct letters from x/y/z, got {axes!r}")
    return axes


def jog_keys_for(axes):
    keys = {}
    for axis in axes:
        neg_key, pos_key = AXIS_JOG_KEYS[axis]
        neg_dir, pos_dir = AXIS_DIRS[axis]
        keys[neg_key], keys[pos_key] = neg_dir, pos_dir
    return keys


class Follow:
    """Nudges the stage to recentre a chosen track. update() (main thread) feeds the cell's centre to a
    dedicated single-point Kalman filter (position + velocity; separate from the main BayesTracker's own
    5-point one, which is tuned for matching/shape, not for smoothing a control signal) and publishes the
    filtered result into a one-slot mailbox. An independent mover thread reads whatever the latest published
    estimate is, runs it through a per-axis PID to get a correction in mm (via the track's own apparent size in
    pixels against --object-size), and sends it to the stage. The two never block each other: a fresher mailbox
    entry just replaces a stale one the mover hasn't gotten to yet, so the stage's serial round-trips (the
    actual slow part) can take as long as they take without stalling the camera loop. Only the mover thread
    touches the stage; id/start/offset mutations beyond it happen on the main thread via update()/cancel(), so
    there's no need to lock those."""

    def __init__(self, stage, size, object_size, gain, ki, kd, max_step, deadband, interval, limit, lookahead,
                 flip_x, flip_y, axes="xz"):
        self.stage, self.size, self.object_size, self.max_step = stage, size, object_size, max_step
        self.deadband, self.interval, self.limit, self.lookahead = deadband, interval, limit, lookahead
        self.flip_x, self.flip_y = flip_x, flip_y
        self.dirs = (AXIS_DIRS[axes[0]], AXIS_DIRS[axes[1]])  # image-x's axis, then image-y's axis
        self.idx = (AXIS_INDEX[axes[0]], AXIS_INDEX[axes[1]])
        self.id = self.start = None
        self.offset = np.zeros(2)  # commanded displacement since start_on, mm, one per axes letter; mover-owned
        self.frame_i = 0
        self.kalman = ConstantVelocityKalman(dim=2, q=4.0, r=16.0)  # tracks the cell's centre (px) for follow only
        self.pid_x = PID(gain, ki, kd, i_limit=max_step, out_limit=max_step)
        self.pid_z = PID(gain, ki, kd, i_limit=max_step, out_limit=max_step)
        self._lock = threading.Lock()
        self._target = None  # mailbox: latest (x, y, vx, vy, size_px) px for the mover thread, or None once consumed
        self._stop = threading.Event()
        self._abort_reason = None
        self._thread = None

    @property
    def active(self):
        return self.id is not None

    @property
    def gain(self):
        return self.pid_x.kp

    @gain.setter
    def gain(self, value):
        self.pid_x.kp = self.pid_z.kp = value

    def start_on(self, track):
        self.id, self.start, self.frame_i = track.id, self.stage.get_position(), 0
        self.offset[:] = 0
        self.kalman.reset([track.x, track.y])
        self.pid_x.reset()
        self.pid_z.reset()
        self._target, self._abort_reason = None, None
        self._stop.clear()
        self._thread = threading.Thread(target=self._mover, daemon=True)
        self._thread.start()
        print(f"following track {self.id} from {self.start}")

    def cancel(self, reason="cancelled"):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        print(f"follow {reason} (track {self.id})")
        self.id = self.start = None

    def _nearest_to_centre(self, tracks):
        if not tracks:
            return None
        c = self.size / 2
        return min(tracks, key=lambda t: (t.x - c) ** 2 + (t.y - c) ** 2)

    def update(self, result):
        """Call every tracking frame from the main thread: feeds the Kalman filter and publishes its latest
        estimate for the mover thread, or finalises a stop the mover decided on its own (lost/strayed). If the
        followed track disappears, switches to whichever other track is nearest the frame centre rather than
        stopping - f toggles tracking on/off, it shouldn't also quietly turn itself off on an ordinary track
        swap. Only the Kalman filter is reset on a switch (it's main-thread-only); the PID state and the stage
        offset/limit accounting carry over, since those belong to the mover thread and to the whole follow
        session, not to any one track."""
        if self._stop.is_set():
            self.cancel(self._abort_reason or "stopped")
            return
        track = next((t for t in result.tracks if t.id == self.id), None)
        if track is None:
            track = self._nearest_to_centre(result.tracks)
            if track is None:
                self.cancel("lost (no other tracks to switch to)")
                return
            print(f"lost track {self.id}, now following nearest-to-centre track {track.id}")
            self.id = track.id
            self.kalman.reset([track.x, track.y])
        self.frame_i += 1
        x, y = self.kalman.step([track.x, track.y])
        if self.frame_i % self.interval:
            return
        vx, vy = self.kalman.velocity
        size_px = max(np.linalg.norm(track.points.max(0) - track.points.min(0)), 1.0)
        with self._lock:
            self._target = (x, y, vx, vy, size_px)

    def position(self, pos):
        """Estimated stage position for the HUD (start + commanded offset so far, not re-queried from the
        hardware; the axis not in axes is never moved, so it stays at its start value); pos: fallback while not
        following."""
        if self.start is None:
            return pos
        p = list(self.start)
        p[self.idx[0]] += self.offset[0]
        p[self.idx[1]] += self.offset[1]
        return tuple(p)

    def _mover(self):
        """Runs on its own thread: moves toward whatever the latest published target is, at whatever pace the
        stage's serial round-trips allow - never the camera loop's pace."""
        while not self._stop.is_set():
            with self._lock:
                target, self._target = self._target, None
            if target is None:
                time.sleep(0.01)
                continue
            x, y, vx, vy, size_px = target
            # aim at where the filtered velocity says the cell will be, not where it was last seen - corrects
            # for the lag between here and whenever this thread actually gets to act on it
            c = self.size / 2
            dx, dy = (x + vx * self.lookahead) - c, (y + vy * self.lookahead) - c
            if dx * dx + dy * dy <= self.deadband * self.deadband:  # squared distance of centre from centre, px^2
                continue
            mm_per_px = self.object_size / size_px
            if abs(dx) > self.deadband:
                step = self.pid_x.step(dx * mm_per_px)
                neg, pos = self.dirs[0]
                towards = neg if (step > 0) != self.flip_x else pos
                self.stage.move_towards(towards, abs(step))
                self.offset[0] += -abs(step) if towards == neg else abs(step)
            if self._stop.is_set():
                break
            if abs(dy) > self.deadband:
                step = self.pid_z.step(dy * mm_per_px)
                neg, pos = self.dirs[1]
                towards = neg if (step > 0) != self.flip_y else pos
                self.stage.move_towards(towards, abs(step))
                self.offset[1] += -abs(step) if towards == neg else abs(step)
            if np.abs(self.offset).max() > self.limit:
                self._abort_reason = f"aborted: strayed past {self.limit} mm from {self.start}"
                self._stop.set()
                break


def resolve_port(port, kind):
    """The given port, or the sole detected serial port; raises with the candidate list otherwise."""
    if port:
        return port
    ports = SerialDevice.serial_ports()
    if len(ports) == 1:
        print(f"{kind} port: {ports[0]} (sole serial port found)")
        return ports[0]
    raise SystemExit(f"can't pick a {kind} port among {ports or '(none found)'}: pass --{kind}-port explicitly")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=os.path.join(HERE, "weights", "cellpoint.pt"))
    p.add_argument("--cam", type=int, default=-1, help="camera index; -1: USB camera if present, else integrated")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--masks", default="none", choices=("none", "fg", "flow"), help="none: points only, the decoder does not run")
    p.add_argument("--no-invert", dest="invert", action="store_false", help="invert is on by default; pass this to turn it off")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--keep-dense", action="store_true",
                    help="keep the dense decoder loaded even with --masks none, so 'm' can switch it on later")
    p.add_argument("--level", type=float, default=0.1, help="outline at sum_i exp(-d_i^2 / 2) = level; lower: larger")
    p.add_argument("--no-stage", action="store_true", help="skip the stage/lights entirely (vision only)")
    p.add_argument("--stage-port", help="e.g. /dev/ttyUSB0; auto-picked if only one serial port is present")
    p.add_argument("--stage-baud", type=int, default=115200)
    p.add_argument("--light-port", help="EnderLights port; omit to run without an illuminator")
    p.add_argument("--light-baud", type=int, default=57600)
    p.add_argument("--home", action="store_true", help="home the stage on startup")
    p.add_argument("--axes", default="xz", help="the two stage axes jog/follow drive, e.g. xz (default), xy, yz")
    p.add_argument("--x-step", type=float, default=5.0, help="mm per jog key in x")
    p.add_argument("--y-step", type=float, default=5.0, help="mm per jog key in y")
    p.add_argument("--z-step", type=float, default=1.0, help="mm per jog key in z")
    p.add_argument("--object-size", type=float, default=0.05,
                    help="assumed real size (mm) of a tracked object, to scale pixel error to a stage move - set this "
                         "to match your sample, e.g. ~0.002 for a 2 micron bacterium, or follow moves will be wrong-scale")
    p.add_argument("--follow-gain", type=float, default=0.5,
                    help="PID Kp: scales the proportional correction (1.0 = the raw pixel-to-mm conversion); "
                         "lower if follow drifts/overshoots instead of settling; +/- adjust this live")
    p.add_argument("--follow-ki", type=float, default=0.0, help="PID Ki: corrects a persistent residual offset; 0 disables")
    p.add_argument("--follow-kd", type=float, default=0.0, help="PID Kd: damps oscillation; 0 disables")
    p.add_argument("--follow-max-step", type=float, default=0.5, help="cap (mm) on a single follow correction")
    p.add_argument("--follow-deadband", type=float, default=20.0,
                    help="radius (px) around centre within which follow does nothing")
    p.add_argument("--follow-interval", type=int, default=1,
                    help="tracking frames between publishing a fresh follow target (the mover thread paces itself "
                         "independently off the stage's own speed, not off this)")
    p.add_argument("--follow-limit", type=float, default=100.0, help="abort follow past this many mm from where it started")
    p.add_argument("--follow-lookahead", type=float, default=3.0,
                    help="frames of the tracker's own velocity estimate to aim ahead of the cell by, to "
                         "compensate for lag between corrections; 0 disables")
    p.add_argument("--flip-x", action="store_true", help="invert the image-x to stage-east/west mapping for follow")
    p.add_argument("--flip-y", action="store_true", help="invert the image-y to stage-up/down (Z) mapping for follow")
    args = p.parse_args()
    args.axes = parse_axes(args.axes)
    jog_keys = jog_keys_for(args.axes)
    key_axis = {k: axis for axis in args.axes for k in AXIS_JOG_KEYS[axis]}
    level = args.level

    device = "cpu" if args.cpu else None
    masks = None if args.masks == "none" else args.masks
    tracker = CellTracker(args.weights, masks=masks, device=device, invert=args.invert, keep_dense=args.keep_dense)
    print(f"device: {tracker.device} ({'GPU' if tracker.device.type == 'cuda' else 'CPU'})"
          + (f" - {torch.cuda.get_device_name(tracker.device)}, half={tracker.half}" if tracker.device.type == "cuda" else ""))
    print(f"dense decoder loaded: {tracker.net.decoder is not None}")

    stage = lights = None
    if not args.no_stage:
        stage = Stage(resolve_port(args.stage_port, "stage"), args.stage_baud, homing=args.home)
        if args.light_port:
            lights = EnderLights(args.light_port, baud_rate=args.light_baud)
    light_on = False
    pos = stage.get_position() if stage else None
    follow = Follow(stage, args.size, args.object_size, args.follow_gain, args.follow_ki, args.follow_kd,
                     args.follow_max_step, args.follow_deadband, args.follow_interval, args.follow_limit,
                     args.follow_lookahead, args.flip_x, args.flip_y, args.axes) if stage else None

    cap = open_camera(args.cam)
    trails, fps = defaultdict(lambda: deque(maxlen=15)), 0.0
    selected_id = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = center_crop(frame, args.size)
            t0 = time.perf_counter()
            result = tracker.update(frame)
            ms = (time.perf_counter() - t0) * 1000
            fps = 0.9 * fps + 0.1 * 1000 / max(ms, 1e-3)
            if follow and follow.active:
                follow.update(result)
                pos = follow.position(pos)
            ids = sorted(t.id for t in result.tracks)
            if selected_id not in ids:
                selected_id = ids[0] if ids else None
            vis = draw(frame, result, trails, tracker.names, level=level)
            if follow and follow.active:
                target = next((t for t in result.tracks if t.id == follow.id), None)
                if target is not None:
                    cv2.drawMarker(vis, (int(target.x), int(target.y)), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            elif selected_id is not None:
                sel = next(t for t in result.tracks if t.id == selected_id)
                cv2.circle(vis, (int(sel.x), int(sel.y)), 10, (255, 255, 255), 1, cv2.LINE_AA)
            stage_info = f" | xyz {pos}" if pos is not None else ""
            follow_info = (f" | follow {follow.id} gain {follow.gain:.2f}" if follow and follow.active
                            else f" | selected {selected_id} gain {follow.gain:.2f}" if follow else "")
            hud = (f"{tracker.device.type} {ms:5.1f} ms | {fps:4.1f} fps | {len(result.tracks)} tracks | "
                   f"masks {tracker.masks} | lv {level:.2f}{stage_info}{follow_info}")
            cv2.rectangle(vis, (0, 0), (args.size, 18), (0, 0, 0), -1)
            cv2.putText(vis, hud, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("cellpoint + enderscope (q to quit)", vis)

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
            elif key in PREV_KEYS or key in NEXT_KEYS:
                if ids:
                    i = ids.index(selected_id)
                    selected_id = ids[(i + (1 if key in NEXT_KEYS else -1)) % len(ids)]
            elif follow and key in (ord("+"), ord("="), ord("-")):
                follow.gain = float(np.clip(follow.gain + (0.1 if key != ord("-") else -0.1), 0.05, 3.0))
                print(f"follow gain: {follow.gain:.2f}")
            elif follow and key == ord("f"):
                if follow.active:
                    follow.cancel()
                elif selected_id is not None:
                    follow.start_on(next(t for t in result.tracks if t.id == selected_id))
                else:
                    print("no tracks to follow")
            elif follow and follow.active and chr(key) in (*jog_keys, "h"):
                print("stop following (f) before jogging")
            elif stage and chr(key) in jog_keys:
                step = getattr(args, AXIS_STEP_ARG[key_axis[chr(key)]])
                stage.move_towards(jog_keys[chr(key)], step)
                pos = stage.get_position()
            elif stage and key == ord("h"):
                stage.home()
                pos = stage.get_position()
            elif lights and key == ord("l"):
                light_on = not light_on
                lights.shutter(light_on)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if follow and follow.active:
            follow.cancel("stopped")
        if stage:
            stage.serial.close()
        if lights:
            lights.serial.close()


if __name__ == "__main__":
    main()
