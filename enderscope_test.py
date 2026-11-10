"""First-contact stage test: nudges one axis at a time by --step mm and back to where it started, checking the
reported position against --limit mm from the start after every single move. A move that strays past the limit
aborts immediately instead of continuing the sequence. Ctrl+C aborts between moves at any time.

    python enderscope_test.py --stage-port /dev/ttyUSB0
    python enderscope_test.py --stage-port /dev/ttyUSB0 --step 2 --repeats 3 --limit 15 --axes xy
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "enderscopy"))
from enderscope import SerialDevice, Stage  # noqa: E402


def resolve_port(port):
    if port:
        return port
    ports = SerialDevice.serial_ports()
    if len(ports) == 1:
        print(f"stage port: {ports[0]} (sole serial port found)")
        return ports[0]
    raise SystemExit(f"can't pick a stage port among {ports or '(none found)'}: pass --stage-port explicitly")


def strayed(start, pos, limit):
    return any(abs(p - s) > limit for p, s in zip(pos, start))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage-port", help="e.g. /dev/ttyUSB0; auto-picked if only one serial port is present")
    p.add_argument("--stage-baud", type=int, default=115200)
    p.add_argument("--step", type=float, default=1.0, help="mm per nudge")
    p.add_argument("--repeats", type=int, default=2, help="out-and-back repeats per axis")
    p.add_argument("--limit", type=float, default=10.0, help="abort if any axis strays more than this many mm from the start")
    p.add_argument("--axes", default="xyz", help="axes to test, e.g. xy to skip z")
    p.add_argument("--settle", type=float, default=0.5, help="seconds to pause after each move, for a visual check")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()

    stage = Stage(resolve_port(args.stage_port), args.stage_baud)
    try:
        start = stage.get_position()
        print(f"start position: {start}  (not homed: this is relative to whatever the board booted up with)")
        print(f"plan: axes {list(args.axes)}, step {args.step} mm x {args.repeats} out-and-back, "
              f"abort past {args.limit} mm from start")
        if not args.yes:
            input("press Enter to start moving, Ctrl+C to abort ")

        for axis in args.axes:
            print(f"--- axis {axis} ---")
            for i in range(args.repeats):
                for sign in (1, -1):
                    stage.move_axis(axis, sign * args.step)
                    stage.finish_moves()
                    time.sleep(args.settle)
                    pos = stage.get_position()
                    print(f"  [{i + 1}/{args.repeats}] {axis}{'+' if sign > 0 else '-'}{args.step}: {pos}")
                    if strayed(start, pos, args.limit):
                        print(f"ABORT: {pos} is more than {args.limit} mm from start {start} on some axis")
                        return
        print("done: all axes returned within the limit")
    except KeyboardInterrupt:
        print("\naborted by user")
    finally:
        pos = stage.get_position()
        print(f"final position: {pos}")
        stage.serial.close()


if __name__ == "__main__":
    main()
