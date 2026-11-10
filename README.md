# cellpoint

Real-time detection, segmentation and tracking of bacteria and other cells in microscopy video.

A small network finds each cell's centre and 4 eigen points, and how far the cell extends around each point. Those give
every cell a Gaussian shape with an outline, without a segmentation decoder. A Bayesian tracker follows the cells: each
detection is matched by the symmetric KL divergence between shapes, and existence is a probability updated on every
frame. An optional dense head adds a foreground map, which gives masks and a better estimate of which detections are real.

## Install

```
pip install -r requirements.txt
```

## Use

```python
from cell_tracker import CellTracker
from tracking.gmm import outline

tracker = CellTracker("weights/cellpoint.pt")   # points only; masks="fg" or "flow" adds the dense head
result = tracker.update(frame)                  # (H, W) grayscale or (H, W, 3) BGR

result.tracks                                   # [Track(id, y, x, vy, vx, p_exist, cls, cls_prob, points, shape)]
result.labels                                   # (H, W) int64 masks holding track ids, 0 = background; None if points only
outline(result.tracks[0].points, result.tracks[0].shape)   # a cell's outline as (N, 2) x, y
```

| `masks=` | What you get | 640 x 640, GPU (RTX A4000 Laptop) | 640 x 640, CPU (4 threads) |
|---|---|---|---|
| `None` (default) | Tracked cells with outlines | 18 ms | 49 ms |
| `"fg"` | Also masks from the foreground map, and better tracking | 40 ms | 139 ms |
| `"flow"` | Also masks from following the predicted flow | 42 ms | 170 ms |

Timed on `update` with about 90 tracked cells, without drawing. Call `tracker.reset()` between videos.

Tracking on 51 held-out frames of one bacteria video, with frames skipped (IDF1, higher is better):

| `masks=` | Every frame | Every 3rd frame | Every 5th frame |
|---|---|---|---|
| `None` | 0.906 | 0.917 | 0.898 |
| `"fg"` | 0.943 | 0.944 | 0.925 |
| `"flow"` | 0.926 | 0.934 | 0.919 |

That video has no touching cells, so these numbers do not show how tracking behaves when cells touch.

Classes: `~` long rod, `-` short rod or oval, `o` round, `x` irregular.

## Live webcam

```
python test_live.py
python test_live.py --cam 0 --masks fg --cpu
```

Uses a USB camera if one is present, otherwise the integrated one. Keys: `q` quit, `m` masks (off / fg / flow),
`i` invert, `r` reset tracks, `s` save frame.

## Enderscope

```
pip install -r requirements-enderscope.txt
python enderscope_test.py --stage-port /dev/ttyUSB0       # small bounded moves, run this first
python enderscope_live.py --stage-port /dev/ttyUSB0
python enderscope_live.py --no-stage                      # vision only, no hardware
```

Drives an [Enderscope](https://github.com/mutterer/enderscopy) (a 3D-printer-chassis microscope) while tracking,
on the scope's own camera via OpenCV. Only X (east/west) and Z (up/down) are ever driven - this rig's camera
moves with those two, not with Y (the bed's front-to-back travel) - so Y is never jogged and never used by follow.

### Keys (`enderscope_live.py`)

| Key | Action |
|---|---|
| `q` / `Esc` | Quit |
| `m` | Cycle masks (off / fg / flow) |
| `i` | Toggle invert |
| `r` | Reset tracks |
| `s` | Save frame |
| `[` / `]` | Outline level down / up (lower = larger outline) |
| `a` / `d` | Jog stage west / east (X) |
| `e` / `c` | Jog stage up / down (Z) |
| `h` | Home stage |
| `l` | Toggle illuminator light |
| `←`/`→` (or `↑`/`↓`) | Select a tracked cell (white ring) |
| `f` | Follow the selected cell (yellow cross) until it's lost, strays past `--follow-limit`, or `f` again |
| `+` / `-` | Increase / decrease follow gain (speed of the follow correction) |

Jogging and home are disabled while `f` is actively following - press `f` again to stop first.

`f` has the stage follow the selected cell: its pixel offset from frame centre is converted to mm using its own
apparent size in pixels against `--object-size` (set this to the real size of your sample, e.g. ~0.002mm for a
2 micron bacterium, or the moves will be the wrong scale), scaled by `--follow-gain` (lower it if follow
drifts/overshoots instead of settling - `+`/`-` adjust this live), capped at `--follow-max-step` mm per
correction, and stopped if the track is lost or the stage strays past `--follow-limit` mm from where it started.
The camera/tracking loop and the stage move independently on separate threads - the stage's serial round-trips
are the slow part, and publishing a fresh target just replaces a stale one the mover hasn't gotten to yet, so
the camera/display frame rate is never held hostage by how fast the hardware responds.

`--cpu` forces the model onto CPU (no GPU/CUDA needed at all), in case this runs on a machine without a GPU.

`third_party/enderscopy` is an
unmodified copy of its `enderscope` package (pinned to commit `ea935a7`, with its original `LICENSE`), not
`pip install`ed: its `pyproject.toml` has a placeholder author email that current `setuptools` rejects outright,
so `pip install` (even `--no-deps`) fails before it gets anywhere near resolving dependencies like `picamera2`.
A plain file copy sidesteps that entirely - `enderscope_live.py`/`enderscope_test.py` just add it to `sys.path`
and `import enderscope` normally. Requires the user to be in the `dialout` group
(`sudo usermod -aG dialout $USER`, then re-login) to open the serial port at all.

To refresh the vendored copy from a newer commit, re-download `enderscope/enderscope.py`, `enderscope/__init__.py`
and `LICENSE` from the repo at the desired ref into `third_party/enderscopy/`.

`enderscope_test.py` is a small bounded motion smoke test for first contact with a stage: it nudges one axis at a
time by `--step` mm and back, checking the reported position against `--limit` mm from the start after every move,
and aborts on Ctrl+C between moves. Run it before `enderscope_live.py` on any stage/wiring you haven't moved
through this code before.

## Video or TIFF stack

```
python test_on_video.py movie.mp4
python test_on_video.py movie.mp4 --masks fg --contours
python test_on_video.py stack.tif --fps 10 --out result.mp4
```

Saves an mp4 with outlines or masks and tracks drawn as in the live view. `--contours` draws outlines with `--masks fg`.

## Files

| File | Contents |
|---|---|
| `cell_tracker.py` | `CellTracker`: frame in, tracks and masks out |
| `tracking/bayes_tracker.py` | The Bayesian tracker |
| `tracking/cell_probability.py` | How likely a detection is to be a real cell |
| `tracking/gmm.py` | A cell's Gaussians and their outline |
| `tracking/masks.py` | Masks from the foreground map and flow |
| `model.py` | The network |
| `utils/kalman.py` | A reusable constant-velocity Kalman filter |
| `utils/pid.py` | A reusable PID controller |
| `test_live.py` | Webcam view |
| `enderscope_live.py` | Webcam view driven by an Enderscope stage/illuminator |
| `enderscope_test.py` | Small bounded stage-motion smoke test |
| `third_party/enderscopy/` | Vendored copy of the `enderscope` package (`Stage`, `EnderLights`, ...) |
| `third_party/ultralytics_cfg/` | Vendored copy of `yolo26.yaml` (AGPL-3.0, see its `LICENSE`) |
| `test_on_video.py` | Process a video or TIFF stack, save the result |
