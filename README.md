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
from gmm import outline

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
| `bayes_tracker.py` | The Bayesian tracker |
| `cell_probability.py` | How likely a detection is to be a real cell |
| `gmm.py` | A cell's Gaussians and their outline |
| `masks.py` | Masks from the foreground map and flow |
| `model.py` | The network |
| `test_live.py` | Webcam view |
| `test_on_video.py` | Process a video or TIFF stack, save the result |
