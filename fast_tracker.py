"""FastCellTracker: CellTracker on an exported network (export_fast.py) and NumPy post-processing.

    tracker = FastCellTracker()                         # masks "fg"; the GPU engine if there is a GPU, else OpenVINO
    tracker = FastCellTracker(masks=None)               # points only: the network without its dense decoder
    tracker = FastCellTracker("weights/cellpoint_640_openvino", masks="flow")
    result = tracker.update(frame)          # frame: (S, S) gray or (S, S, 3) BGR uint8, S the exported size
    result.tracks, result.labels            # labels hold track ids, as CellTracker; None with masks=None

The backend follows the model path: .engine TensorRT, an OpenVINO directory or .xml, .onnx onnxruntime, .pt the
torch network itself (no export needed: slow, for checking). tracker.ms holds the last frame's net / post / track ms.
"""

import os
import time

import cv2
import numpy as np
import torch

from cell_tracker import Result
from tracking.bayes_tracker import BayesTracker
from tracking.cell_probability import calibrated_score
from tracking.fast_post import decode, fg_masks, fg_probability, flow_gmm_masks, lohi
from tracking.masks import UNTRACKED

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "cellpoint.pt")
MASKS = ("fg", "flow", None)


class TensorRTNet:
    """A FastNet engine on the GPU; pinned host buffers, one CUDA stream. fp16 outputs are cast to fp32 on the GPU
    (numpy's own fp16 -> fp32 conversion costs more than the network)."""
    device = "gpu"

    def __init__(self, path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"cannot load {path}: built for another GPU or TensorRT version? rerun export_fast.py")
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        types = {trt.uint8: torch.uint8, trt.float16: torch.float16, trt.float32: torch.float32}
        self.dev, self.dev32, self.host, self.outputs = {}, {}, {}, []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape, dtype = tuple(self.engine.get_tensor_shape(name)), types[self.engine.get_tensor_dtype(name)]
            self.dev[name] = torch.empty(shape, dtype=dtype, device="cuda")
            self.ctx.set_tensor_address(name, self.dev[name].data_ptr())
            out = self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            if out:
                self.outputs.append(name)
                self.dev32[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            self.host[name] = torch.empty(shape, dtype=torch.float32 if out else dtype, pin_memory=True)
        self.size = self.dev["img"].shape[-1]
        self.name = f"tensorrt {torch.cuda.get_device_name()}"

    def __call__(self, gray, bounds):
        """(grid, dense or None) float32 views of pinned buffers, valid until the next call."""
        self.host["img"][0, 0].numpy()[:] = gray
        self.host["lohi"].numpy()[:] = bounds
        with torch.cuda.stream(self.stream):
            for name in ("img", "lohi"):
                self.dev[name].copy_(self.host[name], non_blocking=True)
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            for name in self.outputs:
                self.dev32[name].copy_(self.dev[name])
                self.host[name].copy_(self.dev32[name], non_blocking=True)
        self.stream.synchronize()
        return self.host["grid"].numpy()[0], self.host["dense"].numpy()[0] if "dense" in self.host else None


class OpenVINONet:
    device = "cpu"

    def __init__(self, path, threads=None):
        import openvino as ov
        xml = os.path.join(path, "cellpoint.xml") if os.path.isdir(path) else path
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            cfg["INFERENCE_NUM_THREADS"] = threads
        compiled = ov.Core().compile_model(xml, "CPU", cfg)
        self.n_out = len(compiled.outputs)
        self.req = compiled.create_infer_request()
        self.size = self.req.get_input_tensor(0).shape[-1]
        self.name = "openvino cpu"

    def __call__(self, gray, bounds):
        self.req.infer({"img": gray[None, None], "lohi": bounds})
        return self.req.get_output_tensor(0).data[0], self.req.get_output_tensor(1).data[0] if self.n_out > 1 else None


class OnnxNet:
    device = "cpu"

    def __init__(self, path, threads=None):
        import onnxruntime as ort
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opt.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opt, providers=["CPUExecutionProvider"])
        self.size = self.sess.get_inputs()[0].shape[-1]
        self.name = "onnxruntime cpu"

    def __call__(self, gray, bounds):
        out = self.sess.run(None, {"img": gray[None, None], "lohi": bounds})
        return out[0][0], out[1][0] if len(out) > 1 else None


class TorchNet:
    """FastNet in eager torch, any size: for checking the exports."""

    def __init__(self, weights, device=None, size=640, points=False):
        from export_fast import load_fastnet
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_fastnet(weights, points=points)[0].to(dev)
        self.dev, self.device = dev, "gpu" if dev == "cuda" else "cpu"
        self.size, self.name = size, f"torch {dev}"

    @torch.no_grad()
    def __call__(self, gray, bounds):
        out = self.net(torch.from_numpy(gray)[None, None].to(self.dev), torch.from_numpy(bounds).to(self.dev))
        grid, dense = (out, None) if torch.is_tensor(out) else out
        return grid[0].cpu().numpy(), None if dense is None else dense[0].cpu().numpy()


def default_model(size=640, device=None, points=False):
    """The TensorRT engine if there is a GPU and the engine was built, else the OpenVINO model, else the ONNX file;
    points: the network without its dense decoder."""
    base = os.path.join(HERE, "weights", f"cellpoint_{size}" + ("_points" if points else ""))
    order = (["_fp16.engine"] if device != "cpu" and torch.cuda.is_available() else []) + ["_openvino", ".onnx"]
    for ext in order:
        if os.path.exists(base + ext):
            return base + ext
    raise SystemExit(f"no exported {'points' if points else 'full'} model for size {size} in weights/: "
                     f"run python export_fast.py --size {size}")


def load_net(model, threads=None, device=None, size=640, points=False):
    if model.endswith(".engine"):
        return TensorRTNet(model)
    if model.endswith(".onnx"):
        return OnnxNet(model, threads)
    if model.endswith(".pt"):
        return TorchNet(model, device, size, points)
    return OpenVINONet(model, threads)


class FastCellTracker:
    def __init__(self, model=None, weights=WEIGHTS, masks="fg", min_score=0.05, fg_thr=0.7, invert=False,
                 threads=None, device=None, size=640, flow_kw=None, show_untracked=False, points_model=None,
                 **tracker_kw):
        """model: the full network (export_fast.py output; default: default_model(size, device)); points_model: the
        network without its dense decoder, for masks=None (default: default_model(size, device, points=True)). Only
        the one the mask mode needs is loaded, the other when set_masks first needs it. A .pt checkpoint serves as
        both. weights: the checkpoint, for the class names and tracking parameters.
        masks: "fg" (connected fg regions, fastest masks), "flow" (fg pixels follow the smoothed flow and join the
        detection whose Gaussians they end in: masks keep their track's id) or None (points only: no dense output,
        the fastest; outlines from each track's Gaussians).
        flow_kw: options of tracking.fast_post.flow_gmm_masks (sigma, steps, min_level, ...); show_untracked: also
        return masks without a reported track (ids >= masks.UNTRACKED)."""
        self.check_masks(masks)
        self.models = {"dense": model, "points": points_model or (model if model and model.endswith(".pt") else None)}
        self.nets, self.threads, self.device_arg, self.size_arg = {}, threads, device, size
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.names, self.k, self.params = ck["names"], ck["k"], ck["tracking"]
        self.masks, self.min_score, self.fg_thr, self.invert = masks, min_score, fg_thr, invert
        self.flow_kw, self.show_untracked = flow_kw or {}, show_untracked
        make = lambda key: BayesTracker(self.params["Q"], nc=len(self.names), k=self.k,
                                        **{**self.params[key], **tracker_kw})
        self.trackers = {"points": make("points_tracker"), "dense": make("fg_tracker")}
        self.ms = {"net": 0.0, "post": 0.0, "track": 0.0}
        self.size = self.net.size

    @staticmethod
    def check_masks(masks):
        if masks not in MASKS:
            raise ValueError(f"unknown masks {masks!r}: use 'fg', 'flow' or None")

    @property
    def kind(self):
        return "dense" if self.masks else "points"

    @property
    def net(self):
        """The network the mask mode needs, loaded on first use."""
        if self.kind not in self.nets:
            path = self.models[self.kind] or default_model(self.size_arg, self.device_arg, self.kind == "points")
            self.models[self.kind] = path
            self.nets[self.kind] = load_net(path, self.threads, self.device_arg, self.size_arg, self.kind == "points")
            if hasattr(self, "size") and self.nets[self.kind].size != self.size:
                raise SystemExit(f"{path} is {self.nets[self.kind].size} px, the other network {self.size} px")
        return self.nets[self.kind]

    @property
    def model(self):
        self.net
        return self.models[self.kind]

    @property
    def tracker(self):
        return self.trackers[self.kind]

    @property
    def backend(self):
        return self.net.name

    @property
    def device(self):
        """"gpu" or "cpu"."""
        return self.net.device

    def set_masks(self, masks):
        """Change the mask mode (loading the other network if needed); tracks start again."""
        self.check_masks(masks)
        self.masks = masks
        self.net
        self.reset()

    def reset(self):
        """Forget all tracks (new video)."""
        for t in self.trackers.values():
            t.reset()

    def infer(self, frame, copy=False):
        """The network's (grid, dense) for one frame, dense None with masks=None. Without copy they are the backend's
        buffers, overwritten by the next call; copy them to run the network on the next frame while this one is
        post-processed."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if gray.shape != (self.size, self.size) or gray.dtype != np.uint8:
            raise ValueError(f"frame must be uint8 {self.size} x {self.size} (the exported size), got {gray.dtype} {gray.shape}")
        t0 = time.perf_counter()
        grid, dense = self.net(np.ascontiguousarray(gray), lohi(gray, self.invert))
        if copy:
            grid, dense = grid.copy(), None if dense is None else dense.copy()
        self.ms["net"] = (time.perf_counter() - t0) * 1e3
        return grid, dense

    def detect(self, frame, out=None):
        """Network and P(real) of one frame: what BayesTracker.update takes, plus the masks (None with masks=None).
        out: infer(frame) if already run, with the same mask mode."""
        grid, dense = out if out is not None else self.infer(frame)
        t1 = time.perf_counter()
        det = decode(grid, len(self.names), self.k, self.min_score)
        points = np.concatenate([det["yx"][:, None], det["kpts"]], 1)
        n = len(points)
        labels = None
        if dense is None or not self.masks:       # points only: P(real) is the calibrated point score
            prob = calibrated_score(det["score"].astype(np.float64), *self.params["score_calibration"])
        else:
            fg, flow = dense[0], dense[1:3]
            prob = fg_probability(points, det["score"], fg, flow, self.params["logistic"]) if n else np.zeros(0)
            if self.masks == "fg":
                labels = fg_masks(fg, self.fg_thr)
            else:
                labels = flow_gmm_masks(fg, flow, points, det["shape"], prob, self.fg_thr, **self.flow_kw)
        self.ms["post"] = (time.perf_counter() - t1) * 1e3
        f64 = lambda a: a.astype(np.float64)
        return dict(mu=f64(points.reshape(n, -1)), R=f64(det["R"]) + 1e-3 * np.eye(2 * (self.k + 1)),
                    shape=f64(det["shape"]), prob=prob, cls_prob=f64(det["cls_prob"]), labels=labels)

    def track(self, det):
        """Tracks, and masks labelled with their track's id. Unless show_untracked, only the masks of the reported
        tracks are kept: fg no track holds and tentative tracks (p_exist < p_show) are background."""
        t0 = time.perf_counter()
        tracks = self.tracker.update(**det)
        labels = det["labels"]
        if labels is None:
            self.ms["track"] = (time.perf_counter() - t0) * 1e3
            return Result(tracks, None)
        M = len(det["prob"])
        lut = np.arange(max(int(labels.max()), M) + 1, dtype=np.int64) + UNTRACKED     # mask number -> id
        lut[0] = 0
        if self.masks == "fg":                    # a mask takes the most probable track whose centre lies inside
            ids, yx, p = self.tracker.current()
            H, W = labels.shape
            at = labels[np.clip(np.round(yx[:, 0]).astype(int), 0, H - 1), np.clip(np.round(yx[:, 1]).astype(int), 0, W - 1)]
            for j in np.argsort(p):               # most probable written last
                if at[j] > 0:
                    lut[at[j]] = ids[j]
        else:                                     # a mask is its detection's: it takes that detection's track
            ids = self.tracker.det_ids
            lut[1:M + 1] = np.where(ids > 0, ids, lut[1:M + 1])
        if not self.show_untracked:
            lut[~np.isin(lut, [t.id for t in tracks])] = 0
        res = Result(tracks, np.take(lut, labels))
        self.ms["track"] = (time.perf_counter() - t0) * 1e3
        return res

    def update(self, frame):
        return self.track(self.detect(frame))
