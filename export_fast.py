"""Export the network for fast inference at a fixed frame size: a TensorRT FP16 engine for this machine's GPU, an
OpenVINO model for the CPU (and an ONNX file, which both are built from and onnxruntime can run). Two networks:
full (points + dense head, for masks) and points (the dense decoder cut off, the backbone stopped at P3: detections
only, the fastest).

    python export_fast.py                      # both networks, all three formats, 640 x 640
    python export_fast.py --kind points --only openvino
    python export_fast.py --size 512

Outputs, in weights/: cellpoint_<size>[_points].onnx, cellpoint_<size>[_points]_fp16.engine,
cellpoint_<size>[_points]_openvino/. A TensorRT engine only runs on the GPU model and TensorRT version that built it:
rebuild after changing either.

The exported graph (FastNet) takes the raw uint8 gray frame and the two normalisation bounds, so normalising, the
second (empty) input channel, the sigmoids and the 3 x 3 peak search all happen inside the engine. It returns
    grid (1, 1 + nc + 2 + 2k + 3P + E, H/8, W/8): peak (centre probability where it is a 3 x 3 maximum, else 0),
          class probabilities, centre offset, eigen points, shape and error Cholesky factors - as CellNet.forward
    dense (1, 3, H, W): fg probability, flow dy, dx (full network only)
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import P3, STRIDE, TAPS, CellNet

HERE = os.path.dirname(os.path.abspath(__file__))


class FastNet(nn.Module):
    def __init__(self, net, half=False, points=False):
        """points: drop the dense decoder and the backbone after P3; returns grid only."""
        super().__init__()
        self.net, self.fp16, self.points_only = (net.points_only() if points else net), half, points
        conv = net.adapter.conv                  # the second input channel is always zero: keep only the first
        self.inp = nn.Conv2d(1, conv.out_channels, 1)
        with torch.no_grad():
            self.inp.weight.copy_(conv.weight[:, :1])
            self.inp.bias.copy_(conv.bias)

    def forward(self, img, lohi):
        """img (1, 1, H, W) uint8, lohi (2,) float32: x = clip((img - lo) / (hi - lo), 0, 1). Swapping lo and hi
        inverts."""
        x = ((img.float() - lohi[0]) / (lohi[1] - lohi[0])).clamp(0, 1)
        x = self.inp(x.half() if self.fp16 else x)
        feats = None if self.points_only else self.net.features(x)
        p3 = self.net.features(x, (TAPS[P3],))[0] if self.points_only else feats[P3]
        pts, cov = self.net.points(p3), self.net.cov(p3)
        heat = pts["cls"].sigmoid()
        best = heat.amax(1, keepdim=True)
        peak = best * (best == F.max_pool2d(best, 3, 1, 1)).to(best.dtype)
        grid = torch.cat([peak, heat, pts["off"], pts["kpt"], cov["shape"], cov["err"]], 1)
        if self.points_only:
            return grid
        d = self.net.decoder(feats)
        dense = torch.cat([d[:, 4:5].sigmoid(), d[:, 2:4]], 1)
        return grid, dense


def load_fastnet(weights, half=False, points=False):
    """FastNet (fused, eval; with half, the network in fp16 on the GPU) and the checkpoint."""
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    net = CellNet(len(ck["names"]), ck["k"])
    net.load_state_dict(ck["model"])
    model = FastNet(net.eval().fuse(), half, points).eval()
    return (model.cuda().half() if half else model), ck


def export_onnx(weights, path, size, half=False, points=False):
    """half: an fp16 graph (fp16 weights and outputs; the input stays uint8, normalising stays fp32)."""
    model, _ = load_fastnet(weights, half, points)
    dev = "cuda" if half else "cpu"
    img, lohi = torch.zeros(1, 1, size, size, dtype=torch.uint8, device=dev), torch.tensor([0.0, 255.0], device=dev)
    with torch.no_grad():
        torch.onnx.export(model, (img, lohi), path, input_names=["img", "lohi"],
                          output_names=["grid"] if points else ["grid", "dense"],
                          opset_version=17, dynamo=False)
    print("onnx:", path)


def build_trt(onnx_path, path, workspace_gb=2):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))  # fp16 from the graph
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.builder_optimization_level = 5
    t = time.time()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("TensorRT build failed")
    with open(path, "wb") as f:
        f.write(engine)
    print(f"tensorrt: {path} ({time.time() - t:.0f} s, {torch.cuda.get_device_name()}, trt {trt.__version__})")


def build_openvino(onnx_path, path):
    import openvino as ov
    ov.save_model(ov.convert_model(onnx_path), os.path.join(path, "cellpoint.xml"), compress_to_fp16=False)
    print("openvino:", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=os.path.join(HERE, "weights", "cellpoint.pt"))
    p.add_argument("--size", type=int, default=640, help=f"square frame size, a multiple of {STRIDE}")
    p.add_argument("--only", choices=("onnx", "tensorrt", "openvino"), help="build just this format")
    p.add_argument("--kind", choices=("full", "points"), help="build just this network")
    args = p.parse_args()
    if args.size % STRIDE:
        raise SystemExit(f"--size must be a multiple of {STRIDE}")
    for kind in [args.kind] if args.kind else ["full", "points"]:
        points = kind == "points"
        out = os.path.join(os.path.dirname(args.weights), f"cellpoint_{args.size}" + ("_points" if points else ""))
        onnx_cpu, onnx_gpu = out + ".onnx", out + "_fp16.onnx"
        if args.only in (None, "onnx", "openvino"):
            export_onnx(args.weights, onnx_cpu, args.size, points=points)
        if args.only in (None, "openvino"):
            build_openvino(onnx_cpu, out + "_openvino")
        if args.only in (None, "tensorrt"):
            export_onnx(args.weights, onnx_gpu, args.size, half=True, points=points)   # TensorRT takes fp16 from the graph
            build_trt(onnx_gpu, out + "_fp16.engine")
            os.remove(onnx_gpu)


if __name__ == "__main__":
    main()
