#!/usr/bin/env python3
"""Validate VRCaps_lumen_100 against the patched PR-ENDO reader."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from PIL import Image
import torch

TEST_INDICES = [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95]
TRAIN_INDICES = [i for i in range(100) if i not in TEST_INDICES]
EXPECTED = {
    "w": 1350,
    "h": 1080,
    "fl_x": 664.6633833646774,
    "fl_y": 664.6633672714233,
    "cx": 675.0,
    "cy": 540.0,
    "camera_angle_x": 1.586227683125316,
}


def install_tinycudann_stub():
    """Allow the reader import on a GPU-less node; the reader never uses tcnn."""
    stub = types.ModuleType("tinycudann")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tinycudann stub: GPU-only component instantiated")

    stub.Encoding = _Dummy
    stub.Network = _Dummy
    stub.NetworkWithInputEncoding = _Dummy
    stub.supports_jit_fusion = lambda: False
    stub.free_temporary_memory = lambda: None
    sys.modules["tinycudann"] = stub


def fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov / 2.0))


def focal2fov(focal, pixels):
    return 2.0 * math.atan(pixels / (2.0 * focal))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_metadata(data: Path, report: dict):
    for name, indices in (
        ("transforms.json", TRAIN_INDICES),
        ("transforms_test.json", TEST_INDICES),
    ):
        meta = load_json(data / name)
        for key, expected in EXPECTED.items():
            if key not in meta:
                report["errors"].append(f"{name}: missing {key}")
            elif key in ("w", "h"):
                if int(meta[key]) != int(expected):
                    report["errors"].append(f"{name}: {key}={meta[key]} expected {expected}")
            elif abs(float(meta[key]) - expected) > 1e-9 * max(1.0, abs(expected)):
                report["errors"].append(f"{name}: {key}={meta[key]} expected {expected}")
        frames = meta.get("frames", [])
        got = sorted(int(str(f["file_path"])) for f in frames)
        if got != indices:
            report["errors"].append(f"{name}: frame indices mismatch")
        for frame in frames:
            m = np.asarray(frame["transform_matrix"], dtype=np.float64)
            if m.shape != (4, 4):
                report["errors"].append(f"{name}: bad transform for {frame['file_path']}")
                continue
            if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0]):
                report["errors"].append(f"{name}: bad homogeneous row for {frame['file_path']}")
            if abs(np.linalg.det(m[:3, :3]) - 1.0) > 1e-3:
                report["errors"].append(f"{name}: bad rotation determinant for {frame['file_path']}")
        report["splits"][name] = len(frames)


def check_depth_files(data: Path, report: dict):
    stats = []
    sample = TRAIN_INDICES[:3] + [TRAIN_INDICES[len(TRAIN_INDICES) // 2]]
    for idx in sample:
        p = data / "depth_train" / f"{idx:04d}.tiff"
        arr = np.asarray(Image.open(p))
        if arr.ndim != 2:
            report["errors"].append(f"{p}: expected 2D, got {arr.shape}")
            continue
        if arr.dtype != np.float32:
            report["errors"].append(f"{p}: expected float32, got {arr.dtype}")
        if arr.shape != (EXPECTED["h"], EXPECTED["w"]):
            report["errors"].append(f"{p}: shape {arr.shape}")
        if not np.isfinite(arr).all() or (arr <= 0).any():
            report["errors"].append(f"{p}: non-finite or non-positive depth")
        stats.append(
            {
                "frame": idx,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "p1": float(np.percentile(arr, 1)),
                "p50": float(np.percentile(arr, 50)),
                "p99": float(np.percentile(arr, 99)),
            }
        )
    report["depth_stats"] = stats


def _w2c_from_cam(cam):
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = cam.R.T
    w2c[:3, 3] = cam.T
    return w2c


def check_reader(data: Path, repo: Path, report: dict):
    sys.path.insert(0, str(repo))
    from scene.dataset_readers import readVRCapsSceneInfo

    info = readVRCapsSceneInfo(str(data))
    report["reader"] = {
        "train": len(info.train_cameras),
        "test": len(info.test_cameras),
    }
    if len(info.train_cameras) != 88:
        report["errors"].append(f"reader train count {len(info.train_cameras)}")
    if len(info.test_cameras) != 12:
        report["errors"].append(f"reader test count {len(info.test_cameras)}")
    if [c.image_name for c in info.train_cameras] != [f"{i:04d}" for i in TRAIN_INDICES]:
        report["errors"].append("reader train order/names mismatch")
    if [c.image_name for c in info.test_cameras] != [f"{i:04d}" for i in TEST_INDICES]:
        report["errors"].append("reader test order/names mismatch")

    max_rel = 0.0
    for cam in info.train_cameras[:3]:
        raw = np.asarray(Image.open(cam.depth_path), dtype=np.float64)
        target = np.asarray(cam.depth, dtype=np.float64)
        if target.ndim == 3:
            target = target[..., 0]
        # CameraInfo.depth stores raw / depth_scale * 255; PILtoTorch divides by 255.
        target = target / 255.0
        max_rel = max(max_rel, float(np.max(np.abs(target - raw) / np.maximum(np.abs(raw), 1e-8))))
    report["depth_target_max_rel_err"] = max_rel
    if max_rel > 1e-5:
        report["errors"].append(f"depth target max rel err {max_rel}")

    cam = info.train_cameras[0]
    fx = fov2focal(cam.FovX, cam.width)
    fy = fov2focal(cam.FovY, cam.height)
    report["intrinsics"] = {"fx": fx, "fy": fy, "cx": cam.cx, "cy": cam.cy}
    for got, expected, key in (
        (fx, EXPECTED["fl_x"], "fx"),
        (fy, EXPECTED["fl_y"], "fy"),
        (cam.cx, EXPECTED["cx"], "cx"),
        (cam.cy, EXPECTED["cy"], "cy"),
    ):
        if abs(got - expected) > 1e-6 * max(1.0, abs(expected)):
            report["errors"].append(f"{key} mismatch {got} vs {expected}")
    return info


def check_roundtrip(info, report: dict, n=10000):
    max_err = 0.0
    rng = np.random.default_rng(0)
    for cam in info.train_cameras[:3]:
        raw = np.asarray(Image.open(cam.depth_path), dtype=np.float64)
        h, w = raw.shape
        v = rng.integers(0, h, size=n)
        u = rng.integers(0, w, size=n)
        d = raw[v, u]
        keep = np.isfinite(d) & (d > 0)
        u, v, d = u[keep], v[keep], d[keep]
        if u.size == 0:
            continue
        fx = fov2focal(cam.FovX, cam.width)
        fy = fov2focal(cam.FovY, cam.height)
        xc = (u + 0.5 - cam.cx) / fx * d
        yc = (v + 0.5 - cam.cy) / fy * d
        zc = d
        w2c = _w2c_from_cam(cam)
        c2w = np.linalg.inv(w2c)
        pts = np.stack([xc, yc, zc, np.ones_like(xc)], axis=1) @ c2w.T
        pc = pts @ w2c.T
        u_rec = fx * pc[:, 0] / pc[:, 2] + cam.cx - 0.5
        v_rec = fy * pc[:, 1] / pc[:, 2] + cam.cy - 0.5
        max_err = max(max_err, float(np.max(np.hypot(u_rec - u, v_rec - v))))
    report["roundtrip_max_err_px"] = max_err
    if max_err > 0.01:
        report["errors"].append(f"roundtrip max error {max_err} px")


def check_ply(info, data: Path, report: dict):
    from plyfile import PlyData

    vertices = PlyData.read(str(data / "init_point_cloud.ply"))["vertex"]
    xyz = np.stack([np.asarray(vertices[a], dtype=np.float64) for a in "xyz"], axis=1)
    report["point_count"] = int(xyz.shape[0])
    if xyz.shape[0] != 300000:
        report["errors"].append(f"point count {xyz.shape[0]}")

    max_median = 0.0
    for cam in info.train_cameras[:3]:
        raw = np.asarray(Image.open(cam.depth_path), dtype=np.float64)
        h, w = raw.shape
        fx = fov2focal(cam.FovX, cam.width)
        fy = fov2focal(cam.FovY, cam.height)
        w2c = _w2c_from_cam(cam)
        pc = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1) @ w2c.T
        z = pc[:, 2]
        u = fx * pc[:, 0] / z + cam.cx - 0.5
        v = fy * pc[:, 1] / z + cam.cy - 0.5
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        d = raw[vi[ok], ui[ok]]
        diff = np.abs(d - z[ok])
        valid = np.isfinite(d) & (d > 0)
        if valid.any():
            max_median = max(max_median, float(np.median(diff[valid])))
    report["point_depth_median_abs_err"] = max_median
    if max_median > 0.05:
        report["errors"].append(f"point depth median error {max_median}")


def gpu_smoke(data: Path, repo: Path, report: dict, resolution: int = 1):
    sys.path.insert(0, str(repo))
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams
    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    parser = ArgumentParser()
    model = ModelParams(parser)
    pipe = PipelineParams(parser)
    opt = OptimizationParams(parser)
    args = parser.parse_args([])
    args.source_path = str(data)
    args.model_path = tempfile.mkdtemp(prefix="prendo_smoke_")
    args.resolution = resolution
    args.eval = True
    args.use_hg = False
    args.white_background = False
    args.data_device = "cuda"

    dataset = model.extract(args)
    pipeline = pipe.extract(args)
    optp = opt.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, init_train_args=optp)
    gaussians.training_setup(optp)

    cam = scene.getTrainCameras()[0]
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    override_color = gaussians.compute_lighted_rgb(
        camera_center=cam.camera_center, light=scene.light, iter=1
    )
    pkg = render(cam, gaussians, pipeline, background, override_color=override_color)
    loss = pkg["render"].mean()
    loss.backward()

    peak = torch.cuda.max_memory_allocated() / 1e9
    report["gpu"] = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "resolution": resolution,
        "peak_vram_gb": peak,
    }
    shutil.rmtree(args.model_path, ignore_errors=True)


def self_test():
    f = fov2focal(EXPECTED["camera_angle_x"], EXPECTED["w"])
    assert abs(f - EXPECTED["fl_x"]) < 1e-6, (f, EXPECTED["fl_x"])
    assert abs(focal2fov(f, EXPECTED["w"]) - EXPECTED["camera_angle_x"]) < 1e-12
    m = np.eye(4)
    assert np.allclose(m[3], [0.0, 0.0, 0.0, 1.0])
    assert abs(np.linalg.det(m[:3, :3]) - 1.0) < 1e-12
    print("self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cpu-stubs", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.data is None:
        parser.error("--data is required unless --self-test is used")
    if args.cpu_stubs and args.gpu:
        parser.error("--cpu-stubs cannot be combined with --gpu")
    if args.cpu_stubs:
        install_tinycudann_stub()

    report = {"errors": [], "splits": {}}
    check_metadata(args.data, report)
    check_depth_files(args.data, report)
    info = check_reader(args.data, args.repo, report)
    if info is not None:
        check_roundtrip(info, report)
        check_ply(info, args.data, report)
    if args.gpu:
        gpu_smoke(args.data, args.repo, report, args.resolution)
    report["ok"] = not report["errors"]

    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
