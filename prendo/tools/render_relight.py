#!/usr/bin/env python3
"""Render PR-ENDO relighting predictions for a VRCaps B1 configuration."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import OpenEXR
import lpips

from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim

PRENDO_BASE_COMMIT = "ee5362817ccc944292ae43aa9f667386a5ebc069"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_rotation(direction, device="cuda"):
    d = torch.tensor(direction, dtype=torch.float32, device=device)
    d = d / (d.norm() + 1e-12)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    if abs(float(d[1])) > 0.99:
        up = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    r = torch.cross(up, d, dim=0)
    r = r / (r.norm() + 1e-12)
    u = torch.cross(d, r, dim=0)
    R = torch.stack([r, u, d], dim=1)
    if float(torch.det(R)) < 0:
        R[:, 0] = -R[:, 0]
    return R


def write_png(path: Path, image_chw: torch.Tensor):
    arr = (image_chw.permute(1, 2, 0).detach().cpu().numpy() * 255.0).round()
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def write_exr(path: Path, linear_chw: torch.Tensor):
    arr = linear_chw.permute(1, 2, 0).detach().cpu().numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
    arr = np.ascontiguousarray(arr)
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    channels = {
        "R": OpenEXR.Channel("R", np.ascontiguousarray(arr[..., 0])),
        "G": OpenEXR.Channel("G", np.ascontiguousarray(arr[..., 1])),
        "B": OpenEXR.Channel("B", np.ascontiguousarray(arr[..., 2])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with OpenEXR.File(header, channels) as f:
        f.write(str(path))


def compute_train_metrics(gaussians, scene, pipe, background, iteration, lpips_model):
    train_cams = scene.getTrainCameras()
    psnrs, ssims, lpipss = [], [], []
    with torch.no_grad():
        for cam in tqdm(train_cams, desc="train metrics"):
            override = gaussians.compute_lighted_rgb(
                camera_center=cam.camera_center, light=scene.light, iter=iteration
            )
            img = render(cam, gaussians, pipe, background, override_color=override)["render"].clamp(0.0, 1.0)
            gt = cam.original_image.to("cuda")[:3].clamp(0.0, 1.0)
            psnrs.append(float(psnr(img, gt).mean()))
            ssims.append(float(ssim(img, gt).mean()))
            lpipss.append(float(lpips_model(img, gt).mean()))
    return {
        "num_views": len(train_cams),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--relight-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--spot-power-n", type=float, default=None)
    parser.add_argument("--levels", type=str, default=None, help="comma-separated levels, e.g. L2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-train-metrics", action="store_true")
    parser.add_argument("--code-commit", default="unknown")
    parser.add_argument("--prendo-commit", default=PRENDO_BASE_COMMIT)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    args = parser.parse_args()

    data = args.data.resolve()
    model = args.model.resolve()
    relight_dir = (args.relight_dir or (data / "relight")).resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lights = load_json(relight_dir / "lights.json")
    relight_meta = load_json(relight_dir / "transforms_relight.json")
    relight_frames = {str(f["file_path"]).zfill(4): f for f in relight_meta["frames"]}
    configs = {c["id"]: c for c in lights["configs"]}
    n = args.spot_power_n
    if n is None:
        n = lights.get("spot_power_n")
    levels = set(args.levels.split(",")) if args.levels else None

    dataset = SimpleNamespace(
        sh_degree=0,
        source_path=str(data),
        model_path=str(model),
        images="images",
        depths="depths",
        resolution=1,
        white_background=False,
        data_device="cuda",
        eval=True,
        use_hg=False,
    )
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.mlp.eval()
    gaussians.positional_encoding_camera.eval()
    iteration = scene.loaded_iter

    cams = {c.image_name: c for c in scene.getTrainCameras()}
    if not cams:
        raise RuntimeError("no training cameras loaded")

    # Training-frame metrics
    train_metrics_path = out_dir / "train_metrics.json"
    if args.skip_train_metrics and train_metrics_path.exists():
        train_metrics = load_json(train_metrics_path)
        print("loaded train metrics:", train_metrics)
    elif args.skip_train_metrics:
        train_metrics = {"num_views": 0, "psnr": None, "ssim": None, "lpips": None}
        print("skipping train metrics")
    else:
        lpips_model = lpips.LPIPS(net="vgg").cuda().eval()
        train_metrics = compute_train_metrics(
            gaussians, scene, pipe, background, iteration, lpips_model
        )
        train_metrics_path.write_text(json.dumps(train_metrics, indent=2) + "\n", encoding="utf-8")
        print("train metrics:", train_metrics)

    entries = [e for e in lights["frames"] if e.get("valid") is True]
    invalid_count = len(lights["frames"]) - len(entries)
    filtered = []
    skipped_l2_no_n = 0
    for e in entries:
        cfg = configs.get(e["config"])
        if cfg is None:
            raise RuntimeError(f"unknown config {e['config']}")
        if levels is not None and cfg["level"] not in levels:
            continue
        if cfg["level"] == "L2" and n is None:
            skipped_l2_no_n += 1
            continue
        filtered.append(e)
    if args.limit is not None:
        filtered = filtered[: args.limit]

    rendered = 0
    rendered_configs = set()
    for e in tqdm(filtered, desc="relight"):
        cfg = configs[e["config"]]
        frame = str(e["frame"]).zfill(4)
        if frame not in relight_frames:
            raise RuntimeError(f"relight frame {frame} missing from transforms_relight.json")
        cam = cams.get(frame)
        if cam is None:
            raise RuntimeError(f"relight frame {frame} is not a training camera")
        light = scene.light.clone().detach()
        light[0] = light[0] * float(cfg["intensity_scale"])
        light_center = torch.tensor(e["light_pos_world"], dtype=torch.float32, device="cuda")
        light_rot = None
        spot_power = 0.0
        if cfg["level"] == "L2":
            light_rot = build_rotation(e["light_dir_world"], device="cuda")
            spot_power = float(n)
        override = gaussians.compute_lighted_rgb(
            camera_center=cam.camera_center,
            light=light,
            iter=iteration,
            light_center_sep=light_center,
            light_rot=light_rot,
            spotlight_attenuation_power=spot_power,
        )
        color = torch.tensor(cfg["color_scale"], dtype=torch.float32, device="cuda")
        if not torch.allclose(color, torch.ones_like(color)):
            override = ((override.clamp(min=1e-6) ** 2.2) * color).clamp(0.0, 1.0) ** (1.0 / 2.2)
        with torch.no_grad():
            img = render(cam, gaussians, pipe, background, override_color=override)["render"].clamp(0.0, 1.0)
        config_dir = out_dir / cfg["id"]
        write_png(config_dir / f"{frame}.png", img)
        write_exr(config_dir / f"{frame}.exr", img.clamp(min=0.0) ** 2.2)
        rendered += 1
        rendered_configs.add(cfg["id"])

    total_png = len(list(out_dir.glob("*/*.png")))
    command_path = model / "command.txt"
    train_command = command_path.read_text(encoding="utf-8").strip() if command_path.exists() else ""

    notes = [
        "B1 configuration: VRCaps_lumen_100_v2_relight_config",
        f"valid frame entries in B1: {len(entries)}; invalid skipped: {invalid_count}",
        f"rendered this run: {rendered}; total PNG pairs in output: {total_png}",
        "shadow/noshadow variants merged: one image per (config, frame)",
        "mask companion not applied; masks are not used by the reader",
        "test depth is an internal placeholder, not test ground truth",
        "physical/numerical scale unverified; mm_per_unit is null",
    ]
    if skipped_l2_no_n:
        notes.append(f"L2 skipped: spot_power_n is null (entries skipped: {skipped_l2_no_n})")
    if n is not None:
        notes.append(f"L2 rendered with spot_power_n={n}")

    record = {
        "dataset": data.name,
        "prendo_commit": args.prendo_commit,
        "code_commit": args.code_commit,
        "train_command": train_command,
        "iteration": iteration,
        "depth_scale": args.depth_scale,
        "spot_power": n,
        "train_psnr": train_metrics.get("psnr"),
        "train_ssim": train_metrics.get("ssim"),
        "train_lpips": train_metrics.get("lpips"),
        "notes": " | ".join(notes),
    }
    (out_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_id": args.run_id,
        "dataset": data.name,
        "model": str(model),
        "iteration": iteration,
        "relight_dir": str(relight_dir),
        "output": str(out_dir),
        "levels": sorted(levels) if levels is not None else None,
        "spot_power_n": n,
        "valid_entries_total": len(entries),
        "invalid_skipped": invalid_count,
        "skipped_l2_no_n": skipped_l2_no_n,
        "rendered_this_run": rendered,
        "total_png_pairs_in_output": total_png,
        "rendered_configs": sorted(rendered_configs),
        "train_metrics": train_metrics,
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
