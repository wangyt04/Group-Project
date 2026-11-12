#!/usr/bin/env python3
"""Build a sha256 manifest for a PR-ENDO training run."""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path

AVG_RE = re.compile(
    r"\[ITER (\d+)\] Evaluating test avg: L1 ([\d.]+) PSNR ([\d.]+) "
    r"LPIPS ([\d.]+) SSIM ([\d.]+) MSSIM ([\d.]+)"
)
TEST_INDICES = [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95]
OUTPUT_FILES = [
    "point_cloud/iteration_30000/point_cloud.ply",
    "chkpnt30000.pth",
    "chkpnt_mlp30000.pth",
    "chkpnt_grid_camera30000.pth",
    "chkpnt_light30000.pth",
    "cfg_args",
    "cameras.json",
    "results.txt",
    "train.log",
    "job_info.json",
    "command.txt",
    "nvidia_smi.txt",
    "gpu_info.txt",
    "gpu_mem_samples.txt",
    "environment/pip_freeze.txt",
    "environment/conda_export.txt",
]
DATASET_FILES = [
    "transforms.json",
    "transforms_test.json",
    "prendo_config.json",
    "init_point_cloud.ply",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_metrics(results_path: Path) -> dict:
    if not results_path.exists():
        return {}
    matches = AVG_RE.findall(results_path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        return {}
    iteration, l1, psnr, lpips, ssim, mssim = matches[-1]
    return {
        "iteration": int(iteration),
        "l1": float(l1),
        "psnr": float(psnr),
        "lpips": float(lpips),
        "ssim": float(ssim),
        "mssim": float(mssim),
    }


def _file_entry(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def collect(run_dir: Path, dataset_dir: Path, patch_path: Path, output: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    patch_path = Path(patch_path).resolve()

    job_info = {}
    job_path = run_dir / "job_info.json"
    if job_path.exists():
        job_info = json.loads(job_path.read_text(encoding="utf-8"))

    gpu_name = None
    gpu_mem = None
    gpu_path = run_dir / "gpu_info.txt"
    if gpu_path.exists():
        first = gpu_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) >= 2:
            gpu_name, gpu_mem = parts[0], int(float(parts[1]))

    peak_vram_gb = None
    mem_path = run_dir / "gpu_mem_samples.txt"
    if mem_path.exists():
        samples = [
            int(line.strip())
            for line in mem_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip().isdigit()
        ]
        if samples:
            peak_vram_gb = max(samples) / 1024.0

    command = None
    command_path = run_dir / "command.txt"
    if command_path.exists():
        command = command_path.read_text(encoding="utf-8").strip()

    outputs = []
    for rel in OUTPUT_FILES:
        path = run_dir / rel
        if path.is_file():
            outputs.append(_file_entry(path, run_dir))
    for sub in ("test", "rotate_light_video_maxrot12_cospower10"):
        root = run_dir / sub
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    outputs.append(_file_entry(path, run_dir))

    dataset = {}
    for name in DATASET_FILES:
        path = dataset_dir / name
        dataset[name] = sha256(path) if path.is_file() else None

    manifest = {
        "run_id": run_dir.name,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "repo": "SanoScience/PR-ENDO",
        "base_commit": "ee5362817ccc944292ae43aa9f667386a5ebc069",
        "branch": "vrcaps-reader",
        "reader_patch_sha256": sha256(patch_path) if patch_path.is_file() else None,
        "dataset": {
            "name": dataset_dir.name,
            "train_count": 88,
            "test_count": 12,
            "test_indices": TEST_INDICES,
            "depth_scale": 1.0,
            **dataset,
        },
        "environment": {
            "pip_freeze": "environment/pip_freeze.txt",
            "conda_export": "environment/conda_export.txt",
        },
        "hardware": {
            "gpu": gpu_name,
            "gpu_memory_gb": gpu_mem / 1024.0 if gpu_mem else None,
            "pbs_job_id": job_info.get("pbs_jobid"),
            "pbs_resources": "select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S",
        },
        "command": command,
        "iterations": job_info.get("iterations"),
        "resolution": job_info.get("resolution"),
        "walltime_seconds": job_info.get("walltime_seconds"),
        "peak_vram_gb": peak_vram_gb,
        "metrics": parse_metrics(run_dir / "results.txt"),
        "outputs": outputs,
        "notes": "Mask companion not applied; reader ignores mask keys. Test depth is an internal placeholder.",
    }
    Path(output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = collect(args.run_dir, args.dataset, args.patch, args.output)
    print(f"wrote {args.output} with {len(manifest['outputs'])} outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
