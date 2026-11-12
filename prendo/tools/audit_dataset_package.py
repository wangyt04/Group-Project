#!/usr/bin/env python3
"""Audit the VRCaps_lumen_100 package without image libraries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TEST_INDICES = [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95]
TRAIN_INDICES = [i for i in range(100) if i not in TEST_INDICES]
EXPECTED_W, EXPECTED_H = 1350, 1080
EXPECTED_FX = 664.6633833646774
EXPECTED_FY = 664.6633672714233
EXPECTED_CX, EXPECTED_CY = 675.0, 540.0
EXPECTED_FOVX = 1.586227683125316
EXPECTED_POINTS = 300_000


def _err(errors, msg):
    errors.append(msg)


def _load_json(path, errors):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        _err(errors, f"cannot read {path}: {exc}")
        return None


def _check_transforms(path, expected_indices, label, errors):
    data = _load_json(path, errors)
    if data is None:
        return
    for key in ("camera_angle_x", "fl_x", "fl_y", "cx", "cy", "w", "h", "frames"):
        if key not in data:
            _err(errors, f"{label}: missing key {key}")
    if data.get("w") != EXPECTED_W or data.get("h") != EXPECTED_H:
        _err(errors, f"{label}: expected {EXPECTED_W}x{EXPECTED_H}")
    for key, expected in (
        ("fl_x", EXPECTED_FX),
        ("fl_y", EXPECTED_FY),
        ("cx", EXPECTED_CX),
        ("cy", EXPECTED_CY),
        ("camera_angle_x", EXPECTED_FOVX),
    ):
        if key in data and abs(float(data[key]) - expected) > 1e-6 * max(1.0, abs(expected)):
            _err(errors, f"{label}: {key}={data[key]} expected {expected}")
    frames = data.get("frames", [])
    if len(frames) != len(expected_indices):
        _err(errors, f"{label}: expected {len(expected_indices)} frames, got {len(frames)}")
    got = sorted(int(str(f.get("file_path"))) for f in frames)
    if got != sorted(expected_indices):
        _err(errors, f"{label}: frame indices mismatch")
    for f in frames:
        m = f.get("transform_matrix")
        if not (isinstance(m, list) and len(m) == 4 and all(len(r) == 4 for r in m)):
            _err(errors, f"{label}: bad transform_matrix for {f.get('file_path')}")
            continue
        if [m[3][0], m[3][1], m[3][2], m[3][3]] != [0.0, 0.0, 0.0, 1.0]:
            _err(errors, f"{label}: bad homogeneous row for {f.get('file_path')}")


def _check_files(root, errors):
    train_dir = root / "train_views"
    depth_dir = root / "depth_train"
    test_dir = root / "test_views_1"
    for d in (train_dir, depth_dir, test_dir):
        if not d.is_dir():
            _err(errors, f"missing directory {d}")
            return
    train_pngs = sorted(p.name for p in train_dir.glob("*.png"))
    depth_tiffs = sorted(p.name for p in depth_dir.glob("*.tiff"))
    test_pngs = sorted(p.name for p in test_dir.glob("*.png"))
    if train_pngs != [f"{i:04d}.png" for i in TRAIN_INDICES]:
        _err(errors, f"train_views mismatch: {len(train_pngs)} files")
    if depth_tiffs != [f"{i:04d}.tiff" for i in TRAIN_INDICES]:
        _err(errors, f"depth_train mismatch: {len(depth_tiffs)} files")
    if test_pngs != [f"{i:04d}.png" for i in TEST_INDICES]:
        _err(errors, f"test_views_1 mismatch: {len(test_pngs)} files")


def _check_config(root, errors):
    cfg = _load_json(root / "prendo_config.json", errors)
    if cfg is None:
        return
    if float(cfg.get("depth_scale", -1)) != 1.0:
        _err(errors, f"prendo_config.json depth_scale must be 1.0, got {cfg.get('depth_scale')}")
    if cfg.get("test_dir") != "test_views_1":
        _err(errors, "prendo_config.json test_dir must be test_views_1")


def _check_ply(path, errors):
    try:
        with open(path, "rb") as f:
            header = b""
            while b"end_header" not in header:
                line = f.readline()
                if not line:
                    _err(errors, f"{path}: missing end_header")
                    return
                header += line
    except Exception as exc:
        _err(errors, f"{path}: {exc}")
        return
    text = header.decode("ascii", errors="replace")
    if "format binary_little_endian 1.0" not in text:
        _err(errors, f"{path}: expected binary_little_endian 1.0")
    count = None
    props = []
    for line in text.splitlines():
        if line.startswith("element vertex "):
            count = int(line.split()[2])
        elif line.startswith("property float "):
            props.append(line.split()[-1])
    if count != EXPECTED_POINTS:
        _err(errors, f"{path}: expected {EXPECTED_POINTS} vertices, got {count}")
    for axis in ("x", "y", "z"):
        if axis not in props:
            _err(errors, f"{path}: missing property {axis}")


def audit(root: Path) -> dict:
    errors = []
    root = Path(root).resolve()
    if "C3VD" in str(root):
        _err(errors, f"path contains forbidden substring C3VD: {root}")
    for name in ("transforms.json", "transforms_test.json", "prendo_config.json", "init_point_cloud.ply"):
        if not (root / name).is_file():
            _err(errors, f"missing file {name}")
    _check_files(root, errors)
    _check_transforms(root / "transforms.json", TRAIN_INDICES, "transforms.json", errors)
    _check_transforms(root / "transforms_test.json", TEST_INDICES, "transforms_test.json", errors)
    _check_config(root, errors)
    _check_ply(root / "init_point_cloud.ply", errors)
    return {
        "ok": not errors,
        "root": str(root),
        "train_count": len(TRAIN_INDICES),
        "test_count": len(TEST_INDICES),
        "test_indices": TEST_INDICES,
        "depth_scale": 1.0,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.data)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
