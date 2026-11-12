# VRCaps + PR-ENDO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate `VRCaps_lumen_100` against the patched PR-ENDO reader, build a reproducible GPU environment on Imperial CX3, run one official RotateColon 30k training, and produce a complete artifact bundle for later relighting evaluation.

**Architecture:** A minimal two-file reader patch is applied to a clean `ee53628` checkout on branch `vrcaps-reader`; the dataset stays in its native RotateColon layout with `depth_scale=1.0`. A GPU PBS job builds the conda environment and CUDA extensions, a validation script checks the reader/depth/projection/VRAM before training, and a second PBS job runs the official 30k command. Post-training PBS jobs render test views and light rotations, then a stdlib collector writes a sha256 manifest and tarball.

**Tech Stack:** Python 3.10, PyTorch 2.1.0+cu121, CUDA 12.1.1, tiny-cuda-nn, depth-diff-gaussian-rasterization, simple-knn, gaussian-norms, FAISS-GPU, PBSPro on Imperial CX3 (L40S).

## Global Constraints

- Work only on branch `vrcaps-reader`; `main` must stay at upstream `ee53628`.
- The only source change is the two-file official reader patch. Do not modify `train_viewdir.py`, losses, cameras, or rendering code.
- Do not apply the optional mask companion. Do not enable mask supervision.
- Use `--resolution 1` unless the GPU smoke reports peak VRAM > 40 GB; then use `--resolution 2` and record the change.
- Training recipe: 30000 iterations, `K_normals=30`, `diffuse_loss_weight=0.01`, `lambda_depth=0.2`, `lambda_norm=0.1`, `albedo_loss_weight=1.0`, densification/opacity/MLP settings exactly as in the RotateColon block of `run_all.sh`.
- Data stays at `depth_scale=1.0`; do not rescale poses, depth, or point cloud.
- No path component may contain the substring `C3VD`.
- HPC user `yl15425`, login host `login.cx3.hpc.imperial.ac.uk`, SSH multiplex socket `~/.ssh/hpc.sock`.
- `$EPHEMERAL` is not exported on login nodes; use `/rds/general/user/yl15425/ephemeral` explicitly.
- Home: `/rds/general/user/yl15425/home`. Code/env/artifacts under `$HOME/prendo`; data/runs under `/rds/general/user/yl15425/ephemeral/prendo`.
- GPU queue is `v1_gpu72`; request `ngpus=1:gpu_type=L40S`.
- Commit after every task. Do not commit the 678 MB `FOR_TEAM_VRCaps.zip`.

---

## File Structure

Local repo additions:

```text
patches/vrcaps_reader.patch                 # official reader patch, byte-identical to the zip
tools/audit_dataset_package.py              # stdlib-only package layout audit
tools/validate_vrcaps_interface.py          # HPC reader/depth/projection/GPU validation
tools/collect_artifacts.py                  # stdlib-only manifest + sha256 collector
tests/test_audit_dataset_package.py         # unittest for the audit tool
tests/test_collect_artifacts.py             # unittest for the collector
hpc/paths.sh                                # HPC paths, modules, conda hook
hpc/env.sh                                  # activate prendo env and assert torch
hpc/sync_to_hpc.sh                          # local rsync of repo + dataset
hpc/env_build.pbs                           # GPU job: create env, build extensions
hpc/smoke.pbs                               # GPU job: validate reader + VRAM
hpc/train.env                               # RESOLUTION/ITERATIONS/K_NORMALS/etc.
hpc/train.pbs                               # GPU job: 30k training
hpc/render.pbs                              # GPU job: render + light rotations + metrics
hpc/package_bundle.sh                       # HPC: tar bundle and copy to $HOME
docs/runbooks/vrcaps_training.md            # operator runbook
```

`main` stays clean; all additions are on `vrcaps-reader`.

---

### Task 1: Branch, official reader patch, submodules

**Files:**
- Create: `patches/vrcaps_reader.patch`
- No source file changes committed.

**Interfaces:**
- Consumes: `FOR_TEAM_VRCaps.zip` at repo root.
- Produces: `patches/vrcaps_reader.patch` with sha256 `96d19a8d314257cf9ed98b45bea77eddf2cb29aa0cd6741ad62020d6f7e41895`; branch `vrcaps-reader`; populated submodule directories.

- [ ] **Step 1: Confirm branch and clean state**

```bash
cd /home/yiping/Documents/my_repo/PR-ENDO
git branch --show-current
git status --short
```

Expected: `vrcaps-reader`; only `?? FOR_TEAM_VRCaps.zip` is untracked.

- [ ] **Step 2: Extract the official reader patch**

```bash
mkdir -p patches
unzip -p FOR_TEAM_VRCaps.zip \
  FOR_TEAM_VRCaps/optional_mask_support/receiving_team_original.patch \
  > patches/vrcaps_reader.patch
sha256sum patches/vrcaps_reader.patch
```

Expected sha256: `96d19a8d314257cf9ed98b45bea77eddf2cb29aa0cd6741ad62020d6f7e41895`.

- [ ] **Step 3: Verify the patch applies to the pinned base**

```bash
git apply -p0 --check patches/vrcaps_reader.patch
grep -n "readVRCapsSceneInfo" patches/vrcaps_reader.patch
grep -n '"vrcaps"' patches/vrcaps_reader.patch
```

Expected: `git apply --check` exits 0; both grep commands print matches.

- [ ] **Step 4: Initialize submodules locally**

```bash
git submodule update --init --recursive
ls submodules/depth-diff-gaussian-rasterization | head
ls submodules/simple-knn | head
ls submodules/gaussian-norms | head
```

Expected: each directory contains source files, not empty. If `git submodule update` fails (network or GitLab access), stop and report the error; do not sync empty submodule directories to HPC.

- [ ] **Step 5: Commit the patch only**

```bash
git add patches/vrcaps_reader.patch
git commit -m "chore: add official VRCaps reader patch"
git status --short
```

Expected: commit created; only `?? FOR_TEAM_VRCaps.zip` remains.

---

### Task 2: Stdlib dataset package audit

**Files:**
- Create: `tools/audit_dataset_package.py`
- Create: `tests/test_audit_dataset_package.py`

**Interfaces:**
- Consumes: extracted `VRCaps_lumen_100` directory.
- Produces: `audit(root: Path) -> dict` with keys `ok`, `root`, `train_count`, `test_count`, `test_indices`, `depth_scale`, `errors`; CLI exit 0/1.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit_dataset_package.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_dataset_package import TEST_INDICES, TRAIN_INDICES, audit

EXPECTED = {
    "camera_angle_x": 1.586227683125316,
    "fl_x": 664.6633833646774,
    "fl_y": 664.6633672714233,
    "cx": 675.0,
    "cy": 540.0,
    "w": 1350,
    "h": 1080,
}


def write_transforms(path, indices):
    frames = [
        {
            "file_path": str(i),
            "transform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        for i in indices
    ]
    path.write_text(json.dumps({**EXPECTED, "frames": frames}), encoding="utf-8")


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "VRCaps_lumen_100"
        (self.root / "train_views").mkdir(parents=True)
        (self.root / "depth_train").mkdir()
        (self.root / "test_views_1").mkdir()
        for i in TRAIN_INDICES:
            (self.root / "train_views" / f"{i:04d}.png").write_bytes(b"")
            (self.root / "depth_train" / f"{i:04d}.tiff").write_bytes(b"")
        for i in TEST_INDICES:
            (self.root / "test_views_1" / f"{i:04d}.png").write_bytes(b"")
        write_transforms(self.root / "transforms.json", TRAIN_INDICES)
        write_transforms(self.root / "transforms_test.json", TEST_INDICES)
        (self.root / "prendo_config.json").write_text(
            json.dumps({"depth_scale": 1.0, "test_dir": "test_views_1"}), encoding="utf-8"
        )
        (self.root / "init_point_cloud.ply").write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 300000\n"
            b"property float x\nproperty float y\nproperty float z\nend_header\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_package_passes(self):
        report = audit(self.root)
        self.assertTrue(report["ok"], report["errors"])

    def test_missing_depth_fails(self):
        (self.root / "depth_train" / "0000.tiff").unlink()
        report = audit(self.root)
        self.assertFalse(report["ok"])
        self.assertTrue(any("depth_train" in e for e in report["errors"]))

    def test_wrong_depth_scale_fails(self):
        (self.root / "prendo_config.json").write_text(
            json.dumps({"depth_scale": 1.4, "test_dir": "test_views_1"}), encoding="utf-8"
        )
        report = audit(self.root)
        self.assertFalse(report["ok"])
        self.assertTrue(any("depth_scale" in e for e in report["errors"]))

    def test_c3vd_path_fails(self):
        c3vd_root = Path(self.tmp.name) / "C3VD_pkg"
        self.root.rename(c3vd_root)
        report = audit(c3vd_root)
        self.assertFalse(report["ok"])
        self.assertTrue(any("C3VD" in e for e in report["errors"]))

    def test_wrong_test_index_fails(self):
        (self.root / "test_views_1" / "0095.png").rename(
            self.root / "test_views_1" / "0096.png"
        )
        report = audit(self.root)
        self.assertFalse(report["ok"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_audit_dataset_package -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.audit_dataset_package'`.

- [ ] **Step 3: Implement the audit tool**

Create `tools/audit_dataset_package.py`:

```python
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
        _err(errors, f"prendo_config.json test_dir must be test_views_1")


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
```

Create `tools/__init__.py` and `tests/__init__.py` (empty) so `python3 -m unittest` can import the packages:

```bash
mkdir -p tools tests
touch tools/__init__.py tests/__init__.py
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
python3 -m unittest tests.test_audit_dataset_package -v
```

Expected: 5 tests, all PASS.

- [ ] **Step 5: Extract and audit the real dataset**

```bash
mkdir -p /home/yiping/Documents/my_repo/PR-ENDO-data
rm -rf /tmp/vrcaps_extract
mkdir -p /tmp/vrcaps_extract
unzip -q FOR_TEAM_VRCaps.zip 'FOR_TEAM_VRCaps/VRCaps_lumen_100/*' -d /tmp/vrcaps_extract
rsync -a /tmp/vrcaps_extract/FOR_TEAM_VRCaps/VRCaps_lumen_100/ \
  /home/yiping/Documents/my_repo/PR-ENDO-data/VRCaps_lumen_100/
python3 tools/audit_dataset_package.py \
  /home/yiping/Documents/my_repo/PR-ENDO-data/VRCaps_lumen_100
```

Expected: JSON with `"ok": true`, `"train_count": 88`, `"test_count": 12`, `"errors": []`.

- [ ] **Step 6: Commit**

```bash
git add tools/__init__.py tests/__init__.py tools/audit_dataset_package.py tests/test_audit_dataset_package.py
git commit -m "tools: add VRCaps dataset package audit"
```

---

### Task 3: HPC paths, environment activation, and sync

**Files:**
- Create: `hpc/paths.sh`
- Create: `hpc/env.sh`
- Create: `hpc/sync_to_hpc.sh`

**Interfaces:**
- Consumes: verified SSH socket `~/.ssh/hpc.sock`; local repo; staged dataset.
- Produces: exported `PRENDO_HOME`, `PRENDO_CODE`, `PRENDO_DATA`, `PRENDO_RUNS`, `PRENDO_ARTIFACTS`, `PRENDO_LOGS`, `EPHEMERAL`; remote directories populated.

- [ ] **Step 1: Create `hpc/paths.sh`**

```bash
#!/bin/bash
set -euo pipefail

export PRENDO_HOME="${HOME}/prendo"
export PRENDO_CODE="${PRENDO_HOME}/code/PR-ENDO"
export EPHEMERAL=/rds/general/user/yl15425/ephemeral
export PRENDO_DATA="${EPHEMERAL}/prendo/data/VRCaps_lumen_100"
export PRENDO_RUNS="${EPHEMERAL}/prendo/runs"
export PRENDO_ARTIFACTS="${PRENDO_HOME}/artifacts"
export PRENDO_LOGS="${PRENDO_HOME}/logs"

if ! command -v module >/dev/null 2>&1; then
  set +eu
  source /etc/profile.d/modules.sh
  set -eu
fi
module use /sw-eb/modules/all
module load miniforge/3
if [ ! -x "${HOME}/miniforge3/bin/conda" ]; then
  miniforge-setup
fi
eval "$("${HOME}/miniforge3/bin/conda" shell.bash hook)"
module load CUDA/12.1.1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

mkdir -p "${PRENDO_HOME}" "${PRENDO_LOGS}" "${PRENDO_ARTIFACTS}" "${PRENDO_RUNS}"
```

- [ ] **Step 2: Create `hpc/env.sh`**

```bash
#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
conda activate prendo
python - <<'PY'
import torch
assert torch.__version__.startswith("2.1.0"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
if torch.cuda.is_available():
    print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
else:
    print("torch", torch.__version__, "cuda", torch.version.cuda, "no CUDA device visible")
PY
```

- [ ] **Step 3: Create `hpc/sync_to_hpc.sh`**

```bash
#!/bin/bash
set -euo pipefail

HPC_USER=yl15425
HPC_HOST=login.cx3.hpc.imperial.ac.uk
SOCK="${HOME}/.ssh/hpc.sock"
REPO="${HOME}/Documents/my_repo/PR-ENDO"
DATA_STAGE="${HOME}/Documents/my_repo/PR-ENDO-data/VRCaps_lumen_100"
EPHEMERAL=/rds/general/user/yl15425/ephemeral

remote() {
  ssh -S "${SOCK}" "${HPC_USER}@${HPC_HOST}" "$@"
}

remote "mkdir -p \$HOME/prendo/code \$HOME/prendo/logs \$HOME/prendo/artifacts ${EPHEMERAL}/prendo/data ${EPHEMERAL}/prendo/runs"

rsync -az -e "ssh -S ${SOCK}" \
  --exclude 'FOR_TEAM_VRCaps.zip' \
  --exclude '*.pyc' \
  --exclude '__pycache__' \
  "${REPO}/" "${HPC_USER}@${HPC_HOST}:\$HOME/prendo/code/PR-ENDO/"

rsync -az -e "ssh -S ${SOCK}" \
  "${DATA_STAGE}/" "${HPC_USER}@${HPC_HOST}:${EPHEMERAL}/prendo/data/VRCaps_lumen_100/"

remote "set -e; \
  test \$(ls ${EPHEMERAL}/prendo/data/VRCaps_lumen_100/train_views/*.png | wc -l) -eq 88; \
  test \$(ls ${EPHEMERAL}/prendo/data/VRCaps_lumen_100/depth_train/*.tiff | wc -l) -eq 88; \
  test \$(ls ${EPHEMERAL}/prendo/data/VRCaps_lumen_100/test_views_1/*.png | wc -l) -eq 12; \
  echo data-ok"

remote "set -e; cd \$HOME/prendo/code/PR-ENDO; \
  if ! grep -q readVRCapsSceneInfo scene/dataset_readers.py; then \
    git apply -p0 --check patches/vrcaps_reader.patch; \
    git apply -p0 patches/vrcaps_reader.patch; \
  fi; \
  grep -q readVRCapsSceneInfo scene/dataset_readers.py; \
  grep -q '\"vrcaps\"' scene/__init__.py; \
  echo patch-ok"
```

- [ ] **Step 4: Syntax-check the scripts**

```bash
bash -n hpc/paths.sh
bash -n hpc/env.sh
bash -n hpc/sync_to_hpc.sh
```

Expected: no output, exit 0.

- [ ] **Step 5: Sync repo and data to HPC**

```bash
chmod +x hpc/sync_to_hpc.sh
./hpc/sync_to_hpc.sh
```

Expected: final lines `data-ok` and `patch-ok`.

- [ ] **Step 6: Verify remote layout**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'ls $HOME/prendo/code/PR-ENDO | head; ls $HOME/prendo/code/PR-ENDO/patches; ls /rds/general/user/yl15425/ephemeral/prendo/data/VRCaps_lumen_100 | head'
```

Expected: repo files, `vrcaps_reader.patch`, dataset files.

- [ ] **Step 7: Commit**

```bash
git add hpc/paths.sh hpc/env.sh hpc/sync_to_hpc.sh
git commit -m "hpc: add paths, environment activation, and sync scripts"
```

---

### Task 4: GPU environment build

**Files:**
- Create: `hpc/env_build.pbs`

**Interfaces:**
- Consumes: synced repo and submodules; `hpc/paths.sh`.
- Produces: conda env `prendo`; importable `torch`, `tinycudann`, `diff_gaussian_rasterization`, `simple_knn`, `faiss`; `$PRENDO_LOGS/env_build.log`, `pip_freeze.txt`, `conda_export.txt`.

- [ ] **Step 1: Create `hpc/env_build.pbs`**

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=02:00:00
#PBS -N prendo_envbuild
#PBS -j oe
set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME}"
source "$HOME/prendo/code/PR-ENDO/hpc/paths.sh"

LOG="$PRENDO_LOGS/env_build.log"
exec > >(tee -a "$LOG") 2>&1
echo "start $(date -u +%FT%TZ)"
echo "host $(hostname)"

if ! conda env list | awk '{print $1}' | grep -qx prendo; then
  conda create -y -n prendo python=3.10
fi
conda activate prendo

python -m pip install --upgrade pip
python -m pip install "setuptools==75.8.0" wheel
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2" tqdm lpips plyfile matplotlib natsort pytorch_ssim \
  pytorch_msssim "imageio[ffmpeg]" ninja cmake tensorboard
python -m pip install openexr || conda install -y -c conda-forge openexr

cd "$PRENDO_CODE"
python -m pip install --no-build-isolation \
  submodules/depth-diff-gaussian-rasterization/ \
  submodules/simple-knn/ \
  submodules/gaussian-norms/
python -m pip install \
  https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python -m pip install --no-build-isolation \
  git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
print("mem_gb", torch.cuda.get_device_properties(0).total_memory / 1e9)
import tinycudann, diff_gaussian_rasterization, simple_knn, faiss
print("extensions ok")
PY

python -m pip freeze > "$PRENDO_LOGS/pip_freeze.txt"
conda list --export > "$PRENDO_LOGS/conda_export.txt"
echo "done $(date -u +%FT%TZ)"
```

- [ ] **Step 2: Syntax-check and submit**

```bash
bash -n hpc/env_build.pbs
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'cd $HOME/prendo/code/PR-ENDO && qsub hpc/env_build.pbs'
```

Expected: a numeric PBS job id.

- [ ] **Step 3: Wait for the job and inspect the log**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'qstat -u yl15425; tail -n 80 $HOME/prendo/logs/env_build.log'
```

Repeat until `qstat` shows no running job. Expected log lines:

```text
torch 2.1.0
cuda 12.1
device NVIDIA L40S
extensions ok
done ...
```

If `tinycudann` or a submodule build fails, read the tail, fix the environment issue, and resubmit. Do not proceed to Task 5 until `extensions ok` is present.

- [ ] **Step 4: Commit**

```bash
git add hpc/env_build.pbs
git commit -m "hpc: add GPU environment build job"
```

---

### Task 5: Interface validation script

**Files:**
- Create: `tools/validate_vrcaps_interface.py`

**Interfaces:**
- Consumes: patched reader in the HPC checkout; dataset; `prendo` env.
- Produces: JSON report with `errors`, `splits`, `depth_stats`, `reader`, `depth_target_max_rel_err`, `intrinsics`, `roundtrip_max_err_px`, `point_count`, `point_depth_median_abs_err`, and optional `gpu.device`/`gpu.peak_vram_gb`; exit 0/1.

- [ ] **Step 1: Create the script**

Create `tools/validate_vrcaps_interface.py`:

```python
#!/usr/bin/env python3
"""Validate VRCaps_lumen_100 against the patched PR-ENDO reader."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
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


def gpu_smoke(data: Path, repo: Path, report: dict):
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
    args.resolution = 1
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
        "peak_vram_gb": peak,
    }
    if peak > 40.0:
        report["errors"].append(f"peak VRAM {peak:.2f} GB > 40 GB")
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
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.data is None:
        parser.error("--data is required unless --self-test is used")

    report = {"errors": [], "splits": {}}
    check_metadata(args.data, report)
    check_depth_files(args.data, report)
    info = check_reader(args.data, args.repo, report)
    if info is not None:
        check_roundtrip(info, report)
        check_ply(info, args.data, report)
    if args.gpu:
        gpu_smoke(args.data, args.repo, report)
    report["ok"] = not report["errors"]

    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the self-test on HPC**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'cd $HOME/prendo/code/PR-ENDO && source hpc/env.sh && python tools/validate_vrcaps_interface.py --self-test'
```

Expected: `self-test ok`.

- [ ] **Step 3: Run the CPU/reader validation**

A preliminary login-node run can stub `tinycudann` (the reader never uses it):

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'cd $HOME/prendo/code/PR-ENDO && source hpc/env.sh && \
   python tools/validate_vrcaps_interface.py \
     --data /rds/general/user/yl15425/ephemeral/prendo/data/VRCaps_lumen_100 \
     --repo $HOME/prendo/code/PR-ENDO \
     --cpu-stubs \
     --report $HOME/prendo/logs/interface_cpu_login.json'
```

The authoritative run without stubs is the first command in `hpc/smoke.pbs`.

Expected report: JSON with `"ok": true`; `reader.train == 88`, `reader.test == 12`, `roundtrip_max_err_px < 0.01`, `point_depth_median_abs_err < 0.05`, `depth_target_max_rel_err < 1e-5`.

- [ ] **Step 4: Commit**

```bash
git add tools/validate_vrcaps_interface.py
git commit -m "tools: add VRCaps reader and GPU validation"
```

---

### Task 6: GPU smoke job and resolution decision

**Files:**
- Create: `hpc/smoke.pbs`

**Interfaces:**
- Consumes: `prendo` env; dataset; validation script.
- Produces: `$HOME/prendo/logs/smoke_report.json` with `gpu.peak_vram_gb`.

- [ ] **Step 1: Create `hpc/smoke.pbs`**

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=00:40:00
#PBS -N prendo_smoke
#PBS -j oe
set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME}"
source "$HOME/prendo/code/PR-ENDO/hpc/env.sh"
cd "$PRENDO_CODE"

nvidia-smi | tee "$PRENDO_LOGS/smoke_nvidia_smi.txt"
python tools/validate_vrcaps_interface.py \
  --data "$PRENDO_DATA" \
  --repo "$PRENDO_CODE" \
  --report "$PRENDO_LOGS/interface_cpu.json"
python tools/validate_vrcaps_interface.py \
  --data "$PRENDO_DATA" \
  --repo "$PRENDO_CODE" \
  --gpu \
  --report "$PRENDO_LOGS/smoke_report.json"
```

- [ ] **Step 2: Submit the smoke job**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'cd $HOME/prendo/code/PR-ENDO && qsub hpc/smoke.pbs'
```

- [ ] **Step 3: Wait and read the report**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'qstat -u yl15425; cat $HOME/prendo/logs/smoke_report.json'
```

Expected: `"ok": true`, `gpu.peak_vram_gb` reported, and `gpu.device` contains `L40S`.

- [ ] **Step 4: Decide resolution**

If `gpu.peak_vram_gb <= 40`, create `hpc/train.env` with `RESOLUTION=1`.

If `gpu.peak_vram_gb > 40`, create `hpc/train.env` with `RESOLUTION=2` and add a line to `docs/runbooks/vrcaps_training.md` recording the measured peak and the reason.

Create `hpc/train.env` with these exact contents (edit `RESOLUTION` only if needed):

```bash
export RUN_NAME=vrcaps_lumen_100_r30k
export RESOLUTION=1
export ITERATIONS=30000
export K_NORMALS=30
export DIFFUSE_LOSS_WEIGHT=0.01
```

- [ ] **Step 5: Commit**

```bash
git add hpc/smoke.pbs hpc/train.env
git commit -m "hpc: add GPU smoke job and training config"
```

---

### Task 7: 30k training job

**Note:** During execution the GPU queue was long, so `hpc/train.pbs` was extended to run the reader validation and GPU smoke first, choose `RESOLUTION` from the measured peak VRAM, train in the same 24h PBS job, and then run `render.py`, `render_light_rotations.py`, and `metrics.py`. This avoids a second GPU queue wait. `hpc/smoke.pbs` and `hpc/render.pbs` remain available as standalone checks.

**Files:**
- Create: `hpc/train.pbs`

**Interfaces:**
- Consumes: `hpc/env.sh`, `hpc/train.env`, dataset, patched reader.
- Produces: `$PRENDO_RUNS/$RUN_NAME/` with `point_cloud/iteration_30000/point_cloud.ply`, `chkpnt*30000.pth`, `cfg_args`, `cameras.json`, `results.txt`, `train.log`, `job_info.json`, `command.txt`, `nvidia_smi.txt`, `gpu_mem_samples.txt`.

- [ ] **Step 1: Create `hpc/train.pbs`**

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -N prendo_train
#PBS -j oe
set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME}"
source "$HOME/prendo/code/PR-ENDO/hpc/env.sh"
source "$PRENDO_CODE/hpc/train.env"

RUN_DIR="$PRENDO_RUNS/$RUN_NAME"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/train.log"
exec > >(tee -a "$LOG") 2>&1

echo "PBS_JOBID=${PBS_JOBID:-unknown}"
echo "host=$(hostname)"
nvidia-smi | tee "$RUN_DIR/nvidia_smi.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits > "$RUN_DIR/gpu_info.txt"

(
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    sleep 10
  done
) > "$RUN_DIR/gpu_mem_samples.txt" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

cd "$PRENDO_CODE"

echo "=== pre-flight reader validation ==="
python tools/validate_vrcaps_interface.py \
  --data "$PRENDO_DATA" \
  --repo "$PRENDO_CODE" \
  --report "$PRENDO_LOGS/interface_cpu.json"

echo "=== pre-flight GPU smoke ==="
if python tools/validate_vrcaps_interface.py \
  --data "$PRENDO_DATA" \
  --repo "$PRENDO_CODE" \
  --gpu \
  --resolution 1 \
  --report "$PRENDO_LOGS/smoke_report.json"; then
  SMOKE_RES=1
else
  echo "GPU smoke at resolution 1 failed; retrying at resolution 2"
  python tools/validate_vrcaps_interface.py \
    --data "$PRENDO_DATA" \
    --repo "$PRENDO_CODE" \
    --gpu \
    --resolution 2 \
    --report "$PRENDO_LOGS/smoke_report.json"
  SMOKE_RES=2
fi

SMOKE_PEAK=$(python -c "import json; print(json.load(open('$PRENDO_LOGS/smoke_report.json'))['gpu']['peak_vram_gb'])")
if [ "$SMOKE_RES" -eq 1 ] && python -c "import sys; sys.exit(0 if float('$SMOKE_PEAK') <= 40.0 else 1)"; then
  RESOLUTION=1
else
  RESOLUTION=2
fi
echo "smoke resolution ${SMOKE_RES}, peak VRAM ${SMOKE_PEAK} GB; training resolution ${RESOLUTION}"

COMMAND="python train_viewdir.py --model_path $RUN_DIR -s $PRENDO_DATA --eval --resolution $RESOLUTION --iteration $ITERATIONS --lambda_depth 0.2 --lambda_norm 0.1 --albedo_loss_weight 1.0 --densification_interval 1000 --opacity_reset_interval 3000 --densify_until_iter 10000 --densify_from_iter 1000 --mlp_lr 0.0005 --grid_lr 0.0005 --end_diffuse_loss_iter 20000 --port 6000 --K_normals $K_NORMALS --diffuse_loss_weight $DIFFUSE_LOSS_WEIGHT"
printf '%s\n' "$COMMAND" > "$RUN_DIR/command.txt"

START=$(date +%s)
set +e
eval "$COMMAND"
STATUS=$?
set -e
END=$(date +%s)

kill "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

python - "$RUN_DIR" "$STATUS" "$START" "$END" "$RESOLUTION" "$ITERATIONS" <<'PY'
import json
import os
import sys

run_dir, status, start, end, resolution, iterations = sys.argv[1:]
info = {
    "pbs_jobid": os.environ.get("PBS_JOBID"),
    "hostname": os.uname().nodename,
    "exit_status": int(status),
    "walltime_seconds": int(end) - int(start),
    "resolution": int(resolution),
    "iterations": int(iterations),
    "command_file": os.path.join(run_dir, "command.txt"),
}
with open(os.path.join(run_dir, "job_info.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))
PY

if [ "$STATUS" -ne 0 ]; then
  echo "training failed with status $STATUS"
  exit "$STATUS"
fi
test -f "$RUN_DIR/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
test -f "$RUN_DIR/chkpnt${ITERATIONS}.pth"
test -f "$RUN_DIR/chkpnt_mlp${ITERATIONS}.pth"
test -f "$RUN_DIR/chkpnt_grid_camera${ITERATIONS}.pth"
test -f "$RUN_DIR/chkpnt_light${ITERATIONS}.pth"
if grep -q "LOSS NAN" "$RUN_DIR/train.log"; then
  echo "LOSS NAN detected"
  exit 2
fi

echo "=== post-training render ==="
python render.py \
  --model_path "$RUN_DIR" \
  --eval \
  -s "$PRENDO_DATA" \
  --skip_train \
  --iteration "$ITERATIONS"

python render_light_rotations.py \
  --iteration "$ITERATIONS" \
  --model_path "$RUN_DIR" \
  --eval \
  -s "$PRENDO_DATA" \
  --spotlight_attenuation_power 10 \
  --max_rot 12

python metrics.py "$PRENDO_RUNS"

test -d "$RUN_DIR/test/ours_${ITERATIONS}/renders"
test -d "$RUN_DIR/rotate_light_video_maxrot12_cospower10"
test -f "$PRENDO_RUNS/mean_results.txt"
echo "training and render complete"

echo "=== packaging ==="
bash "$PRENDO_CODE/hpc/package_bundle.sh"
echo "pipeline complete"
```

- [ ] **Step 2: Syntax-check and submit**

```bash
bash -n hpc/train.pbs
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'cd $HOME/prendo/code/PR-ENDO && qsub hpc/train.pbs'
```

Expected: a numeric PBS job id.

- [ ] **Step 3: Monitor checkpoints and log**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'qstat -u yl15425; tail -n 40 /rds/general/user/yl15425/ephemeral/prendo/runs/vrcaps_lumen_100_r30k/train.log; ls /rds/general/user/yl15425/ephemeral/prendo/runs/vrcaps_lumen_100_r30k/point_cloud'
```

Expected over time: `iteration_1000`, `iteration_10000`, `iteration_20000`, `iteration_30000`. At completion, the log contains final test metrics and `training complete`.

- [ ] **Step 4: Handle interruptions**

If the job is killed by walltime or a node failure, resubmit with the same command plus:

```bash
--start_checkpoint "$RUN_DIR/chkpnt<last_saved_iteration>.pth"
```

Do not change hyperparameters when resuming. Record the interruption and resumed iteration in `docs/runbooks/vrcaps_training.md`.

- [ ] **Step 5: Verify completion**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'RUN=/rds/general/user/yl15425/ephemeral/prendo/runs/vrcaps_lumen_100_r30k; \
   test -f $RUN/point_cloud/iteration_30000/point_cloud.ply; \
   test -f $RUN/chkpnt_light30000.pth; \
   ! grep -q "LOSS NAN" $RUN/train.log; \
   tail -n 20 $RUN/results.txt'
```

Expected: exit 0 and the final `[ITER 30000] Evaluating test avg:` line.

- [ ] **Step 6: Commit**

```bash
git add hpc/train.pbs
git commit -m "hpc: add 30k training job"
```

---

### Task 8: Post-training render, light rotations, metrics

**Files:**
- Create: `hpc/render.pbs`

**Interfaces:**
- Consumes: trained run directory; `prendo` env.
- Produces: `test/ours_30000/{renders,gt,albedo,normals,disable_reflections}`, `rotate_light_video_maxrot12_cospower10/`, `$PRENDO_RUNS/mean_results.txt`.

- [ ] **Step 1: Create `hpc/render.pbs`**

```bash
#!/bin/bash
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=02:00:00
#PBS -N prendo_render
#PBS -j oe
set -euo pipefail

cd "${PBS_O_WORKDIR:-$HOME}"
source "$HOME/prendo/code/PR-ENDO/hpc/env.sh"
source "$PRENDO_CODE/hpc/train.env"

RUN_DIR="$PRENDO_RUNS/$RUN_NAME"
cd "$PRENDO_CODE"

python render.py \
  --model_path "$RUN_DIR" \
  --eval \
  -s "$PRENDO_DATA" \
  --skip_train \
  --iteration "$ITERATIONS"

python render_light_rotations.py \
  --iteration "$ITERATIONS" \
  --model_path "$RUN_DIR" \
  --eval \
  -s "$PRENDO_DATA" \
  --spotlight_attenuation_power 10 \
  --max_rot 12

python metrics.py "$PRENDO_RUNS"

test -d "$RUN_DIR/test/ours_${ITERATIONS}/renders"
test -d "$RUN_DIR/test/ours_${ITERATIONS}/albedo"
test -d "$RUN_DIR/test/ours_${ITERATIONS}/normals"
test -d "$RUN_DIR/rotate_light_video_maxrot12_cospower10"
test -f "$PRENDO_RUNS/mean_results.txt"
echo "render complete"
```

- [ ] **Step 2: Submit and wait**

```bash
bash -n hpc/render.pbs
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'cd $HOME/prendo/code/PR-ENDO && qsub hpc/render.pbs'
```

- [ ] **Step 3: Verify outputs**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'RUN=/rds/general/user/yl15425/ephemeral/prendo/runs/vrcaps_lumen_100_r30k; \
   find $RUN/test/ours_30000 -maxdepth 1 -type d; \
   ls $RUN/rotate_light_video_maxrot12_cospower10; \
   tail -n 30 /rds/general/user/yl15425/ephemeral/prendo/runs/mean_results.txt'
```

Expected: five test subdirectories, at least one `.mp4`, and metric means.

- [ ] **Step 4: Commit**

```bash
git add hpc/render.pbs
git commit -m "hpc: add post-training render job"
```

---

### Task 9: Manifest collector and bundle

**Files:**
- Create: `tools/collect_artifacts.py`
- Create: `tests/test_collect_artifacts.py`
- Create: `hpc/package_bundle.sh`

**Interfaces:**
- Consumes: run directory, dataset, patch, env logs.
- Produces: `run_manifest.json`; `${RUN_NAME}_bundle.tar.gz` and `.sha256` in `$PRENDO_ARTIFACTS`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_collect_artifacts.py`:

```python
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.collect_artifacts import collect


class CollectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.run_dir = base / "run"
        self.data_dir = base / "data"
        (self.run_dir / "point_cloud/iteration_30000").mkdir(parents=True)
        (self.run_dir / "test/ours_30000/renders").mkdir(parents=True)
        (self.run_dir / "point_cloud/iteration_30000/point_cloud.ply").write_bytes(b"ply")
        (self.run_dir / "chkpnt30000.pth").write_bytes(b"ckpt")
        (self.run_dir / "results.txt").write_text(
            "[ITER 30000] Evaluating test avg: L1 0.1000 PSNR 20.5000 LPIPS 0.2000 SSIM 0.7000 MSSIM 0.8000\n",
            encoding="utf-8",
        )
        (self.run_dir / "test/ours_30000/renders/0007.png").write_bytes(b"png")
        (self.run_dir / "job_info.json").write_text(
            json.dumps({"pbs_jobid": "123", "walltime_seconds": 42, "resolution": 1, "iterations": 30000}),
            encoding="utf-8",
        )
        (self.run_dir / "nvidia_smi.txt").write_text(
            "full nvidia-smi output\n", encoding="utf-8"
        )
        (self.run_dir / "gpu_info.txt").write_text(
            "NVIDIA L40S, 46068\n", encoding="utf-8"
        )
        (self.run_dir / "gpu_mem_samples.txt").write_text("1024\n30720\n2048\n", encoding="utf-8")
        (self.run_dir / "command.txt").write_text("python train_viewdir.py\n", encoding="utf-8")
        self.data_dir.mkdir()
        for name in ("transforms.json", "transforms_test.json", "prendo_config.json", "init_point_cloud.ply"):
            (self.data_dir / name).write_bytes(name.encode())
        self.patch = base / "patch.diff"
        self.patch.write_bytes(b"patch")

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_fields_and_hashes(self):
        out = Path(self.tmp.name) / "run_manifest.json"
        manifest = collect(self.run_dir, self.data_dir, self.patch, out)
        saved = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(saved["run_id"], "run")
        self.assertEqual(saved["metrics"]["psnr"], 20.5)
        self.assertEqual(saved["hardware"]["gpu"], "NVIDIA L40S")
        self.assertAlmostEqual(saved["peak_vram_gb"], 30.0, places=3)
        self.assertEqual(manifest["reader_patch_sha256"], hashlib.sha256(b"patch").hexdigest())
        paths = {o["path"] for o in manifest["outputs"]}
        self.assertIn("point_cloud/iteration_30000/point_cloud.ply", paths)
        self.assertIn("test/ours_30000/renders/0007.png", paths)
        self.assertEqual(
            manifest["dataset"]["transforms.json"],
            hashlib.sha256(b"transforms.json").hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python3 -m unittest tests.test_collect_artifacts -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.collect_artifacts'`.

- [ ] **Step 3: Implement the collector**

Create `tools/collect_artifacts.py`:

```python
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
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
python3 -m unittest tests.test_collect_artifacts -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Create `hpc/package_bundle.sh`**

```bash
#!/bin/bash
set -euo pipefail

source "$HOME/prendo/code/PR-ENDO/hpc/paths.sh"
source "$PRENDO_CODE/hpc/train.env"

RUN_DIR="$PRENDO_RUNS/$RUN_NAME"
BUNDLE_DIR="$PRENDO_RUNS/bundles"
mkdir -p "$BUNDLE_DIR" "$PRENDO_ARTIFACTS"

mkdir -p "$RUN_DIR/environment"
cp "$PRENDO_LOGS/pip_freeze.txt" "$RUN_DIR/environment/" 2>/dev/null || true
cp "$PRENDO_LOGS/conda_export.txt" "$RUN_DIR/environment/" 2>/dev/null || true

python3 "$PRENDO_CODE/tools/collect_artifacts.py" \
  --run-dir "$RUN_DIR" \
  --dataset "$PRENDO_DATA" \
  --patch "$PRENDO_CODE/patches/vrcaps_reader.patch" \
  --output "$RUN_DIR/run_manifest.json"

cd "$PRENDO_RUNS"
tar -czf "$BUNDLE_DIR/${RUN_NAME}_bundle.tar.gz" "$RUN_NAME"
cd "$BUNDLE_DIR"
sha256sum "${RUN_NAME}_bundle.tar.gz" \
  | tee "${RUN_NAME}_bundle.sha256"
cp "$BUNDLE_DIR/${RUN_NAME}_bundle.tar.gz" "$PRENDO_ARTIFACTS/"
cp "$BUNDLE_DIR/${RUN_NAME}_bundle.sha256" "$PRENDO_ARTIFACTS/"
echo "bundle ready at $PRENDO_ARTIFACTS/${RUN_NAME}_bundle.tar.gz"
```

- [ ] **Step 6: Run on HPC and download**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'bash $HOME/prendo/code/PR-ENDO/hpc/package_bundle.sh'

mkdir -p /home/yiping/Documents/my_repo/PR-ENDO/artifacts
rsync -av -e "ssh -S ~/.ssh/hpc.sock" \
  'yl15425@login.cx3.hpc.imperial.ac.uk:$HOME/prendo/artifacts/vrcaps_lumen_100_r30k_bundle.tar.gz' \
  /home/yiping/Documents/my_repo/PR-ENDO/artifacts/
rsync -av -e "ssh -S ~/.ssh/hpc.sock" \
  'yl15425@login.cx3.hpc.imperial.ac.uk:$HOME/prendo/artifacts/vrcaps_lumen_100_r30k_bundle.sha256' \
  /home/yiping/Documents/my_repo/PR-ENDO/artifacts/
cd /home/yiping/Documents/my_repo/PR-ENDO/artifacts
sha256sum -c vrcaps_lumen_100_r30k_bundle.sha256
```

Expected: `OK` from `sha256sum -c`; download succeeds.

- [ ] **Step 7: Commit**

```bash
git add tools/collect_artifacts.py tests/test_collect_artifacts.py hpc/package_bundle.sh
git commit -m "tools: add artifact manifest and bundle packaging"
```

---

### Task 10: Runbook and final acceptance

**Files:**
- Create: `docs/runbooks/vrcaps_training.md`

**Interfaces:**
- Consumes: all prior outputs.
- Produces: an operator runbook and a final acceptance record.

- [ ] **Step 1: Write the runbook**

Create `docs/runbooks/vrcaps_training.md` with:

```markdown
# VRCaps PR-ENDO training runbook

## Paths
- code: `$HOME/prendo/code/PR-ENDO`
- env: `$HOME/prendo/envs/prendo` (created by `hpc/env_build.pbs`)
- data: `/rds/general/user/yl15425/ephemeral/prendo/data/VRCaps_lumen_100`
- run: `/rds/general/user/yl15425/ephemeral/prendo/runs/vrcaps_lumen_100_r30k`
- artifacts: `$HOME/prendo/artifacts`

## Ordered commands
1. `./hpc/sync_to_hpc.sh`
2. `ssh ... 'cd $HOME/prendo/code/PR-ENDO && qsub hpc/env_build.pbs'`
3. `ssh ... 'cd $HOME/prendo/code/PR-ENDO && qsub hpc/smoke.pbs'`
4. `ssh ... 'cd $HOME/prendo/code/PR-ENDO && qsub hpc/train.pbs'`
5. `ssh ... 'cd $HOME/prendo/code/PR-ENDO && qsub hpc/render.pbs'`
6. `ssh ... 'bash $HOME/prendo/code/PR-ENDO/hpc/package_bundle.sh'`

## Resume
If training is interrupted, resubmit `hpc/train.pbs` with:
`--start_checkpoint "$RUN_DIR/chkpnt<last_saved_iteration>.pth"`
Record the interruption and resumed iteration here.

## Failure signatures
- `LOSS NAN`: keep the last checkpoint and log, inspect depth percentiles and loss terms, do not silently change hyperparameters.
- CUDA OOM: set `RESOLUTION=2` in `hpc/train.env`, record the measured peak VRAM, resubmit.
- Missing `extensions ok`: rebuild the env and inspect the compiler/CUDA module versions.

## Acceptance record
- [ ] interface CPU validation passed
- [ ] GPU smoke passed; peak VRAM recorded
- [ ] 30k training completed without `LOSS NAN`
- [ ] render + light rotation videos produced
- [ ] `run_manifest.json` written
- [ ] bundle downloaded and sha256 verified
```

- [ ] **Step 2: Run the final acceptance checks**

```bash
ssh -S ~/.ssh/hpc.sock yl15425@login.cx3.hpc.imperial.ac.uk \
  'RUN=/rds/general/user/yl15425/ephemeral/prendo/runs/vrcaps_lumen_100_r30k; \
   test -f $RUN/run_manifest.json; \
   test -f $RUN/point_cloud/iteration_30000/point_cloud.ply; \
   test -f $RUN/chkpnt_light30000.pth; \
   ! grep -q "LOSS NAN" $RUN/train.log; \
   echo acceptance-ok'
```

Expected: `acceptance-ok`.

- [ ] **Step 3: Commit**

```bash
git add docs/runbooks/vrcaps_training.md
git commit -m "docs: add VRCaps training runbook"
```

---

## Self-Review

- **Spec coverage:** Task 1 covers branch/patch/submodules; Task 2 covers data staging/layout audit; Task 3 covers HPC paths/sync; Task 4 covers env build; Task 5 covers interface validation; Task 6 covers GPU smoke and resolution decision; Task 7 covers 30k training; Task 8 covers renders/metrics; Task 9 covers manifest/bundle; Task 10 covers runbook/acceptance. All spec sections 6.1–6.5, 7, 8, 9, 10, 11 are represented.
- **Placeholder scan:** no TBD/TODO; the runbook has a blank interruption-record section by design, and the manifest schema fields are filled by code.
- **Type consistency:** `audit()` returns the documented dict; `validate_vrcaps_interface.py` consumes `SceneInfo.train_cameras/test_cameras`; `collect()` consumes `job_info.json`, `nvidia_smi.txt`, `gpu_mem_samples.txt`, `command.txt` exactly as produced by `train.pbs`; bundle name matches `RUN_NAME`.
