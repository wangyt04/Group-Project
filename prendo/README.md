# PR-ENDO side: VRCaps relighting

This folder contains the PR-ENDO-side code, HPC pipeline, run metadata and B1 configuration snapshot used for the VRCaps relighting handoff. Image data, checkpoints and bundles are delivered separately via OneDrive; they are not stored in this repository.

## Code commit

- Group-Project code snapshot: `a13e272206e0477fc380b5500363f96d2995a33e`
- `runs/20261112-0332/record.json` records `code_commit` as `https://github.com/wangyt04/Group-Project@a13e272206e0477fc380b5500363f96d2995a33e`
- Underlying PR-ENDO branch commit used for rendering: `d2a6e4422add68f3441a0a3da3e60d5a406d625a`

## Official base

- Repository: `https://github.com/SanoScience/PR-ENDO`
- Base commit: `ee5362817ccc944292ae43aa9f667386a5ebc069`
- The only source change is `patches/vrcaps_reader.patch` (VR-Caps reader + dispatch branch).

## C3VD vs VRCaps interface

Official PR-ENDO's C3VD reader (`readColonSceneInfo`) does not read a plain C3VD folder. It derives `C3VD_endogslam_optimized/` and consumes EndoGSLAM's `params.npz` (`cam_unnorm_rots`, `cam_trans`, `intrinsics`) plus `point_cloud/iteration_*/point_cloud.ply`.

VRCaps is synthetic: Unity provides GT OpenGL c2w poses, depth and a back-projected XYZ cloud. It therefore uses the RotateColon-style reader via `patches/vrcaps_reader.patch`; no EndoGSLAM is run, and the VRCaps path must not contain `C3VD`.

## Contents

| Path | Contents |
| --- | --- |
| `patches/vrcaps_reader.patch` | Official reader patch, applied with `git apply -p0` |
| `tools/` | Dataset audit, interface validation, artifact manifest, relight renderer |
| `tests/` | Stdlib unit tests for the audit and manifest tools |
| `hpc/` | PBS environment build, smoke, training, render, relight, packaging scripts |
| `docs/superpowers/` | Training design spec and implementation plan |
| `docs/runbooks/vrcaps_training.md` | Environment/training/relight runbook and run records |
| `runs/20261112-0332/` | Relight `record.json`, `render_manifest.json`, `train_metrics.json` |
| `runs/vrcaps_lumen_100_v2_r30k/` | Training manifest, results, command and cfg_args |
| `configs/VRCaps_lumen_100_v2_relight_config/` | B1 metadata snapshot (JSON only; no B2 images) |

## Reproduce on a PR-ENDO checkout

```bash
git clone https://github.com/SanoScience/PR-ENDO
cd PR-ENDO
git checkout ee5362817ccc944292ae43aa9f667386a5ebc069
git apply -p0 /path/to/Group-Project/prendo/patches/vrcaps_reader.patch
cp -r /path/to/Group-Project/prendo/tools .
cp -r /path/to/Group-Project/prendo/hpc .
cp -r /path/to/Group-Project/prendo/tests .
```

The HPC scripts contain the Imperial CX3 paths for user `yl15425` (`hpc/paths.sh`, `hpc/train.env`, `hpc/sync_to_hpc.sh`); adapt those paths/username for another account.

## Data and runs

- Training data: `VRCaps_lumen_100_v2` (point-light training data, 88 train / 12 test).
- B1 configuration: `VRCaps_lumen_100_v2_relight_config` (25 configs, 49 relight cameras, 744 `(config, frame)` entries, 671 valid, 73 invalid L3).
- Training: 30,000 iterations, resolution 1, 213,091 Gaussians. Test metrics: PSNR 33.64 / SSIM 0.99 / LPIPS 0.06. Training-frame metrics: PSNR 34.787 / SSIM 0.9886 / LPIPS 0.0580.
- Relight run `20261112-0332`: **671 valid `(config, frame)` pairs** (391 non-L2 + 280 L2; PNG + EXR). L2 was rendered with `n=23.13538193587864`; calibration artifacts are in `runs/20261112-0332/calibration/`.
- The exact code commit and run command are recorded in `runs/20261112-0332/record.json` and `runs/vrcaps_lumen_100_v2_r30k/command.txt`.
- B2 GT images are not part of the PRENDO delivery; Yutong receives them from the VRCaps group.

## Notes

- `shadow` / `noshadow` variants are merged: PR-ENDO produces one image per `(config, frame)`, per the handoff.
- The mask companion is not applied; the reader ignores mask keys.
- Test depth is an internal placeholder, not test ground truth.
- Physical/numerical scale is unverified (`mm_per_unit=null`).
- PNG is display-space RGB8; EXR is linear RGB float16.
