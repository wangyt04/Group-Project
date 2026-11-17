# VRCaps_lumen_100_v2 PRENDO model (30k)

OneDrive upload card for the trained model. No OneDrive client is mounted on the
preparation machine; upload the files below manually.

## Model

| Item | Value |
|---|---|
| Dataset | `VRCaps_lumen_100_v2` (point-light training data; 88 train / 12 test) |
| Base PR-ENDO commit | `ee5362817ccc944292ae43aa9f667386a5ebc069` |
| Code snapshot | `https://github.com/wangyt04/Group-Project@a13e272206e0477fc380b5500363f96d2995a33e` (PR-ENDO rendering branch commit `d2a6e4422add68f3441a0a3da3e60d5a406d625a`) |
| Iteration | 30,000 |
| Resolution | 1 (1350 × 1080) |
| Gaussians | 213,091 |
| Training-frame metrics | PSNR **34.787**, SSIM **0.9886**, LPIPS **0.0580** (88 views) |
| Test-frame metrics | PSNR **33.64**, SSIM **0.99**, LPIPS **0.06** (12 held-out views) |
| GPU / walltime | NVIDIA L40S; training walltime 1,131 s; peak VRAM 9.18 GB |

The model is unchanged for the L2 4× relight delivery (run `20261113-0110`); no retraining
was performed for L2.

## Upload set (recommended)

Folder on OneDrive: `PRENDO/vrcaps_lumen_100_v2_r30k/`

| File | Size | SHA256 |
|---|---:|---|
| `vrcaps_lumen_100_v2_r30k_bundle.tar.gz` | 337 MB | `bb314d69ed15385615bf1fbe232bae679010bcf98cb75dd7cd8e5b5a2bfaeb93` |
| `vrcaps_lumen_100_v2_r30k_bundle.sha256` | 105 B | — |
| `vrcaps_lumen_100_v2_r30k_model_only.tar.gz` (optional, quick model-only copy) | 12 MB | `24778f363278695df280e584d22bb9e7ff9630318e7cee69a26bf498b11c39db` |
| `vrcaps_lumen_100_v2_r30k_model_only.tar.gz.sha256` | 109 B | — |
| `MODEL_CARD.md` | this file | — |

Local paths:

```
/home/yiping/Documents/my_repo/PR-ENDO/artifacts/
├── vrcaps_lumen_100_v2_r30k_bundle.tar.gz
├── vrcaps_lumen_100_v2_r30k_bundle.sha256
├── vrcaps_lumen_100_v2_r30k_model_only.tar.gz
├── vrcaps_lumen_100_v2_r30k_model_only.tar.gz.sha256
└── vrcaps_lumen_100_v2_r30k/                  # extracted full training run (412 MB)
```

If the teammate also needs the L2 predictions, use a separate folder
`PRENDO/L2_4x_20261113-0110/` with:

| File | Size | SHA256 |
|---|---:|---|
| `pred_VRCaps_lumen_100_v2_20261113-0110.tar.gz` | 1.0 GB | `ae192089e286ba8aa6546d4d1ec91f59840e96afdacd7f136bd72be9231f0930` |
| `pred_VRCaps_lumen_100_v2_20261113-0110.sha256` | 112 B | — |
| `EVALUATION.md` | 8 KB | from `prendo/runs/20261113-0110/EVALUATION.md` |

## What is inside the full bundle

- `point_cloud/iteration_{0,1000,10000,20000,30000}/point_cloud.ply` — Gaussian models
- `chkpnt{1000,10000,20000,30000}.pth` — full training checkpoints (resume)
- `chkpnt_mlp*.pth` — diffuse MLP weights
- `chkpnt_grid_camera*.pth` — camera-embedding grid
- `chkpnt_light*.pth` — learned light parameters
- `cfg_args`, `command.txt`, `results.txt`, `job_info.json`, `run_manifest.json`
- `test/ours_30000/{renders,gt,albedo,normals,disable_reflections}/` — 64 test renders
- `rotate_gifs/`, `rotate_light_video_maxrot12_cospower10/` — qualitative videos
- `environment/{conda_export.txt,pip_freeze.txt}` — environment reproduction
- `train.log`, `events.out.tfevents.*`, `gpu_*`, `render_time.txt`

The model-only archive contains the minimal forward-pass set:
`point_cloud/iteration_30000/point_cloud.ply`, `chkpnt_mlp30000.pth`,
`chkpnt_grid_camera30000.pth`, `chkpnt_light30000.pth`, plus `cfg_args`, `command.txt`,
`results.txt`, `job_info.json`, `run_manifest.json`, and `environment/`.

## How to use

Extract the bundle and point the PRENDO code at the run directory:

```bash
tar -xzf vrcaps_lumen_100_v2_r30k_bundle.tar.gz
# PRENDO base commit ee53628 + the VRCaps reader patch (patches/vrcaps_reader.patch)
# render: --model <extracted>/vrcaps_lumen_100_v2_r30k --iteration 30000
```

The training command and hyperparameters are in `command.txt`; hashes of the delivered
files are in `run_manifest.json`.

## Verification

After uploading/downloading, verify:

```bash
sha256sum -c vrcaps_lumen_100_v2_r30k_bundle.sha256
sha256sum -c vrcaps_lumen_100_v2_r30k_model_only.tar.gz.sha256
```
