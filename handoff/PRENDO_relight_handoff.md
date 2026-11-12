# PR-ENDO → Yutong relight handoff

## 1. Files

| File | Contents | Size | SHA256 |
|---|---|---|---|
| `vrcaps_lumen_100_v2_r30k_bundle.tar.gz` | V2 training run: model, checkpoints, test renders, metrics, manifest | 337 MB | `bb314d69ed15385615bf1fbe232bae679010bcf98cb75dd7cd8e5b5a2bfaeb93` |
| `pred_VRCaps_lumen_100_v2_20261112-0332.tar.gz` | Non-L2 relight predictions + `record.json` / `render_manifest.json` / `train_metrics.json` | 1.8 GB | `ffccc7ea23fb5096f09a90f6c7864cb0c209a5c11405cc480bd1d47ae9e6f36f` |

`pred_VRCaps_lumen_100_v2_20261112-0332/` contains:

```
record.json
render_manifest.json
train_metrics.json
<config>/<frame>.png    display-space RGB8
<config>/<frame>.exr    linear RGB, float16
```

## 2. Model and training

- Dataset: `VRCaps_lumen_100_v2` (point-light training data; 88 train / 12 test).
- Base PR-ENDO commit: `ee5362817ccc944292ae43aa9f667386a5ebc069`.
- Code commit: `https://github.com/wangyt04/Group-Project@a13e272206e0477fc380b5500363f96d2995a33e` (the `prendo/` code snapshot; the underlying PR-ENDO branch commit used for rendering is `d2a6e4422add68f3441a0a3da3e60d5a406d625a`).
- Iteration: 30,000; resolution 1 (1350×1080); 213,091 Gaussians.
- Training-frame metrics: PSNR **34.787**, SSIM **0.9886**, LPIPS **0.0580** (88 views).
- Test-frame metrics: PSNR **33.64**, SSIM **0.99**, LPIPS **0.06** (12 held-out views).

## 3. B1 configuration

Source: `VRCaps_lumen_100_v2_relight_config` (B1 only; no B2 images).

- 25 configs: L0 (1), L1 (2), L2 (9), L3 (8), L4 (3), L5 (2).
- 49 unique relight cameras, all from the 88 training frames; no held-out camera selected.
- 744 `(config, frame)` entries; 73 invalid L3 entries marked `valid=false`.
- 671 valid entries; 1009 planned B2 variant paths; 863 eligible after validity filtering.
- L2 spot angles: outer half-angle 30° (Unity full 60°), inner half-angle 0° (HDRP innerSpotPercent=0); provisional pending Yutong's FOV/cone confirmation.

## 4. What is in this delivery

Current `pred_..._20261112-0332` contains **391 non-L2 valid entries**:

- L0, L1_s0.5, L1_s2.0
- L3_f+0.25, L3_f-0.25, L3_r0.10, L3_r0.25, L3_r0.50, L3_u0.25
- L4_red, L4_green, L4_blue
- L5_seg1, L5_seg2

`L3_u0.50` and `L3_r1.00` have no valid entries (all frames failed the geometry validity check). `L2` is not rendered yet.

`record.json` currently has `spot_power: null` and a note that L2 is pending.

## 5. Pending: L2 (280 entries)

L2 needs Yutong's fitted spotlight exponent `n`. The B1 `lights.json` has `spot_power_n: null`; do not guess it.

When `n` is available, run an L2-only job **into the same run folder**:

```bash
qsub -l walltime=04:00:00 \
  -v RELIGHT_RUN_ID=20261112-0332,LEVELS=L2,SPOT_POWER_N=<n>,SKIP_TRAIN_METRICS=1,CODE_COMMIT=<code_commit> \
  hpc/relight.pbs
```

The script will:

- load the existing `train_metrics.json`;
- render the 280 L2 valid entries into the same `pred_VRCaps_lumen_100_v2_20261112-0332/`;
- overwrite `record.json` with `spot_power=n` and total count 671;
- regenerate the bundle and `.sha256`.

After that, download and verify the updated bundle; the complete handoff is 671 PNG+EXR pairs.

## 6. Notes

- B2 GT images are **not** part of the PRENDO delivery; Yutong receives B2 from the VRCaps group and compares.
- `shadow` / `noshadow` variants are merged: PR-ENDO produces one image per `(config, frame)`, as required by the handoff.
- The mask companion was not applied; the reader ignores mask keys.
- Test depth is an internal placeholder, not test ground truth.
- Physical/numerical scale is unverified (`mm_per_unit=null`).
- EXR is linear RGB, float16; PNG is display-space RGB8. Maximum PNG-vs-EXR difference on a smoke frame was 3.9e-3, consistent with 8-bit/half quantization.
