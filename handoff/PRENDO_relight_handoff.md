# PR-ENDO → Yutong relight handoff

> **2026-11-13 update — L2 superseded.** The L2 images in the `20261112-0332` bundle use the
> old 60° / n=23.135 profile and must not be used for the new L2 evaluation. The L2 delivery is
> now the separate run **`20261113-0110`**: 220 entries (7 directions), 200,000 cd (4× base),
> outer FULL 160° / inner HALF 10°, `n=2.400499`. Evaluation vs the new B2 L2 GT: PSNR 32.99 dB,
> +11.56 dB over the no-relight L0 baseline, SSIM 0.9863, LPIPS 0.0823, ratio_corr 0.993; all
> direction checks pass. See `prendo/runs/20261113-0110/EVALUATION.md`. `L2_yaw±40` is retired;
> the non-L2 levels in the old bundle remain valid.

## 1. Files

| File | Contents | Size | SHA256 |
|---|---|---|---|
| `vrcaps_lumen_100_v2_r30k_bundle.tar.gz` | V2 training run: model, checkpoints, test renders, metrics, manifest | 337 MB | `bb314d69ed15385615bf1fbe232bae679010bcf98cb75dd7cd8e5b5a2bfaeb93` |
| `pred_VRCaps_lumen_100_v2_20261112-0332.tar.gz` | Relight predictions (671 entries: 391 non-L2 + 280 old-profile L2) + metadata; **its L2 is superseded** | 2.6 GB | `668b8622158b9b6abdfad09950914e6ced0df42efb74a08da5f49268165e13e4` |
| `pred_VRCaps_lumen_100_v2_20261113-0110.tar.gz` | **L2 4× delivery** (220 entries: 7 directions, 30 test + 5 calib cameras) + `record.json` / `render_manifest.json` / `train_metrics.json` | 1.0 GB | `ae192089e286ba8aa6546d4d1ec91f59840e96afdacd7f136bd72be9231f0930` |

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
- L2 spot angles: outer half-angle 30° (Unity full 60°), inner half-angle 0° (HDRP innerSpotPercent=0); confirmed for this delivery (used for the B2 reference renders).

## 4. What is in this delivery

`pred_VRCaps_lumen_100_v2_20261112-0332/` contains **671 valid entries**:

- L0, L1_s0.5, L1_s2.0
- L2_fwd, L2_yaw±10/±20/±40, L2_pitch±20
- L3_f+0.25, L3_f-0.25, L3_r0.10, L3_r0.25, L3_r0.50, L3_u0.25
- L4_red, L4_green, L4_blue
- L5_seg1, L5_seg2

`L3_u0.50` and `L3_r1.00` have no valid entries (all frames failed the geometry validity check). `record.json` has `spot_power=23.13538193587864`, `iteration=30000`, train metrics PSNR 34.787 / SSIM 0.9886 / LPIPS 0.0580, and total 671 PNG+EXR pairs.

## 5. L2 calibration and completion (old 60° profile; superseded 2026-11-13)

L2's 280 entries were rendered with **n=23.13538193587864**, fitted from the B2 calibration frames `0001, 0032, 0050, 0067, 0097` (L0 and L2_fwd noshadow + depth). Fit RMSE is 0.019458; see `prendo/runs/20261112-0332/calibration/spot_fit.png`.

The HDRP spotlight is not an exact `cos^n`; the measured profile differs somewhat from the fitted curve, so L2 comparisons should be interpreted with this approximation. The final bundle is 2.6 GB with sha256 `668b8622158b9b6abdfad09950914e6ced0df42efb74a08da5f49268165e13e4`; `record.json` records the completed total of 671.

## 6. Notes

- B2 GT images are **not** part of the PRENDO delivery; Yutong receives B2 from the VRCaps group and compares.
- `shadow` / `noshadow` variants are merged: PR-ENDO produces one image per `(config, frame)`, as required by the handoff.
- The mask companion was not applied; the reader ignores mask keys.
- Test depth is an internal placeholder, not test ground truth.
- Physical/numerical scale is unverified (`mm_per_unit=null`).
- EXR is linear RGB, float16; PNG is display-space RGB8. Maximum PNG-vs-EXR difference on a smoke frame was 3.9e-3, consistent with 8-bit/half quantization.
