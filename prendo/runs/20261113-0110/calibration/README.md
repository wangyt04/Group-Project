# Spotlight exponent n calibration — L2 4x, outer FULL160 / inner HALF10

- Source: `VRCaps_lumen_100_v2_L2_4x_Outer160_InnerHalf10_Matched_Calibration.zip`
  (independent calibration root; auxiliary 200,000 cd Point `L0`, matched 200,000 cd spot).
- Frames: `0001, 0032, 0050, 0067, 0097` (`split=calib`).
- Configs in the fit: `L2_fwd` plus the four supplementary probes
  `CAL_L2_yaw±20`, `CAL_L2_pitch±20`. Probes are calibration-only, not evaluation tuning.
- HDRP spot: outer FULL 160° (half 80°), inner HALF 10° (FULL 20°),
  `innerSpotPercent = 12.5`, EV100 = 20, noshadow, linear HDR float32.
- Fitted with `relight_eval/fit_spot_power_probes.py` (probe-aware; the legacy
  forward-only `fit_spot_power.py` is not used for this profile).

## Fit results

| Fit input | n | RMSE | pixels |
|---|---|---|---|
| forward `L2_fwd` only (angles 0–52.4°) | 2.351407 | 0.01356 | 5,246,264 |
| probes ±20 only (angles 0–~69°) | 2.414376 | 0.02156 | 20,985,056 |
| **all (used for rendering)** | **2.400499** | **0.02047** | **26,231,320** |

The probe-aware value **n = 2.4004990755616116** is used for the new L2-only render.
The forward-only value is reported for transparency; the evaluation views reach angles
beyond 52.4°, so the probe-aware fit is the appropriate one.

## Profile and residual notes

- The inner 10° half-angle is a flat core (measured ratio 1.000); from ~10° the HDRP
  rolloff begins, and `cos(θ)^2.4` tracks it closely to ~50°.
- Beyond ~50° the measured profile falls a little faster than `cos^n`
  (e.g. median 0.154 at 60.5° vs 0.157 predicted; 0.083 at 68° vs ~0.10).
- The 52.44° camera corner lies inside the rolloff, so the forward pair now samples
  real angular falloff; the probes extend coverage to ~69°.
- `cos^n` cannot represent the small flat core exactly; at 10° the fit is ~1% low.
  This is well inside the measured residual and is documented, not corrected.
- **Old n = 23.13538193587864 (60° full cone) is invalid for this profile.**

## Files

- `calibration.json` — machine-readable fit result and per-degree measured profile.
- `spot_fit.png` — measured forward/probe medians vs the fitted `cos^n` curve.

Rendering note: the evaluation supplement uses `intensity_scale = 4` for L2
(200,000 cd vs the unchanged 50,000 cd L0); with `spot_power = n` PRENDO renders
the visible FOV as a 4× illumination with the measured angular falloff.
