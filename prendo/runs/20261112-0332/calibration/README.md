# Spotlight exponent n calibration

- Source: B2 Unity calibration frames (L0 and L2_fwd, noshadow), frames 0001, 0032, 0050, 0067, 0097.
- Fitted with `relight_eval/fit_spot_power.py`.
- n = 23.13538193587864
- RMSE = 0.019458
- Pixels used = 1,981,045 from 5 frames
- `on_axis_ratio` = 0.850 at 9.9 deg: the view centre is dark and masked, so no true inner-cone pixels were available. This is not a 15% on-axis intensity error; the tool marks the inner-cone check as not checkable for inner half-angle 0.
- The Unity HDRP spot profile is not an exact `cos^n`; see `spot_fit.png`. L2 comparisons should be interpreted with this approximation in mind.

`record.json` now records `spot_power = 23.13538193587864` after the L2-only render (job `4626486.pbs-7`, 280 entries); the output totals 671 PNG+EXR pairs.
