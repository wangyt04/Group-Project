# VRCaps_lumen_100_v2 relight configuration (B1 only)

This small package supplies lighting metadata for the completed v2 training
baseline. It contains no RGB, EXR reference images or reference depths. The
existing `VRCaps_lumen_100_v2` directory and its training ZIP are unchanged.

## Use and contents

Read `relight/lights.json`, `relight/transforms_relight.json` and
`relight/export_settings.json` together. The separate `baseline_light_settings.json`
is copied from the actual v2 export. The top-level `transforms.json` is an
unchanged copy of the 88 training cameras for metadata-reader compatibility;
training images/depths/points are supplied by the separate training ZIP.

The team's `relight_eval.relight_io.Dataset` loads this package's metadata.
Image evaluation cannot run until separately authorised B2 references exist.
To assemble a future combined dataset, stage the separate v2 baseline elsewhere
and add the `relight/` directory there; do not replace the delivered baseline.
Keep configuration IDs and four-digit frame IDs unchanged. `file_path` inside
camera JSON retains its original unpadded value, as the actual reader expects.

There are 25 configurations: L0 (1), L1 intensity (2), L2 beam direction (9),
L3 position (8), L4 colour (3), and L5 world-fixed sequence lights (2).
Each light record contains actual derived RH world position and unit direction.
Positions and cameras use the same world system as v2: (-Unity.X,Unity.Y,Unity.Z).
OpenGL camera axes are right/up/backward; forward is minus the third column.
Point-light directions are the current camera forward direction, including L5;
their orientation does not affect point illumination. L5 positions stay at the
first selected camera centre of their own segment.

## Frame selection and roles

`selection.json` records the deterministic lists and rules. Main frames are 30
evenly spaced ranks from the actual 88 training views, with round-half-up rank
selection. Sequence segments use the earliest and latest eligible 20-original-frame
windows and sample offsets 0,2,...,18, ten frames each. Every sampled frame must
belong to the 88 training cameras. Five calibration views are evenly spaced
among remaining training views after excluding both main and sequence views.
No held-out camera is selected, and no training split was changed.

Relight `split=test` means a main/sequence lighting-evaluation role at a known
training camera, not membership in the baseline's 12 held-out camera views.
Calibration views have relight `split=calib` and cannot overlap evaluation views.
Main/sequence overlap is retained as group membership, but duplicate L0
(configuration,frame) requests and camera records are merged. Consequently,
the handoff's nominal 1,015 future variant-image requests reduce by those L0
duplicates, then by invalid geometry entries. Actual planned/eligible counts
are in `CHECKS.json`. Neither count represents generated images.

## Actual baseline versus planned references

Actual v2 baseline: one camera-centred white Point, 50,000 candela, fixed EV100
20, range 6.71999979019165 world units, zero shape radius, inverse-square
attenuation with smooth range attenuation disabled. The reference depth statistic
is d_ref=0.33595186471939087 world units. Actual shadows, sky/environment,
indirect illumination, probes/SSR/SSAO and appearance postprocessing are off.
Training RGB8 PNG encoding is clamp(exposed linear RGB,0,1)^(1/2.2), rounded to
8-bit. HDRP's fixed pre-exposure multiplier is already applied once.

Configs describe future changes relative to that baseline. L1 scales candela
intensity; L4 scales linear light RGB. L3 offsets are multiples of d_ref along
camera right/up/forward. L2 changes Point to Spot at the camera centre, with
the same baseline candela on-axis setting and no cookie. B2 would store exposed
linear float RGB EXRs before PNG gamma, using the same fixed exposure once.
No B2 image linearity, colour, saturation or shadow comparison has been run.

`configs[].variants` lists **planned** future B2 variants. L3/L5 request both
`noshadow` and `shadow`; other levels request `noshadow` only. Shadow references
would require hard shadows and the highest supported resolution. Effective
shadow resolution and EXR depth channel are not claimed as measured settings;
no shadow reference/depth output has been generated. Export settings explicitly
separate the actual baseline from this proposed future work.

## Spot angles and pending calibration

`spot_inner_deg` and `spot_outer_deg` are HALF angles. The provisional handoff
default is outer half-angle 30 degrees, corresponding to Unity full spot angle
60 degrees. HDRP 7.3.1 permits `innerSpotPercent=0`: its setter clamps to [0,100]
and its light loop computes inner half-angle from that percentage. Therefore
effective inner half-angle is 0 degrees (full inner angle 0); HDRP does not use
legacy `Light.innerSpotAngle`. The source API evidence is recorded in provenance.
This is a metadata configuration derived from the installed HDRP implementation;
no spotlight scene was rendered or claimed calibrated in this task.

All nine L2 configs share these angles. The 60-degree full cone is narrower
than the actual projection FOV recorded in export settings. The angle choice
is provisional pending Yutong's FOV/cone review. The fitted spotlight exponent
`n` remains null pending Yutong's reference calibration; it was not guessed.
Point configurations do not require this spotlight exponent.

## Genuine L3 validity checks

The generator uses the local transformed `GI_2_high_connected.fbx` colon mesh
(178,070 vertices, 346,183 triangles). Its actual world transform, mesh asset,
counts and material identity match the v2 capture metadata. Exported world
vertices are reflected in X to match the saved RH camera world.

Each L3 candidate is checked for exact nearest-triangle clearance of at least
0.05*d_ref and a two-sided camera-to-light segment/surface intersection. L5
uses the same conservative criteria as an additional recorded check. AABBs
only prune triangle candidates; bounding-box containment is never used as
validity. The preserved camera trajectory is the lumen witness: a candidate
must have clearance and an unobstructed segment from that known lumen camera.
This does not assert a globally watertight solid classification for the open
anatomical mesh. Per-entry distances, nearest triangle, obstruction results,
invalid reasons and final validity are saved in `geometry_checks.json`.
Invalid entries stay in `lights.json` with `valid=false`; do not render or
evaluate them as valid samples. Other configs retain the validated camera centre.

## Provenance and remaining work

`build_relight_v2_config.py` and `generation_provenance.json` document local
sources and deterministic generation. `CHECKS.json` records minimal practical
schema/count/selection/transform/unit-direction checks and the team's actual
metadata-loader check. Previously passed baseline reconstruction/geometry tests
were reused, not repeated. ZIP checks are limited to count and size.

Physical/anatomical scale remains unverified (`mm_per_unit=null`). No Unity
rendering invocation, B2 reference batch, model training, HPC modification or
GitHub image upload occurred. Remaining work is Yutong's cone confirmation, separately authorised calibration
references for fitting the spotlight exponent, and the remaining B2 references
and image checks. Harry verifies OneDrive sync and creates the sharing link on the
website; cloud sync and recipient receipt are not established by local packaging.
