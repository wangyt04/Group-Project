# VRCaps PR-ENDO training runbook

## Paths

- code: `$HOME/prendo/code/PR-ENDO`
- env: `$HOME/prendo/envs/prendo` (created by `hpc/env_build.pbs`)
- data: `/rds/general/user/yl15425/ephemeral/prendo/data/VRCaps_lumen_100_v2` (`hpc/train.env`)
- run: `/rds/general/user/yl15425/ephemeral/prendo/runs/vrcaps_lumen_100_v2_r30k` (`hpc/train.env`)
- `hpc/paths.sh` still carries the V1 defaults (`VRCaps_lumen_100` / `vrcaps_lumen_100_r30k`); `hpc/train.env` overrides them for the v2 runs.
- artifacts: `$HOME/prendo/artifacts`
- logs: `$HOME/prendo/logs`

## HPC facts

- Login: `ssh -M -S ~/.ssh/hpc.sock -o ControlPersist=yes yl15425@login.cx3.hpc.imperial.ac.uk`
- PBS binaries are not on the non-interactive PATH; use `/opt/pbs/bin/qsub`, `/opt/pbs/bin/qstat`.
- GPU queue: `v1_gpu72`; the job requests `ngpus=1:gpu_type=L40S`.
- `$EPHEMERAL` is not exported on login nodes; the scripts set `/rds/general/user/yl15425/ephemeral`.
- The reader patch must be applied with `git apply -p0` because the diff paths start with `scene/`.

## Ordered commands

0. After the environment build: `bash $HOME/prendo/code/PR-ENDO/hpc/postbuild_check.sh` (prefetch + self-test + stubbed reader validation; the authoritative reader validation and GPU smoke run inside `hpc/smoke.pbs`)
1. `./hpc/sync_to_hpc.sh`
2. `ssh ... 'cd $HOME/prendo/code/PR-ENDO && /opt/pbs/bin/qsub hpc/env_build.pbs'`
3. `ssh ... 'cd $HOME/prendo/code/PR-ENDO && /opt/pbs/bin/qsub hpc/train.pbs'` — this 24h job runs the GPU smoke first, chooses resolution from peak VRAM, trains 30k, then runs `render.py`, `render_light_rotations.py`, and `metrics.py`. No second GPU queue wait. `hpc/render.pbs` remains available as a standalone re-render job.
4. `ssh ... 'bash $HOME/prendo/code/PR-ENDO/hpc/package_bundle.sh'`

## Resume

If training is interrupted, resubmit `hpc/train.pbs` with:

```bash
--start_checkpoint "$RUN_DIR/chkpnt<last_saved_iteration>.pth"
```

Do not change hyperparameters when resuming. Record the interruption and resumed iteration here.

## Known environment fixes

- Pin `numpy<2`: PyTorch 2.1.0 wheels are not compatible with NumPy 2.x.
- Pin `setuptools==75.8.0`: setuptools 84 removed `pkg_resources`, which `torch.utils.cpp_extension` still imports.
- Install the CUDA extensions with `pip install --no-build-isolation`; otherwise pip's isolated build environment cannot see the installed torch.
- Install `tensorboard` even though the official README omits it; `utils/train_utils.py` imports `SummaryWriter`.
- Verify `import OpenEXR` after the NumPy downgrade; `scene/dataset_readers.py` imports it at module level even though the dataset uses TIFF depth.

## Failure signatures

- `LOSS NAN`: keep the last checkpoint and log, inspect depth percentiles and loss terms, do not silently change hyperparameters.
- CUDA OOM: set `RESOLUTION=2` in `hpc/train.env`, record the measured peak VRAM, resubmit.
- Missing `extensions ok`: rebuild the env and inspect the compiler/CUDA module versions.
- `Network is unreachable`: reconnect Imperial VPN; the HPC hostname resolves only over IPv6.

## Acceptance record

- [x] interface CPU validation passed
- [x] GPU smoke passed; peak VRAM recorded
- [x] 30k training completed without `LOSS NAN`
- [x] render + light rotation videos produced
- [x] `run_manifest.json` written
- [x] bundle downloaded and sha256 verified

## Run record: 2026-11-11/12

- PBS job: `4616153.pbs-7`, host `cx3-20-2`
- Started 23:31:51 GMT; training walltime 1133 s (~18.9 min)
- Resolution 1 (1350x1080); 30,000 iterations; 171,022 Gaussians
- Test metrics at 30k: PSNR 31.43, SSIM 0.98, LPIPS 0.10, L1 0.01, MSSIM 0.97
- Peak VRAM: smoke 4.79 GB; training 9.14 GB (from `gpu_mem_samples.txt`)
- Bundle: `vrcaps_lumen_100_r30k_bundle.tar.gz`, 378 MB
- Bundle sha256: `b03976078c0b6ce140a7efb28adf77570044f970a21e70972cb06131cf247a96`
- Manifest: 80 outputs; all 80 hashes verified after regeneration
- Local extracted path: `artifacts/vrcaps_lumen_100_r30k/`
- Note: the first manifest was generated while `train.log` was still being appended, so its `train.log` hash was stale. The bundle was regenerated after the job ended and all 80 hashes now match.

## V2 point-light run record: 2026-11-12

- Dataset: `VRCaps_lumen_100_v2` (point light, 88 train / 12 test)
- Training job: `4617178.pbs-7`; 30,000 iterations; resolution 1; 213,091 Gaussians
- Training walltime: 1,131 s; peak VRAM 9.18 GB
- Test metrics at 30k: PSNR 33.64, SSIM 0.99, LPIPS 0.06
- Training bundle: `vrcaps_lumen_100_v2_r30k_bundle.tar.gz`, 337 MB
- Training bundle sha256: `bb314d69ed15385615bf1fbe232bae679010bcf98cb75dd7cd8e5b5a2bfaeb93`
- Manifest: 80 outputs; all 80 hashes verified
- Local extracted path: `artifacts/vrcaps_lumen_100_v2_r30k/`

## Relight B1 run record: 2026-11-12

- B1: `VRCaps_lumen_100_v2_relight_config` (25 configs, 49 cameras, 744 entries, 671 valid, 73 invalid L3)
- Relight jobs: `4617744.pbs-7` (391 non-L2 entries) + `4626486.pbs-7` (280 L2-only entries); run id `20261112-0332`
- Rendered: 671 valid `(config, frame)` pairs (391 non-L2 + 280 L2; PNG + EXR)
- Training-frame metrics used in `record.json`: PSNR 34.787, SSIM 0.9886, LPIPS 0.0580 (88 views)
- Output bundle: `pred_VRCaps_lumen_100_v2_20261112-0332.tar.gz`, 2.6 GB
- Output bundle sha256: `668b8622158b9b6abdfad09950914e6ced0df42efb74a08da5f49268165e13e4`
- Local extracted path: `artifacts/pred_VRCaps_lumen_100_v2_20261112-0332/`
- PNG: display-space RGB8; EXR: linear RGB float16. Smoke check max PNG-vs-EXR difference 3.9e-3 (8-bit/half quantization).
- L2 completed with `n=23.13538193587864` (fit RMSE 0.019458); `record.json` has `spot_power=23.13538193587864` and total 671. Calibration artifacts: `prendo/runs/20261112-0332/calibration/`.
- L2-only command:

```bash
qsub -l walltime=04:00:00 \
  -v RELIGHT_RUN_ID=20261112-0332,LEVELS=L2,SPOT_POWER_N=23.13538193587864,SKIP_TRAIN_METRICS=1,CODE_COMMIT=https://github.com/wangyt04/Group-Project@a13e272206e0477fc380b5500363f96d2995a33e \
  hpc/relight.pbs
```

- Handoff description: `handoff/PRENDO_relight_handoff.md` (repo root)

### Relight implementation notes

- `tools/render_relight.py` maps B1 frames to the V2 training cameras by `image_name`.
- Point configs pass `light_rot=None`; L2 builds a rotation whose third column is `light_dir_world`.
- Color scale is applied outside `compute_lighted_rgb` in linear space.
- EXR writing requires `np.ascontiguousarray` on each R/G/B slice; otherwise the OpenEXR binding mixes channels.
- `shadow`/`noshadow` variants are merged: one image per `(config, frame)`.
- `render_manifest.json` records `Path.resolve()` output; `/rds/general/ephemeral/user/yl15425/ephemeral` is the canonical target of the `/rds/general/user/yl15425/ephemeral` symlink, so both spellings name the same area.
