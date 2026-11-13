#!/bin/bash
# Stage the L2 4x (outer FULL160 / inner HALF10) supplement and submit an L2-only
# relight job. Run on the CX3 login node, from anywhere:
#
#   bash ~/prendo/code/PR-ENDO/hpc/submit_l2_4x.sh [run_id] [n] [code_commit]
#
# Prerequisite: l2_4x_o160_i10_relight_meta.tar.gz has been uploaded to
#   /rds/general/user/yl15425/home/prendo/
# The original VRCaps_lumen_100_v2 root is never modified: the staged root uses
# symlinks for the training data and a real relight/ dir with the supplement metadata.
set -euo pipefail

HOME_REMOTE=/rds/general/user/yl15425/home
EPHEMERAL=/rds/general/user/yl15425/ephemeral
SRC="${EPHEMERAL}/prendo/data/VRCaps_lumen_100_v2"
STAGE="${EPHEMERAL}/prendo/data/staged_l2_4x/VRCaps_lumen_100_v2"
META="${HOME_REMOTE}/prendo/l2_4x_o160_i10_relight_meta.tar.gz"
ART="${HOME_REMOTE}/prendo/artifacts"

RUN_ID="${1:-20261113-0110}"
N="${2:-2.4004990755616116}"
CODE_COMMIT="${3:-https://github.com/wangyt04/Group-Project@43206c84356f43075dcd2d924e2fe46d7d650d93}"
OUT="${ART}/pred_VRCaps_lumen_100_v2_${RUN_ID}"

test -f "$SRC/transforms.json"
test -f "$META"
mkdir -p "$STAGE/relight"
for x in train_views test_views_1 depth_train transforms.json transforms_test.json \
         init_point_cloud.ply prendo_config.json sidecars optional_mask_support; do
  ln -sfn "$SRC/$x" "$STAGE/$x"
done
rm -f "$STAGE/relight/lights.json" "$STAGE/relight/transforms_relight.json" "$STAGE/relight/export_settings.json"
tar -xzf "$META" -C "$STAGE"
test -f "$STAGE/relight/lights.json"

python3 - "$STAGE/relight/lights.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
valid = sum(1 for e in d["frames"] if e.get("valid"))
print("supplement:", d.get("supplement_version"))
print("configs:", len(d["configs"]), "valid entries:", valid)
assert valid == 220, f"expected 220 valid entries, got {valid}"
PY

mkdir -p "$OUT"
OLD="${ART}/pred_VRCaps_lumen_100_v2_20261112-0332"
if [ -f "$OLD/train_metrics.json" ]; then
  cp "$OLD/train_metrics.json" "$OUT/train_metrics.json"
fi

cd "${HOME_REMOTE}/prendo/code/PR-ENDO"
/opt/pbs/bin/qsub -l walltime=04:00:00 \
  -v "RELIGHT_RUN_ID=${RUN_ID},LEVELS=L2,SPOT_POWER_N=${N},SKIP_TRAIN_METRICS=1,DATA_DIR=${STAGE},CODE_COMMIT=${CODE_COMMIT}" \
  hpc/relight.pbs
