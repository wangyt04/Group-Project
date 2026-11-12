#!/bin/bash
set -euo pipefail

HPC_USER=yl15425
HPC_HOST=login.cx3.hpc.imperial.ac.uk
SOCK="${HOME}/.ssh/hpc.sock"
REPO="${HOME}/Documents/my_repo/PR-ENDO"
DATA_STAGE="${HOME}/Documents/my_repo/PR-ENDO-data/VRCaps_lumen_100"
HOME_REMOTE=/rds/general/user/yl15425/home
EPHEMERAL=/rds/general/user/yl15425/ephemeral

remote() {
  ssh -S "${SOCK}" "${HPC_USER}@${HPC_HOST}" "$@"
}

remote "mkdir -p ${HOME_REMOTE}/prendo/code ${HOME_REMOTE}/prendo/logs ${HOME_REMOTE}/prendo/artifacts ${EPHEMERAL}/prendo/data ${EPHEMERAL}/prendo/runs"

rsync -az -e "ssh -S ${SOCK}" \
  --exclude 'FOR_TEAM_VRCaps.zip' \
  --exclude '*.pyc' \
  --exclude '__pycache__' \
  "${REPO}/" "${HPC_USER}@${HPC_HOST}:${HOME_REMOTE}/prendo/code/PR-ENDO/"

rsync -az -e "ssh -S ${SOCK}" \
  "${DATA_STAGE}/" "${HPC_USER}@${HPC_HOST}:${EPHEMERAL}/prendo/data/VRCaps_lumen_100/"

remote "set -e; \
  test \$(ls ${EPHEMERAL}/prendo/data/VRCaps_lumen_100/train_views/*.png | wc -l) -eq 88; \
  test \$(ls ${EPHEMERAL}/prendo/data/VRCaps_lumen_100/depth_train/*.tiff | wc -l) -eq 88; \
  test \$(ls ${EPHEMERAL}/prendo/data/VRCaps_lumen_100/test_views_1/*.png | wc -l) -eq 12; \
  echo data-ok"

remote "set -e; cd ${HOME_REMOTE}/prendo/code/PR-ENDO; \
  if ! grep -q readVRCapsSceneInfo scene/dataset_readers.py; then \
    git apply -p0 --check patches/vrcaps_reader.patch; \
    git apply -p0 patches/vrcaps_reader.patch; \
  fi; \
  grep -q readVRCapsSceneInfo scene/dataset_readers.py; \
  grep -q '\"vrcaps\"' scene/__init__.py; \
  echo patch-ok"
