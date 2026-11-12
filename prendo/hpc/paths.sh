#!/bin/bash
set -euo pipefail

export PRENDO_HOME="${HOME}/prendo"
export PRENDO_CODE="${PRENDO_HOME}/code/PR-ENDO"
export EPHEMERAL=/rds/general/user/yl15425/ephemeral
export PRENDO_DATA="${EPHEMERAL}/prendo/data/VRCaps_lumen_100"
export PRENDO_RUNS="${EPHEMERAL}/prendo/runs"
export PRENDO_ARTIFACTS="${PRENDO_HOME}/artifacts"
export PRENDO_LOGS="${PRENDO_HOME}/logs"

if ! command -v module >/dev/null 2>&1; then
  set +eu
  source /etc/profile.d/modules.sh
  set -eu
fi
module use /sw-eb/modules/all
module load miniforge/3
if [ ! -x "${HOME}/miniforge3/bin/conda" ]; then
  miniforge-setup
fi
eval "$("${HOME}/miniforge3/bin/conda" shell.bash hook)"
module load CUDA/12.1.1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

mkdir -p "${PRENDO_HOME}" "${PRENDO_LOGS}" "${PRENDO_ARTIFACTS}" "${PRENDO_RUNS}"
