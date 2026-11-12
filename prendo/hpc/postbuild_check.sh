#!/bin/bash
set -euo pipefail

source "$HOME/prendo/code/PR-ENDO/hpc/env.sh"

bash "$PRENDO_CODE/hpc/prefetch_weights.sh"
python "$PRENDO_CODE/tools/validate_vrcaps_interface.py" --self-test
python "$PRENDO_CODE/tools/validate_vrcaps_interface.py" \
  --data "$PRENDO_DATA" \
  --repo "$PRENDO_CODE" \
  --cpu-stubs \
  --report "$PRENDO_LOGS/interface_cpu_login.json"
# The authoritative run without stubs happens inside hpc/smoke.pbs on a GPU node.

