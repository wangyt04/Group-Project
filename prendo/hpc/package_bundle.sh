#!/bin/bash
set -euo pipefail

source "$HOME/prendo/code/PR-ENDO/hpc/paths.sh"
source "$PRENDO_CODE/hpc/train.env"

RUN_DIR="$PRENDO_RUNS/$RUN_NAME"
BUNDLE_DIR="$PRENDO_RUNS/bundles"
mkdir -p "$BUNDLE_DIR" "$PRENDO_ARTIFACTS"

mkdir -p "$RUN_DIR/environment"
cp "$PRENDO_LOGS/pip_freeze.txt" "$RUN_DIR/environment/" 2>/dev/null || true
cp "$PRENDO_LOGS/conda_export.txt" "$RUN_DIR/environment/" 2>/dev/null || true

python3 "$PRENDO_CODE/tools/collect_artifacts.py" \
  --run-dir "$RUN_DIR" \
  --dataset "$PRENDO_DATA" \
  --patch "$PRENDO_CODE/patches/vrcaps_reader.patch" \
  --output "$RUN_DIR/run_manifest.json"

cd "$PRENDO_RUNS"
tar -czf "$BUNDLE_DIR/${RUN_NAME}_bundle.tar.gz" "$RUN_NAME"
cd "$BUNDLE_DIR"
sha256sum "${RUN_NAME}_bundle.tar.gz" \
  | tee "${RUN_NAME}_bundle.sha256"
cp "$BUNDLE_DIR/${RUN_NAME}_bundle.tar.gz" "$PRENDO_ARTIFACTS/"
cp "$BUNDLE_DIR/${RUN_NAME}_bundle.sha256" "$PRENDO_ARTIFACTS/"
echo "bundle ready at $PRENDO_ARTIFACTS/${RUN_NAME}_bundle.tar.gz"
