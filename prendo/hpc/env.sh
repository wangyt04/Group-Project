#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
conda activate prendo
python - <<'PY'
import torch
assert torch.__version__.startswith("2.1.0"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
if torch.cuda.is_available():
    print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
else:
    print("torch", torch.__version__, "cuda", torch.version.cuda, "no CUDA device visible")
PY
