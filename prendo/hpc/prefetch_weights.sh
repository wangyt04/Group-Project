#!/bin/bash
set -euo pipefail

source "$HOME/prendo/code/PR-ENDO/hpc/env.sh"

python -m pip install "numpy<2" tensorboard

python - <<'PY'
import OpenEXR
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models import VGG16_Weights
import lpips

print("OpenEXR ok")
print("SummaryWriter ok")
torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
lpips.LPIPS(net="vgg")
print("weights prefetched")
PY
