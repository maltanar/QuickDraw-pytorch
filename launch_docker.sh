#!/bin/bash

# This script launches the experiments inside the custom quickdraw-qat image.
# It assumes you have an AMD GPU and ROCm drivers installed.

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v "$(pwd):/data/sandbox/QuickDraw-pytorch" \
  -w /data/sandbox/QuickDraw-pytorch \
  quickdraw-qat \
  bash ./run_experiments.sh
