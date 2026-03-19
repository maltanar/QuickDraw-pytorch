#!/bin/bash
# Requires python3 and virtualenv available
virtualenv -p python3 venv-ml
source venv-ml/bin/activate
pip install onnxruntime || echo "Failed to install onnxruntime"
pip install PyQt6 || echo "Failed to install PyQt6"
pip install matplotlib || echo "Failed to install matplotlib"
