#!/bin/bash

# Ensure we're in the right directory
cd /data/sandbox/QuickDraw-pytorch

echo "Installing Brevitas and preparing data..."
pip install git+https://github.com/Xilinx/brevitas.git@dev onnx onnxscript qonnx onnxoptimizer -q
python DataUtils/prepare_data.py --download 0 --categories 10 -v 0.2

echo "Running 8-bit QAT..."
python main_qat.py --bit_width 8 --epochs 1 --export_qonnx

echo "Running 4-bit QAT..."
python main_qat.py --bit_width 4 --epochs 1 --export_qonnx

echo "Running 2-bit QAT..."
python main_qat.py --bit_width 2 --epochs 1 --export_qonnx

echo "All experiments complete."
