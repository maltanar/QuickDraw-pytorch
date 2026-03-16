#!/bin/bash

# Ensure we're in the right directory
cd /data/sandbox/QuickDraw-pytorch

# Prepare data if not already done (assuming it is but just in case)
# python DataUtils/prepare_data.py --download 0 --categories 10 -v 0.2

EPOCHS=3

echo "Running Baseline (2-bit, LR 0.1)..."
python main_qat.py --bit_width 2 --epochs $EPOCHS -lr 0.1 --log ./logs/baseline

echo "Running Experiment 1 (2-bit, LR 0.01)..."
python main_qat.py --bit_width 2 --epochs $EPOCHS -lr 0.01 --log ./logs/exp1_lr01

echo "Running Experiment 2 (2-bit, Per-channel)..."
python main_qat.py --bit_width 2 --epochs $EPOCHS -lr 0.1 --per_channel --log ./logs/exp2_per_channel

echo "Running Experiment 3 (2-bit, Quant-input)..."
python main_qat.py --bit_width 2 --epochs $EPOCHS -lr 0.1 --quant_input --log ./logs/exp3_quant_input

echo "Running Experiment 4 (2-bit, Combined Optimized)..."
python main_qat.py --bit_width 2 --epochs $EPOCHS -lr 0.01 --per_channel --quant_input --log ./logs/exp4_combined

echo "All 2-bit optimization experiments complete."
