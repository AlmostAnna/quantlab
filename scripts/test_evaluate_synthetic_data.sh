#!/bin/bash
#
# Test script for synthetic data evaluation pipeline
#
# This script demonstrates the evaluation pipeline using market-calibrated
# synthetic data as both training and reference data, serving as a
# self-consistency check for the evaluation framework.
#
# Usage:
#   ./scripts/test_evaluate_synthetic_data.sh
#

set -e  # Exit on any error

echo "Running synthetic data evaluation test..."

# Create necessary directories
mkdir -p scripts/test_data
mkdir -p scripts/results/test

echo "Generating market-calibrated test data..."
python -c "
from quantlab.calibration.market_calibrator import generate_market_calibrated_paths
import torch
import os

# Ensure directory exists
os.makedirs('./scripts/test_data', exist_ok=True)

print('Generating training paths...')
train_paths = generate_market_calibrated_paths('SPY', n_paths=2000, maturity=1.0, n_steps=252)
torch.save(train_paths, './scripts/test_data/spy_calibrated_train.pt')

print('Generating comparison paths...')
other_paths = generate_market_calibrated_paths('SPY', n_paths=2000, maturity=1.0, n_steps=252)
torch.save(other_paths, './scripts/test_data/spy_calibrated_other.pt')

print('Created test synthetic data files')
"

echo "Running evaluation pipeline..."
python -m scripts.evaluate_synthetic_data_for_hedging \
    --market_reference SPY \
    --synthetic_data_paths ./scripts/test_data/spy_calibrated_train.pt ./scripts/test_data/spy_calibrated_other.pt \
    --config_path ./configs/hedging_config_example.json \
    --output_dir ./scripts/results/test \
    --n_test_paths 1000

echo "Test completed successfully!"
echo "Check results in: scripts/results/test/"
