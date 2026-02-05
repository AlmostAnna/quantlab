# Scripts Directory

This directory contains utility and example scripts for the quantlab project.

## Available Scripts

### `evaluate_synthetic_data_for_hedging.py`
Main evaluation script for assessing synthetic time series quality for deep hedging applications.

### `test_evaluate_synthetic_data.sh`
Self-consistency test script that validates the evaluation pipeline using market-calibrated synthetic data as both training and reference data.

## Usage Examples

### Test the evaluation pipeline:
```bash
./scripts/test_evaluate_synthetic_data.sh
```

### Evaluate your synthetic data:
```bash
python -m scripts.evaluate_synthetic_data_for_hedging \
    --market_reference SPY \
    --synthetic_data_paths /path/to/your/data.pt \
    --config_path ./configs/hedging_config_example.json \
    --output_dir ./results
```
