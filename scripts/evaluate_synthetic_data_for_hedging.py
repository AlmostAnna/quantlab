"""
Evaluate synthetic time series data quality for deep hedging applications.

This script evaluates how well models trained on synthetic data perform
when evaluated on market-calibrated synthetic data
using quantlab's deep hedging infrastructure.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quantlab.calibration.market_calibrator import generate_market_calibrated_paths
from quantlab.ml.metrics.pnl import compute_pnl_with_tx, decompose_pnl
from quantlab.ml.models.hedge_net import HedgeNet


def load_synthetic_paths(file_path):
    """
    Load synthetic paths from various formats.

    Supports: .pt (PyTorch tensors), .npz (NumPy arrays),
            .csv (with time series columns)
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pt":
        return torch.load(file_path)
    elif ext == ".npz":
        data = np.load(file_path)
        return (
            torch.tensor(data["paths"])
            if "paths" in data.files
            else torch.tensor(data[list(data.files)[0]])
        )
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return torch.tensor(df.values)  # Assume each row is a path
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def prepare_hedging_inputs_from_paths(S_paths, K, T, M, device="cpu"):
    """
    Convert asset paths to the input format expected by HedgeNet.

    Args:
        S_paths: Tensor of shape (N, M+1) representing N paths of M+1 time steps
        K: Strike price
        T: Time to maturity
        M: Number of rebalancing periods
        device: Device to put tensors on

    Returns:
        tau_flat, moneyness_flat: Flattened inputs for HedgeNet
        S_paths: Original paths for P&L calculation
    """
    # Create time grid and calculate time to maturity
    t_grid = torch.linspace(0, T - T / M, M, device=device)
    tau_grid = T - t_grid  # Time to maturity grid

    N = S_paths.size(0)  # Number of paths

    # Expand tau for each path
    tau_batch = tau_grid.unsqueeze(0).expand(N, -1)  # Shape: (N, M)

    # Calculate moneyness (S_t / K) for all time steps except the last one
    moneyness_batch = S_paths[:, :-1] / K  # Shape: (N, M) - exclude terminal time

    # Flatten for model input
    tau_flat = tau_batch.reshape(-1)  # Shape: (N*M,)
    moneyness_flat = moneyness_batch.reshape(-1)  # Shape: (N*M,)

    return tau_flat, moneyness_flat, S_paths


def train_model_on_data(paths, config, device="cpu"):
    """
    Train HedgeNet on given paths using quantlab's infrastructure.

    Args:
        paths: Tensor of shape (N, M+1) containing asset paths
        config: Training configuration dictionary
        device: Device to train on

    Returns:
        Trained HedgeNet model
    """
    # Initialize model and optimizer
    model = HedgeNet(hidden_dim=config.get("hidden_dim", 64)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0),
    )

    # Prepare training data
    K = config["strike_price"]
    T = config["time_to_maturity"]
    M = config["num_rebalancing_periods"]
    lambda_tx = config.get("transaction_cost", 0.01)

    tau_flat, moneyness_flat, S_batch = prepare_hedging_inputs_from_paths(
        paths, K, T, M, device
    )

    # Training loop
    model.train()
    for epoch in range(config.get("epochs", 100)):
        optimizer.zero_grad()

        # Forward pass
        phi_flat = model(tau_flat, moneyness_flat)
        phi_batch = phi_flat.reshape(S_batch.size(0), -1)  # Reshape to (N, M)

        # Calculate P&L
        pnl = compute_pnl_with_tx(S_batch, K, phi_batch, lambda_tx)
        loss = torch.mean(pnl**2)  # Minimize squared P&L (hedging error)

        # Backward pass
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model


def evaluate_model_performance(model, test_paths, config, device="cpu"):
    """
    Evaluate trained model performance using quantlab metrics.

    Args:
        model: Trained HedgeNet model
        test_paths: Tensor of test paths (N, M+1)
        config: Evaluation configuration
        device: Device to run evaluation on

    Returns:
        Dictionary of performance metrics
    """
    model.eval()
    K = config["strike_price"]
    T = config["time_to_maturity"]
    M = config["num_rebalancing_periods"]
    lambda_tx = config.get("transaction_cost", 0.01)

    with torch.no_grad():
        tau_flat, moneyness_flat, S_batch = prepare_hedging_inputs_from_paths(
            test_paths, K, T, M, device
        )

        # Get hedge ratios
        phi_flat = model(tau_flat, moneyness_flat)
        phi_batch = phi_flat.reshape(S_batch.size(0), -1)  # (N, M)

        # Calculate P&L and decomposition
        pnl = compute_pnl_with_tx(S_batch, K, phi_batch, lambda_tx)
        total_pnl, hedging_gain, tx_cost = decompose_pnl(
            S_batch, K, phi_batch, lambda_tx
        )

        # Calculate metrics
        metrics = {
            "mean_abs_pnl": torch.mean(torch.abs(pnl)).item(),
            "std_pnl": torch.std(pnl).item(),
            "mean_pnl": torch.mean(pnl).item(),
            "max_pnl": torch.max(pnl).item(),
            "min_pnl": torch.min(pnl).item(),
            "pnl_sharpe_ratio": torch.mean(pnl).item() / (torch.std(pnl).item() + 1e-8),
            "mean_hedging_error": torch.mean(torch.abs(pnl)).item(),
            "total_transaction_cost": torch.mean(tx_cost).item(),
            "total_hedging_gain": torch.mean(hedging_gain).item(),
            "number_of_paths": S_batch.size(0),
            "percentile_5_pnl": torch.quantile(pnl, 0.05).item(),
            "percentile_95_pnl": torch.quantile(pnl, 0.95).item(),
        }

    return metrics


def main():
    """Evaluate Deep Hedging Model on a given dataset."""
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic data quality for deep hedging using market-calibrated reference."  # noqa: E501
    )
    parser.add_argument(
        "--market_reference",
        type=str,
        default="SPY",
        help="Market ticker to calibrate reference model (e.g., SPY, QQQ)",
    )
    parser.add_argument(
        "--synthetic_data_paths",
        nargs="+",
        required=True,
        help="Paths to synthetic data files from different generative models",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to JSON config for hedging model and evaluation parameters",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (cpu or cuda)",
    )
    parser.add_argument(
        "--n_test_paths",
        type=int,
        default=5000,
        help="Number of market-calibrated test paths to generate",
    )

    args = parser.parse_args()

    # Setup logging and results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"synthetic_data_evaluation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(results_dir / "evaluation.log"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Starting synthetic data evaluation at {timestamp}")
    logger.info(f"Market reference: {args.market_reference}")

    # Load configuration
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Generate market-calibrated test set for models' evaluation
    logger.info(
        f"Generating market-calibrated test set from {args.market_reference}..."
    )

    # Use the calibrated generator to create realistic test paths
    real_test_paths = generate_market_calibrated_paths(
        ticker=args.market_reference,
        n_paths=args.n_test_paths,
        maturity=config["time_to_maturity"],
        n_steps=config["num_rebalancing_periods"],
    )

    logger.info(f"Market-calibrated test set shape: {real_test_paths.shape}")
    logger.info(
        f"Test set price range: {real_test_paths.min().item():.2f} - {real_test_paths.max().item():.2f}"  # noqa: E501
    )

    # --- Baseline Evaluation (Train & Eval on Market-Calibrated Data) ---
    logger.info("Training baseline model on market-calibrated data...")

    # For baseline, use a subset of market-calibrated data for training
    real_train_size = min(
        int(real_test_paths.size(0) * 0.6), 3000
    )  # Use 60% for training
    real_train_paths = real_test_paths[:real_train_size]

    baseline_model = train_model_on_data(real_train_paths, config, args.device)

    logger.info("Evaluating baseline model on market-calibrated test set...")
    baseline_metrics = evaluate_model_performance(
        baseline_model, real_test_paths, config, args.device
    )

    # Save baseline results
    baseline_results_path = results_dir / "baseline_metrics.json"
    with open(baseline_results_path, "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    logger.info(f"Baseline mean absolute P&L: {baseline_metrics['mean_abs_pnl']:.6f}")

    # --- Evaluation for Each Synthetic Dataset ---
    results_summary = {}
    results_summary["baseline_market_calibrated"] = {
        "metrics": baseline_metrics,
        "trained_on": f"market_calibrated_{args.market_reference}",
        "evaluated_on": f"market_calibrated_{args.market_reference}_test",
        "calibration_ticker": args.market_reference,
    }

    for synth_path in args.synthetic_data_paths:
        synth_name = Path(synth_path).stem  # Extract name without extension
        logger.info(f"Evaluating synthetic data: {synth_name}")

        # Load synthetic training data
        synth_train_paths = load_synthetic_paths(synth_path)
        logger.info(f"Synthetic training set shape: {synth_train_paths.shape}")

        # Train model on synthetic data
        logger.info(f"Training model on {synth_name} synthetic data...")
        synth_model = train_model_on_data(synth_train_paths, config, args.device)

        # Evaluate on market-calibrated test set (common test set)
        logger.info(f"Evaluating {synth_name} model on market-calibrated test set...")
        synth_on_real_metrics = evaluate_model_performance(
            synth_model, real_test_paths, config, args.device
        )

        # Evaluate on synthetic test set (for comparison)
        synth_test_size = min(1000, synth_train_paths.size(0))  # Use subset for test
        synth_test_paths = synth_train_paths[-synth_test_size:]

        logger.info(f"Evaluating {synth_name} model on synthetic test set...")
        synth_on_synth_metrics = evaluate_model_performance(
            synth_model, synth_test_paths, config, args.device
        )

        # Calculate degradation compared to baseline
        degradation_metrics = {}
        for key, baseline_value in baseline_metrics.items():
            if key in synth_on_real_metrics and isinstance(
                baseline_value, (int, float)
            ):
                synth_value = synth_on_real_metrics[key]
                if baseline_value != 0:
                    # Calculate how much worse the synthetic-trained model
                    # performs on real data
                    degradation = (
                        abs(synth_value) / abs(baseline_value)
                        if baseline_value != 0
                        else float("inf")
                    )
                    degradation_metrics[f"{key}_degradation_vs_baseline"] = degradation
                else:
                    degradation_metrics[f"{key}_degradation_vs_baseline"] = (
                        float("inf") if synth_value != 0 else 1.0
                    )

        # Store results
        results_summary[synth_name] = {
            "metrics_on_market_calibrated_test": synth_on_real_metrics,
            "metrics_on_synth_test": synth_on_synth_metrics,
            "degradation_vs_baseline": degradation_metrics,
            "trained_on": synth_name,
            "evaluated_on": ["market_calibrated_test_set", "synthetic_test_set"],
            "calibration_ticker": args.market_reference,
        }

        logger.info(
            f"{synth_name} mean abs P&L on market-calibrated test: {synth_on_real_metrics['mean_abs_pnl']:.6f}"  # noqa: E501
        )

    # --- Generate Summary Report ---
    summary_data = []
    for model_name, results in results_summary.items():
        if "baseline" in model_name:
            summary_row = {
                "model_type": model_name,
                "trained_on": "market_calibrated",
                "mean_abs_pnl_market_test": results["metrics"]["mean_abs_pnl"],
                "std_pnl_market_test": results["metrics"]["std_pnl"],
                "sharpe_ratio_market_test": results["metrics"].get(
                    "pnl_sharpe_ratio", 0
                ),
            }
        else:
            summary_row = {
                "model_type": model_name,
                "trained_on": "synthetic",
                "mean_abs_pnl_market_test": results[
                    "metrics_on_market_calibrated_test"
                ]["mean_abs_pnl"],
                "std_pnl_market_test": results["metrics_on_market_calibrated_test"][
                    "std_pnl"
                ],
                "sharpe_ratio_market_test": results[
                    "metrics_on_market_calibrated_test"
                ].get("pnl_sharpe_ratio", 0),
                "degradation_factor": results["degradation_vs_baseline"].get(
                    "mean_abs_pnl_degradation_vs_baseline", 1.0
                ),
            }
        summary_data.append(summary_row)

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = results_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    logger.info("\nSummary Results:")
    logger.info(f"\n{summary_df.to_string()}")

    # Save complete results
    full_results_path = results_dir / "complete_results.json"
    with open(full_results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nFull results saved to: {full_results_path}")
    logger.info(f"Summary CSV saved to: {summary_csv_path}")
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
