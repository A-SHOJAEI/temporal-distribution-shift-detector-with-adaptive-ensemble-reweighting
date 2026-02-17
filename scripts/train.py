#!/usr/bin/env python3
"""
Training script for the Temporal Distribution Shift Detector.

This script provides both batch and streaming training modes with comprehensive
logging, checkpointing, and experiment tracking via MLflow.
"""

import argparse
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.utils.config import (
    load_config,
    setup_environment,
    Config,
)
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.loader import (
    DataLoader,
    DriftDataLoader,
)
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.preprocessing import (
    DriftInjector,
)
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.training.trainer import (
    EnsembleTrainer,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Temporal Distribution Shift Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["covertype", "elec2", "airlines"],
        help="Dataset to use (overrides config)",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming training mode",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to process (streaming mode only)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLflow experiment name (overrides config)",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        help="MLflow run name",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def load_data(config: Config, data_loader: DataLoader) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset based on configuration.

    Args:
        config: Configuration object.
        data_loader: Data loader instance.

    Returns:
        Tuple of (features, labels).
    """
    dataset_name = config.data.dataset_name.lower()

    logger.info(f"Loading {dataset_name} dataset...")

    if dataset_name == "covertype":
        X, y = data_loader.load_covertype()
    elif dataset_name == "elec2":
        X, y = data_loader.load_elec2(config.data.data_path)
    elif dataset_name == "airlines":
        if config.data.data_path is None:
            raise ValueError("Airlines dataset requires data_path in configuration")
        X, y = data_loader.load_airlines(config.data.data_path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y


def setup_drift_schedule(config: Config, total_batches: int) -> Dict[int, str]:
    """
    Setup drift schedule for streaming training.

    Args:
        config: Configuration object.
        total_batches: Total number of batches.

    Returns:
        Drift schedule dictionary.
    """
    if config.data.drift_schedule:
        return config.data.drift_schedule

    # Create default schedule with drift every 20 batches
    schedule = {}
    drift_types = config.data.drift_types
    drift_interval = max(10, total_batches // 10)  # At least 10 batches apart

    for i, drift_type in enumerate(drift_types):
        batch_idx = (i + 1) * drift_interval
        if batch_idx < total_batches:
            schedule[batch_idx] = drift_type

    logger.info(f"Created drift schedule: {schedule}")
    return schedule


def train_batch_mode(
    config: Config,
    X: pd.DataFrame,
    y: pd.Series,
    trainer: EnsembleTrainer,
) -> None:
    """
    Train in batch mode.

    Args:
        config: Configuration object.
        X: Features.
        y: Labels.
        trainer: Ensemble trainer.
    """
    logger.info("Starting batch training...")

    # Split data BEFORE training to prevent data leakage
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.random_seed, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    start_time = time.time()

    # Train only on training data
    model = trainer.fit(X_train, y_train)

    training_time = time.time() - start_time

    logger.info(f"Batch training completed in {training_time:.2f} seconds")

    # Get model summary
    summary = trainer.get_model_summary()
    logger.info("Model Summary:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    # Evaluate on held-out test set
    predictions, probabilities = trainer.predict(X_test)

    # Calculate test accuracy
    test_accuracy = (predictions == y_test).mean()
    logger.info(f"Test accuracy: {test_accuracy:.3f}")


def train_streaming_mode(
    config: Config,
    X: pd.DataFrame,
    y: pd.Series,
    trainer: EnsembleTrainer,
    max_samples: Optional[int] = None,
) -> None:
    """
    Train in streaming mode with drift simulation.

    Args:
        config: Configuration object.
        X: Features.
        y: Labels.
        trainer: Ensemble trainer.
        max_samples: Maximum samples to process.
    """
    logger.info("Starting streaming training...")

    # Setup drift schedule
    total_batches = len(X) // config.data.batch_size
    drift_schedule = setup_drift_schedule(config, total_batches)

    # Create drift data loader
    data_loader = DriftDataLoader(
        X=X,
        y=y,
        batch_size=config.data.batch_size,
        drift_schedule=drift_schedule,
        random_state=config.random_seed,
    )

    # Apply drift injection if configured
    if config.data.drift_schedule:
        drift_injector = DriftInjector(random_state=config.random_seed)
        # Note: drift injection is handled by DriftDataLoader internally

    start_time = time.time()

    # Train in streaming mode
    result = trainer.train_streaming(
        data_loader=data_loader,
        max_samples=max_samples,
        save_frequency=config.training.save_frequency,
    )

    training_time = time.time() - start_time

    logger.info(f"Streaming training completed in {training_time:.2f} seconds")

    # Log results
    logger.info("Training Results:")
    for key, value in result.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def main():
    """Main training function."""
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override config with command line arguments
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.experiment_name:
        config.mlflow.experiment_name = args.experiment_name
    if args.run_name:
        config.mlflow.run_name = args.run_name
    if args.seed:
        config.random_seed = args.seed
    if args.debug:
        config.log_level = "DEBUG"
    if args.streaming:
        config.training.streaming_mode = True

    # Setup environment
    setup_environment(config)

    logger.info("Starting training script")
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Training mode: {'streaming' if config.training.streaming_mode or args.streaming else 'batch'}")

    try:
        # Load data
        data_loader = DataLoader(random_state=config.random_seed)
        X, y = load_data(config, data_loader)

        # Create trainer - convert dataclass configs to plain dicts
        model_dict = asdict(config.model)
        training_dict = asdict(config.training)
        mlflow_dict = asdict(config.mlflow)

        trainer = EnsembleTrainer(
            model_config=model_dict,
            training_config=training_dict,
            mlflow_config=mlflow_dict,
            random_state=config.random_seed,
        )

        # Train model
        if config.training.streaming_mode or args.streaming:
            train_streaming_mode(config, X, y, trainer, args.max_samples)
        else:
            train_batch_mode(config, X, y, trainer)

        # Evaluate target metrics
        logger.info("Evaluating against target metrics...")

        # Get final model summary
        final_summary = trainer.get_model_summary()

        # Compare against targets
        target_metrics = {
            "prequential_accuracy_under_drift": config.evaluation.target_prequential_accuracy,
            "drift_detection_f1": config.evaluation.target_drift_detection_f1,
            "adaptation_latency_samples": config.evaluation.target_adaptation_latency,
            "regret_vs_oracle_ensemble": config.evaluation.target_regret_vs_oracle,
        }

        logger.info("Target Metrics:")
        for metric, target in target_metrics.items():
            logger.info(f"  {metric}: {target}")

        # Check if we have online learning results
        if hasattr(trainer, 'online_learner') and trainer.online_learner:
            online_summary = trainer.online_learner.get_performance_summary()
            if 'recent_accuracy' in online_summary:
                actual_accuracy = online_summary['recent_accuracy']
                logger.info(f"Achieved prequential accuracy: {actual_accuracy:.3f}")

                if actual_accuracy >= config.evaluation.target_prequential_accuracy:
                    logger.info("✓ Target prequential accuracy achieved!")
                else:
                    logger.warning("✗ Target prequential accuracy not achieved")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()