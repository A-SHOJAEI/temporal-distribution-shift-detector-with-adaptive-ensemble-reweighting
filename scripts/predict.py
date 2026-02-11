#!/usr/bin/env python3
"""
Prediction script for the Temporal Distribution Shift Detector.

Loads a trained model and performs inference on new data.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.utils.config import (
    load_config,
    setup_environment,
)
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.loader import DataLoader

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run predictions with trained Temporal Distribution Shift Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/final_model.pkl",
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to input CSV file (if None, uses sample from dataset)"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/predictions.csv",
        help="Path to save predictions"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to predict (if using sample data)"
    )

    parser.add_argument(
        "--show-weights",
        action="store_true",
        help="Display ensemble weights"
    )

    return parser.parse_args()


def load_model(model_path: str):
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = joblib.load(model_path)
    logger.info("Model loaded successfully")

    return model


def load_input_data(input_path: Optional[str], n_samples: int, config):
    """Load input data for prediction."""
    if input_path is not None:
        logger.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)
        X = data.values
        logger.info(f"Loaded {len(X)} samples from file")
    else:
        logger.info(f"Loading {n_samples} sample data from dataset")
        loader = DataLoader()

        # Use the configured dataset
        dataset_name = config.data.dataset_name
        if dataset_name == "covertype":
            X, y = loader.load_covertype()
        elif dataset_name == "elec2":
            X, y = loader.load_elec2()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Take a random sample
        indices = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
        X = X[indices]
        logger.info(f"Loaded {len(X)} samples from {dataset_name} dataset")

    return X


def save_predictions(predictions: np.ndarray, output_path: str):
    """Save predictions to CSV file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with predictions
    df = pd.DataFrame({
        'prediction': predictions,
        'sample_id': range(len(predictions))
    })

    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


def main():
    """Main prediction workflow."""
    args = parse_arguments()

    # Setup environment
    config = load_config(args.config)
    setup_environment(config)

    logger.info("Starting prediction script")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Input path: {args.input_path or 'sample data'}")
    logger.info(f"Output path: {args.output_path}")

    try:
        # Load trained model
        model = load_model(args.model_path)

        # Load input data
        X = load_input_data(args.input_path, args.n_samples, config)

        # Make predictions
        logger.info("Running predictions...")
        predictions = model.predict(X)

        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            logger.info(f"Prediction confidence (mean): {probabilities.max(axis=1).mean():.3f}")

        # Display ensemble weights if requested
        if args.show_weights and hasattr(model, 'reweighter'):
            weights = model.get_ensemble_weights()
            logger.info("\nCurrent Ensemble Weights:")
            for learner_name, weight in weights.items():
                logger.info(f"  {learner_name}: {weight:.3f}")

        # Save predictions
        save_predictions(predictions, args.output_path)

        # Display summary statistics
        unique, counts = np.unique(predictions, return_counts=True)
        logger.info("\nPrediction Distribution:")
        for class_label, count in zip(unique, counts):
            logger.info(f"  Class {class_label}: {count} ({count/len(predictions)*100:.1f}%)")

        logger.info("\nPrediction completed successfully")

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
