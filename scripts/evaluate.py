#!/usr/bin/env python3
"""
Evaluation script for the Temporal Distribution Shift Detector.

This script provides comprehensive evaluation including drift detection performance,
regret analysis, and comparison against oracle performance.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.models.model import (
    AdaptiveEnsembleDetector,
)
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.training.trainer import (
    EnsembleTrainer,
)
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.evaluation.metrics import (
    DriftMetrics,
    PrequentialEvaluator,
    RegretAnalyzer,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Temporal Distribution Shift Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["covertype", "elec2", "airlines"],
        help="Dataset to use (overrides config)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (overrides config)",
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=5,
        help="Number of evaluation trials for statistical significance",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to evaluate",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to files",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def create_oracle_ensemble(X: pd.DataFrame, y: pd.Series, config: Config) -> Dict:
    """
    Create oracle ensemble for regret analysis.

    Args:
        X: Features.
        y: Labels.
        config: Configuration.

    Returns:
        Dictionary with oracle models and their performance.
    """
    from sklearn.model_selection import cross_val_score
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb

    oracle_models = {}

    # Create base models with optimal parameters (simplified)
    models = {
        "xgboost": xgb.XGBClassifier(**config.model.base_models.xgboost),
        "lightgbm": lgb.LGBMClassifier(**config.model.base_models.lightgbm),
        "catboost": cb.CatBoostClassifier(**config.model.base_models.catboost),
    }

    logger.info("Creating oracle ensemble...")

    for name, model in models.items():
        logger.info(f"Training oracle {name}...")

        try:
            # Train model
            model.fit(X, y)

            # Cross-validation score as oracle performance
            cv_scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
            oracle_models[name] = {
                "model": model,
                "cv_score": cv_scores.mean(),
                "cv_std": cv_scores.std(),
            }

            logger.info(f"Oracle {name} CV score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        except Exception as e:
            logger.warning(f"Oracle {name} failed (sklearn compatibility): {e}. Skipping.")
            continue

    return oracle_models


def evaluate_drift_detection(
    model: AdaptiveEnsembleDetector,
    data_loader: DriftDataLoader,
    config: Config,
) -> Dict:
    """
    Evaluate drift detection performance.

    Args:
        model: Trained adaptive ensemble detector.
        data_loader: Data loader with drift schedule.
        config: Configuration.

    Returns:
        Dictionary with drift detection results.
    """
    logger.info("Evaluating drift detection performance...")

    drift_metrics = DriftMetrics(tolerance_window=config.evaluation.drift_tolerance_window)

    # Get ground truth drift points
    drift_schedule = data_loader.get_drift_schedule()
    for batch_idx, drift_type in drift_schedule.items():
        if drift_type != "none":
            # Convert batch index to sample index (approximate)
            sample_idx = batch_idx * data_loader.batch_size
            drift_metrics.add_true_drift(sample_idx)

    # Process data and collect drift detections
    sample_count = 0
    detected_drifts = []

    for batch_X, batch_y, metadata in data_loader:
        # Update model and check for drift detection
        drift_result = model.partial_fit(batch_X, batch_y)

        # Check adaptation history for drift detections
        if hasattr(model, 'adaptation_history') and model.adaptation_history:
            latest_adaptation = model.adaptation_history[-1]
            if latest_adaptation.get("drift_detected", False):
                detected_drifts.append({
                    "sample_idx": sample_count,
                    "batch_idx": metadata["batch_index"],
                    "drift_score": latest_adaptation.get("drift_score", 0),
                    "drift_type": latest_adaptation.get("drift_type", "unknown"),
                })

        sample_count += len(batch_X)

        # Break if max samples reached
        if config.evaluation and hasattr(config.evaluation, 'max_samples'):
            if config.evaluation.max_samples and sample_count >= config.evaluation.max_samples:
                break

    # Add detected drifts to metrics
    for detection in detected_drifts:
        drift_metrics.add_detected_drift(detection["sample_idx"])

    # Compute metrics
    detection_metrics = drift_metrics.compute_metrics()

    logger.info("Drift Detection Results:")
    for key, value in detection_metrics.items():
        logger.info(f"  {key}: {value}")

    return {
        "detection_metrics": detection_metrics,
        "detected_drifts": detected_drifts,
        "ground_truth_drifts": list(drift_schedule.keys()),
    }


def evaluate_prequential_performance(
    model: AdaptiveEnsembleDetector,
    data_loader: DriftDataLoader,
    config: Config,
) -> Dict:
    """
    Evaluate prequential (test-then-train) performance.

    Args:
        model: Trained adaptive ensemble detector.
        data_loader: Data loader.
        config: Configuration.

    Returns:
        Dictionary with prequential evaluation results.
    """
    logger.info("Evaluating prequential performance...")

    evaluator = PrequentialEvaluator(
        window_size=config.evaluation.prequential_window_size,
        metrics=["accuracy", "f1", "precision", "recall"],
    )

    sample_count = 0
    adaptation_latencies = []

    for batch_X, batch_y, metadata in data_loader:
        # Test: Make predictions before training
        if model.is_fitted:
            y_proba = model.predict_proba(batch_X)
            evaluator.update(batch_X, batch_y, y_proba, metadata)

        # Record adaptation if drift detected
        if hasattr(model, 'adaptation_history'):
            initial_adaptations = len(model.adaptation_history)

        # Train: Update model
        model.partial_fit(batch_X, batch_y)

        # Check if adaptation occurred
        if hasattr(model, 'adaptation_history'):
            if len(model.adaptation_history) > initial_adaptations:
                # New adaptation occurred
                latency = len(batch_X)  # Simplified latency calculation
                adaptation_latencies.append(latency)

        sample_count += len(batch_X)

    # Get evaluation summary
    prequential_summary = evaluator.get_summary()

    # Calculate adaptation statistics
    adaptation_stats = {
        "mean_adaptation_latency": np.mean(adaptation_latencies) if adaptation_latencies else 0,
        "median_adaptation_latency": np.median(adaptation_latencies) if adaptation_latencies else 0,
        "total_adaptations": len(adaptation_latencies),
        "adaptation_latencies": adaptation_latencies,
    }

    logger.info("Prequential Performance:")
    for key, value in prequential_summary.items():
        logger.info(f"  {key}: {value}")

    logger.info("Adaptation Statistics:")
    for key, value in adaptation_stats.items():
        if key != "adaptation_latencies":  # Skip detailed list in logging
            logger.info(f"  {key}: {value}")

    return {
        "prequential_metrics": prequential_summary,
        "adaptation_stats": adaptation_stats,
    }


def evaluate_regret_analysis(
    model: AdaptiveEnsembleDetector,
    oracle_models: Dict,
    data_loader: DriftDataLoader,
    config: Config,
) -> Dict:
    """
    Perform regret analysis against oracle ensemble.

    Args:
        model: Trained adaptive ensemble detector.
        oracle_models: Oracle models dictionary.
        data_loader: Data loader.
        config: Configuration.

    Returns:
        Dictionary with regret analysis results.
    """
    logger.info("Performing regret analysis...")

    regret_analyzer = RegretAnalyzer(
        window_size=config.evaluation.prequential_window_size
    )

    sample_count = 0

    for batch_X, batch_y, metadata in data_loader:
        if model.is_fitted:
            # Get ensemble predictions and loss
            y_proba = model.predict_proba(batch_X)
            ensemble_loss = -np.mean(np.log(y_proba[np.arange(len(batch_y)), batch_y] + 1e-15))

            # Get oracle model losses
            model_losses = {}
            for name, oracle_info in oracle_models.items():
                oracle_model = oracle_info["model"]
                oracle_proba = oracle_model.predict_proba(batch_X)
                oracle_loss = -np.mean(np.log(oracle_proba[np.arange(len(batch_y)), batch_y] + 1e-15))
                model_losses[name] = oracle_loss

            # Update regret analyzer
            regret_metrics = regret_analyzer.update(ensemble_loss, model_losses, batch_y.values)

        # Update model
        model.partial_fit(batch_X, batch_y)

        sample_count += len(batch_X)

    # Get regret analysis summary
    regret_summary = regret_analyzer.get_summary()

    logger.info("Regret Analysis Results:")
    for key, value in regret_summary.items():
        logger.info(f"  {key}: {value}")

    return regret_summary


def run_evaluation_trial(
    config: Config,
    trial_idx: int,
    checkpoint_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Run a single evaluation trial.

    Args:
        config: Configuration.
        trial_idx: Trial index.
        checkpoint_path: Path to checkpoint to load.
        max_samples: Maximum samples to process.

    Returns:
        Dictionary with trial results.
    """
    logger.info(f"Running evaluation trial {trial_idx + 1}")

    # Load data
    data_loader = DataLoader(random_state=config.random_seed + trial_idx)

    if config.data.dataset_name.lower() == "covertype":
        X, y = data_loader.load_covertype()
    elif config.data.dataset_name.lower() == "elec2":
        X, y = data_loader.load_elec2(config.data.data_path)
    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset_name}")

    # Limit samples if specified
    if max_samples and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    # Create drift data loader
    total_batches = len(X) // config.data.batch_size
    drift_schedule = config.data.drift_schedule or {
        total_batches // 4: "covariate_shift",
        total_batches // 2: "concept_drift",
        3 * total_batches // 4: "label_shift",
    }

    streaming_loader = DriftDataLoader(
        X=X,
        y=y,
        batch_size=config.data.batch_size,
        drift_schedule=drift_schedule,
        random_state=config.random_seed + trial_idx,
    )

    # Create or load model
    if checkpoint_path:
        # Load from checkpoint
        trainer = EnsembleTrainer(
            model_config=config.model.__dict__,
            training_config=config.training.__dict__,
            random_state=config.random_seed + trial_idx,
        )
        trainer.load_checkpoint(checkpoint_path)
        model = trainer.model
    else:
        # Train new model
        model = AdaptiveEnsembleDetector(
            drift_detector_params=config.model.drift_detector.__dict__,
            reweighter_params=config.model.reweighter.__dict__,
            base_model_params=config.model.base_models.__dict__,
            random_state=config.random_seed + trial_idx,
        )

        # Initial training on portion of data
        train_size = min(1000, len(X) // 4)
        model.fit(X[:train_size], y[:train_size])

    # Create oracle ensemble
    oracle_models = create_oracle_ensemble(X, y, config)

    # Reset data loader for evaluation
    streaming_loader.reset()

    # Evaluate drift detection
    drift_results = evaluate_drift_detection(model, streaming_loader, config)

    # Reset for prequential evaluation
    streaming_loader.reset()

    # Evaluate prequential performance
    prequential_results = evaluate_prequential_performance(model, streaming_loader, config)

    # Reset for regret analysis
    streaming_loader.reset()

    # Evaluate regret
    regret_results = evaluate_regret_analysis(model, oracle_models, streaming_loader, config)

    return {
        "trial_idx": trial_idx,
        "drift_detection": drift_results,
        "prequential": prequential_results,
        "regret_analysis": regret_results,
        "model_summary": model.get_adaptation_summary(),
    }


def aggregate_results(trial_results: List[Dict], config: Config) -> Dict:
    """
    Aggregate results across multiple trials.

    Args:
        trial_results: List of trial result dictionaries.
        config: Configuration.

    Returns:
        Dictionary with aggregated results.
    """
    logger.info("Aggregating results across trials...")

    # Extract metrics from each trial
    drift_f1_scores = []
    prequential_accuracies = []
    adaptation_latencies = []
    regret_vs_oracle = []

    for result in trial_results:
        # Drift detection F1
        drift_metrics = result["drift_detection"]["detection_metrics"]
        drift_f1_scores.append(drift_metrics.get("drift_detection_f1", 0))

        # Prequential accuracy
        prequential_metrics = result["prequential"]["prequential_metrics"]
        if "accuracy_final" in prequential_metrics:
            prequential_accuracies.append(prequential_metrics["accuracy_final"])
        elif "accuracy_mean" in prequential_metrics:
            prequential_accuracies.append(prequential_metrics["accuracy_mean"])

        # Adaptation latency
        adaptation_stats = result["prequential"]["adaptation_stats"]
        adaptation_latencies.append(adaptation_stats.get("mean_adaptation_latency", 0))

        # Regret vs oracle
        regret_analysis = result["regret_analysis"]
        regret_vs_oracle.append(abs(regret_analysis.get("relative_regret", 0)))

    # Calculate statistics
    aggregated = {
        "drift_detection_f1": {
            "mean": np.mean(drift_f1_scores),
            "std": np.std(drift_f1_scores),
            "values": drift_f1_scores,
        },
        "prequential_accuracy": {
            "mean": np.mean(prequential_accuracies),
            "std": np.std(prequential_accuracies),
            "values": prequential_accuracies,
        },
        "adaptation_latency": {
            "mean": np.mean(adaptation_latencies),
            "std": np.std(adaptation_latencies),
            "values": adaptation_latencies,
        },
        "regret_vs_oracle": {
            "mean": np.mean(regret_vs_oracle),
            "std": np.std(regret_vs_oracle),
            "values": regret_vs_oracle,
        },
    }

    # Compare against targets
    targets = {
        "prequential_accuracy": config.evaluation.target_prequential_accuracy,
        "drift_detection_f1": config.evaluation.target_drift_detection_f1,
        "adaptation_latency": config.evaluation.target_adaptation_latency,
        "regret_vs_oracle": config.evaluation.target_regret_vs_oracle,
    }

    achievements = {}
    for metric, target in targets.items():
        if metric in aggregated:
            achieved = aggregated[metric]["mean"]
            if metric == "adaptation_latency" or metric == "regret_vs_oracle":
                # Lower is better
                achievements[metric] = achieved <= target
            else:
                # Higher is better
                achievements[metric] = achieved >= target

    logger.info("Aggregated Results:")
    for metric, stats in aggregated.items():
        target = targets.get(metric, "N/A")
        status = "✓" if achievements.get(metric, False) else "✗"
        logger.info(f"  {metric}: {stats['mean']:.3f} ± {stats['std']:.3f} (target: {target}) {status}")

    return {
        "aggregated_metrics": aggregated,
        "target_achievements": achievements,
        "targets": targets,
        "n_trials": len(trial_results),
    }


def create_evaluation_plots(
    aggregated_results: Dict,
    output_dir: Path,
) -> None:
    """
    Create evaluation plots.

    Args:
        aggregated_results: Aggregated results dictionary.
        output_dir: Output directory for plots.
    """
    logger.info("Creating evaluation plots...")

    plt.style.use('default')
    sns.set_palette("husl")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Metrics comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metrics = ["drift_detection_f1", "prequential_accuracy", "adaptation_latency", "regret_vs_oracle"]
    titles = ["Drift Detection F1", "Prequential Accuracy", "Adaptation Latency", "Regret vs Oracle"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric in aggregated_results["aggregated_metrics"]:
            values = aggregated_results["aggregated_metrics"][metric]["values"]
            target = aggregated_results["targets"][metric]

            axes[i].boxplot(values)
            axes[i].axhline(y=target, color='r', linestyle='--', label=f'Target: {target}')
            axes[i].set_title(title)
            axes[i].set_ylabel("Value")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Target achievement summary
    fig, ax = plt.subplots(figsize=(10, 6))

    achievement_data = []
    metric_names = []

    for metric, achieved in aggregated_results["target_achievements"].items():
        achievement_data.append(1 if achieved else 0)
        metric_names.append(metric.replace("_", " ").title())

    colors = ['green' if x else 'red' for x in achievement_data]
    bars = ax.bar(metric_names, achievement_data, color=colors, alpha=0.7)

    ax.set_ylabel("Target Achieved")
    ax.set_title("Target Achievement Summary")
    ax.set_ylim(0, 1.2)

    # Add value labels on bars
    for bar, achieved in zip(bars, achievement_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                "✓" if achieved else "✗",
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / "target_achievements.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Plots saved to {plots_dir}")


def save_detailed_results(
    trial_results: List[Dict],
    aggregated_results: Dict,
    output_dir: Path,
) -> None:
    """
    Save detailed results to files.

    Args:
        trial_results: List of trial results.
        aggregated_results: Aggregated results.
        output_dir: Output directory.
    """
    logger.info("Saving detailed results...")

    # Save aggregated results
    import json

    with open(output_dir / "aggregated_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_safe_results = {}
        for key, value in aggregated_results.items():
            if isinstance(value, dict):
                json_safe_value = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        json_safe_value[subkey] = subvalue.tolist()
                    else:
                        json_safe_value[subkey] = subvalue
                json_safe_results[key] = json_safe_value
            else:
                json_safe_results[key] = value

        json.dump(json_safe_results, f, indent=2)

    # Save individual trial results
    for i, result in enumerate(trial_results):
        trial_file = output_dir / f"trial_{i+1}_results.json"
        with open(trial_file, "w") as f:
            # Simplified trial result (remove complex objects)
            simplified_result = {
                "trial_idx": result["trial_idx"],
                "drift_detection_f1": result["drift_detection"]["detection_metrics"].get("drift_detection_f1", 0),
                "prequential_accuracy": result["prequential"]["prequential_metrics"].get("accuracy_final", 0),
                "adaptation_latency": result["prequential"]["adaptation_stats"].get("mean_adaptation_latency", 0),
                "regret_vs_oracle": abs(result["regret_analysis"].get("relative_regret", 0)),
            }
            json.dump(simplified_result, f, indent=2)

    logger.info(f"Detailed results saved to {output_dir}")


def main():
    """Main evaluation function."""
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
    if args.debug:
        config.log_level = "DEBUG"

    # Setup environment
    setup_environment(config)

    logger.info("Starting evaluation script")
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Dataset: {config.data.dataset_name}")
    logger.info(f"Number of trials: {args.n_trials}")

    output_dir = Path(config.output_dir) / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run evaluation trials
        trial_results = []

        for trial_idx in range(args.n_trials):
            result = run_evaluation_trial(
                config=config,
                trial_idx=trial_idx,
                checkpoint_path=args.checkpoint,
                max_samples=args.max_samples,
            )
            trial_results.append(result)

        # Aggregate results
        aggregated_results = aggregate_results(trial_results, config)

        # Create plots if requested
        if args.plot or config.evaluation.plot_results:
            create_evaluation_plots(aggregated_results, output_dir)

        # Save detailed results if requested
        if args.save_results:
            save_detailed_results(trial_results, aggregated_results, output_dir)

        logger.info("Evaluation completed successfully!")

        # Print final summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)

        achievements = aggregated_results["target_achievements"]
        targets_met = sum(achievements.values())
        total_targets = len(achievements)

        print(f"Targets achieved: {targets_met}/{total_targets}")
        print()

        for metric, achieved in achievements.items():
            status = "✓" if achieved else "✗"
            value = aggregated_results["aggregated_metrics"][metric]["mean"]
            target = aggregated_results["targets"][metric]
            print(f"{status} {metric}: {value:.3f} (target: {target})")

        print("="*60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()