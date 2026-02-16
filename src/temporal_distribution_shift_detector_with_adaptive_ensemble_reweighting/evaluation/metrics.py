"""
Evaluation metrics and analysis tools for temporal distribution shift detection.

This module provides specialized metrics for streaming data evaluation, including
prequential accuracy, drift detection performance, and regret analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from collections import deque

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class PrequentialEvaluator:
    """
    Prequential (test-then-train) evaluator for streaming data scenarios.

    Evaluates model performance on a sliding window basis, which is appropriate
    for non-stationary data streams with concept drift.
    """

    def __init__(
        self,
        window_size: int = 1000,
        metrics: List[str] = None,
        alpha: float = 0.05,
    ):
        """
        Initialize the prequential evaluator.

        Args:
            window_size: Size of the sliding window for evaluation.
            metrics: List of metrics to compute.
            alpha: Significance level for statistical tests.
        """
        self.window_size = window_size
        self.metrics = metrics or ["accuracy", "f1", "precision", "recall"]
        self.alpha = alpha

        # Sliding windows for true and predicted values
        self.y_true_window = deque(maxlen=window_size)
        self.y_pred_window = deque(maxlen=window_size)
        self.y_proba_window = deque(maxlen=window_size)

        # Performance tracking
        self.performance_history: List[Dict] = []
        self.samples_seen = 0

    def update(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_proba: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Update evaluator with new batch and compute metrics.

        Args:
            X: Input features (for context).
            y_true: True labels.
            y_proba: Predicted probabilities.
            metadata: Additional metadata.

        Returns:
            Dictionary of computed metrics.
        """
        # Convert probabilities to predictions
        if len(y_proba.shape) == 1:
            # Binary classification
            y_pred = (y_proba > 0.5).astype(int)
        else:
            # Multi-class classification
            y_pred = np.argmax(y_proba, axis=1)

        # Update sliding windows
        for i in range(len(y_true)):
            self.y_true_window.append(y_true.iloc[i])
            self.y_pred_window.append(y_pred[i])
            self.y_proba_window.append(y_proba[i])

        self.samples_seen += len(y_true)

        # Compute metrics if we have enough samples
        if len(self.y_true_window) >= min(100, self.window_size):
            metrics = self._compute_metrics()

            # Add metadata
            if metadata:
                metrics.update(metadata)

            metrics["samples_seen"] = self.samples_seen
            metrics["window_size"] = len(self.y_true_window)

            self.performance_history.append(metrics)
            return metrics

        return {"samples_seen": self.samples_seen}

    def _compute_metrics(self) -> Dict[str, float]:
        """
        Compute metrics on the current window.

        Returns:
            Dictionary of computed metrics.
        """
        y_true = np.array(self.y_true_window)
        y_pred = np.array(self.y_pred_window)
        y_proba = np.array(self.y_proba_window)

        metrics = {}

        # Basic classification metrics
        if "accuracy" in self.metrics:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)

        if "precision" in self.metrics:
            metrics["precision"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

        if "recall" in self.metrics:
            metrics["recall"] = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

        if "f1" in self.metrics:
            metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # AUC for binary classification
        if len(np.unique(y_true)) == 2 and len(y_proba.shape) > 1:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except ValueError:
                pass

        # Confidence-based metrics
        if len(y_proba.shape) == 1:
            confidence = np.abs(y_proba - 0.5) * 2  # For binary classification
        else:
            confidence = np.max(y_proba, axis=1)  # For multi-class

        metrics["avg_confidence"] = np.mean(confidence)
        metrics["confidence_std"] = np.std(confidence)

        # Calibration metrics
        metrics.update(self._compute_calibration_metrics(y_true, y_proba))

        return metrics

    def _compute_calibration_metrics(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute calibration metrics.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.

        Returns:
            Dictionary of calibration metrics.
        """
        if len(y_proba.shape) == 1:
            # Binary classification
            proba_pos = y_proba
        else:
            # Multi-class: use probability of predicted class
            proba_pos = np.max(y_proba, axis=1)

        # Binned calibration
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0  # Expected Calibration Error
        mce = 0  # Maximum Calibration Error

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (proba_pos > bin_lower) & (proba_pos <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = proba_pos[in_bin].mean()

                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += calibration_error * prop_in_bin
                mce = max(mce, calibration_error)

        return {"ece": ece, "mce": mce}

    def get_summary(self) -> Dict[str, Union[float, int]]:
        """
        Get summary statistics of evaluation performance.

        Returns:
            Dictionary of summary statistics.
        """
        if not self.performance_history:
            return {}

        # Extract metrics from history
        metrics_df = pd.DataFrame(self.performance_history)

        summary = {
            "total_samples": self.samples_seen,
            "total_evaluations": len(self.performance_history),
        }

        # Add statistics for each metric
        for metric in self.metrics:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                if len(values) > 0:
                    summary.update({
                        f"{metric}_mean": values.mean(),
                        f"{metric}_std": values.std(),
                        f"{metric}_min": values.min(),
                        f"{metric}_max": values.max(),
                        f"{metric}_final": values.iloc[-1],
                    })

        return summary

    def plot_performance(self, save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics over time.

        Args:
            save_path: Path to save the plot.
        """
        if not self.performance_history:
            logger.warning("No performance history to plot")
            return

        metrics_df = pd.DataFrame(self.performance_history)

        # Create subplots
        n_metrics = len(self.metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, metric in enumerate(self.metrics):
            if metric in metrics_df.columns:
                ax = axes[i] if n_metrics > 1 else axes
                ax.plot(metrics_df[metric], label=metric)
                ax.set_title(f"{metric.title()} Over Time")
                ax.set_xlabel("Evaluation Step")
                ax.set_ylabel(metric.title())
                ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()


class DriftMetrics:
    """
    Specialized metrics for drift detection evaluation.

    Evaluates the performance of drift detection algorithms including
    detection delay, false alarm rate, and detection accuracy.
    """

    def __init__(self, tolerance_window: int = 100):
        """
        Initialize drift metrics evaluator.

        Args:
            tolerance_window: Tolerance window for drift detection delay.
        """
        self.tolerance_window = tolerance_window

        # Ground truth and detected drift points
        self.true_drift_points: List[int] = []
        self.detected_drift_points: List[int] = []
        self.false_alarms: List[int] = []

        # Detection performance
        self.detection_delays: List[int] = []

    def add_true_drift(self, sample_index: int) -> None:
        """
        Add a true drift point.

        Args:
            sample_index: Sample index where drift occurred.
        """
        self.true_drift_points.append(sample_index)

    def add_detected_drift(self, sample_index: int, is_true_drift: bool = None) -> None:
        """
        Add a detected drift point.

        Args:
            sample_index: Sample index where drift was detected.
            is_true_drift: Whether this detection corresponds to true drift.
        """
        self.detected_drift_points.append(sample_index)

        if is_true_drift is False:
            self.false_alarms.append(sample_index)
        elif is_true_drift is None:
            # Automatically determine if it's a true positive or false alarm
            self._evaluate_detection(sample_index)

    def _evaluate_detection(self, detected_index: int) -> bool:
        """
        Evaluate whether a detection is a true positive or false alarm.

        Args:
            detected_index: Index of detected drift.

        Returns:
            True if true positive, False if false alarm.
        """
        for true_index in self.true_drift_points:
            if abs(detected_index - true_index) <= self.tolerance_window:
                # True positive
                delay = detected_index - true_index
                self.detection_delays.append(delay)
                return True

        # False alarm
        self.false_alarms.append(detected_index)
        return False

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute drift detection metrics.

        Returns:
            Dictionary of drift detection metrics.
        """
        n_true_drifts = len(self.true_drift_points)
        n_detected_drifts = len(self.detected_drift_points)
        n_false_alarms = len(self.false_alarms)
        n_true_positives = len(self.detection_delays)
        n_false_negatives = n_true_drifts - n_true_positives

        metrics = {
            "n_true_drifts": n_true_drifts,
            "n_detected_drifts": n_detected_drifts,
            "n_true_positives": n_true_positives,
            "n_false_positives": n_false_alarms,
            "n_false_negatives": n_false_negatives,
        }

        # Precision, Recall, F1
        if n_detected_drifts > 0:
            precision = n_true_positives / n_detected_drifts
            metrics["drift_detection_precision"] = precision
        else:
            metrics["drift_detection_precision"] = 0.0

        if n_true_drifts > 0:
            recall = n_true_positives / n_true_drifts
            metrics["drift_detection_recall"] = recall
        else:
            metrics["drift_detection_recall"] = 1.0

        precision = metrics["drift_detection_precision"]
        recall = metrics["drift_detection_recall"]

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            metrics["drift_detection_f1"] = f1
        else:
            metrics["drift_detection_f1"] = 0.0

        # Detection delay statistics
        if self.detection_delays:
            metrics.update({
                "avg_detection_delay": np.mean(self.detection_delays),
                "median_detection_delay": np.median(self.detection_delays),
                "max_detection_delay": np.max(self.detection_delays),
                "min_detection_delay": np.min(self.detection_delays),
                "std_detection_delay": np.std(self.detection_delays),
            })
        else:
            metrics.update({
                "avg_detection_delay": float("inf"),
                "median_detection_delay": float("inf"),
                "max_detection_delay": float("inf"),
                "min_detection_delay": float("inf"),
                "std_detection_delay": 0.0,
            })

        return metrics


class RegretAnalyzer:
    """
    Analyzer for computing regret bounds and comparing ensemble performance
    against oracle ensemble performance.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize regret analyzer.

        Args:
            window_size: Window size for regret computation.
        """
        self.window_size = window_size

        # Performance tracking
        self.ensemble_losses: List[float] = []
        self.oracle_losses: List[float] = []
        self.cumulative_regret: List[float] = []

        # Model performance tracking
        self.model_losses: Dict[str, List[float]] = {}

    def update(
        self,
        ensemble_loss: float,
        model_losses: Dict[str, float],
        true_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update regret analysis with new performance data.

        Args:
            ensemble_loss: Loss of the adaptive ensemble.
            model_losses: Losses of individual models.
            true_labels: True labels for oracle computation.

        Returns:
            Dictionary with regret metrics.
        """
        self.ensemble_losses.append(ensemble_loss)

        # Track individual model losses
        for model_name, loss in model_losses.items():
            if model_name not in self.model_losses:
                self.model_losses[model_name] = []
            self.model_losses[model_name].append(loss)

        # Compute oracle loss (best performing model at each step)
        oracle_loss = min(model_losses.values())
        self.oracle_losses.append(oracle_loss)

        # Compute instantaneous regret
        instantaneous_regret = ensemble_loss - oracle_loss

        # Compute cumulative regret
        if self.cumulative_regret:
            cumulative_regret = self.cumulative_regret[-1] + instantaneous_regret
        else:
            cumulative_regret = instantaneous_regret

        self.cumulative_regret.append(cumulative_regret)

        # Compute regret metrics
        metrics = self._compute_regret_metrics()
        metrics["instantaneous_regret"] = instantaneous_regret

        return metrics

    def _compute_regret_metrics(self) -> Dict[str, float]:
        """
        Compute various regret metrics.

        Returns:
            Dictionary of regret metrics.
        """
        if not self.cumulative_regret:
            return {}

        T = len(self.cumulative_regret)

        metrics = {
            "cumulative_regret": self.cumulative_regret[-1],
            "average_regret": self.cumulative_regret[-1] / T,
            "regret_bound": np.sqrt(np.log(T) / T) if T > 0 else 0.0,
        }

        # Compute regret relative to oracle
        if self.oracle_losses:
            total_oracle_loss = sum(self.oracle_losses)
            total_ensemble_loss = sum(self.ensemble_losses)

            if total_oracle_loss > 0:
                metrics["relative_regret"] = (
                    total_ensemble_loss - total_oracle_loss
                ) / total_oracle_loss
            else:
                metrics["relative_regret"] = 0.0

        # Statistical analysis of regret
        recent_regrets = self.cumulative_regret[-min(self.window_size, len(self.cumulative_regret)):]
        if recent_regrets:
            metrics.update({
                "recent_regret_mean": np.mean(recent_regrets),
                "recent_regret_std": np.std(recent_regrets),
                "regret_trend": self._compute_regret_trend(),
            })

        return metrics

    def _compute_regret_trend(self) -> float:
        """
        Compute trend in regret (positive = increasing, negative = decreasing).

        Returns:
            Slope of regret trend.
        """
        if len(self.cumulative_regret) < 10:
            return 0.0

        # Use linear regression on recent regret values
        recent_regrets = self.cumulative_regret[-min(50, len(self.cumulative_regret)):]
        x = np.arange(len(recent_regrets))

        # Compute slope
        slope, _, _, _, _ = stats.linregress(x, recent_regrets)
        return slope

    def get_oracle_performance(self) -> Dict[str, float]:
        """
        Get oracle (best possible) performance statistics.

        Returns:
            Dictionary with oracle performance metrics.
        """
        if not self.oracle_losses:
            return {}

        return {
            "oracle_avg_loss": np.mean(self.oracle_losses),
            "oracle_total_loss": sum(self.oracle_losses),
            "oracle_min_loss": min(self.oracle_losses),
            "oracle_max_loss": max(self.oracle_losses),
        }

    def plot_regret_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Plot regret analysis results.

        Args:
            save_path: Path to save the plot.
        """
        if not self.cumulative_regret:
            logger.warning("No regret data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Cumulative regret
        axes[0, 0].plot(self.cumulative_regret)
        axes[0, 0].set_title("Cumulative Regret")
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Cumulative Regret")
        axes[0, 0].grid(True, alpha=0.3)

        # Instantaneous losses
        axes[0, 1].plot(self.ensemble_losses, label="Ensemble", alpha=0.7)
        axes[0, 1].plot(self.oracle_losses, label="Oracle", alpha=0.7)
        axes[0, 1].set_title("Loss Comparison")
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Individual model losses
        for model_name, losses in self.model_losses.items():
            axes[1, 0].plot(losses, label=model_name, alpha=0.7)
        axes[1, 0].set_title("Individual Model Losses")
        axes[1, 0].set_xlabel("Time Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Regret distribution
        instantaneous_regrets = [
            ens - oracle for ens, oracle in zip(self.ensemble_losses, self.oracle_losses)
        ]
        axes[1, 1].hist(instantaneous_regrets, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 1].set_title("Instantaneous Regret Distribution")
        axes[1, 1].set_xlabel("Regret")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def get_summary(self) -> Dict[str, float]:
        """
        Get comprehensive regret analysis summary.

        Returns:
            Summary dictionary.
        """
        summary = self._compute_regret_metrics()
        summary.update(self.get_oracle_performance())

        # Add efficiency metrics
        if self.ensemble_losses and self.oracle_losses:
            efficiency = 1 - (sum(self.ensemble_losses) / sum(self.oracle_losses))
            summary["ensemble_efficiency"] = max(0, efficiency)

        return summary