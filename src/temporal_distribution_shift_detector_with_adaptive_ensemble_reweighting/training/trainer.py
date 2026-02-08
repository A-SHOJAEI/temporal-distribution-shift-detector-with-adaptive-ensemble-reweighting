"""
Training utilities and pipeline for the adaptive ensemble detector.

This module provides the main training loop with MLflow integration,
checkpoint saving, and streaming data support.
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from tqdm import tqdm

from ..models.model import AdaptiveEnsembleDetector
from ..data.loader import DriftDataLoader
from ..data.preprocessing import FeatureProcessor
from ..evaluation.metrics import PrequentialEvaluator

logger = logging.getLogger(__name__)


class EarlyStoppingCallback:
    """Early stopping callback for training."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping callback.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.wait_count = 0
        self.stopped = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score.

        Returns:
            True if training should stop.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            self.stopped = True
            logger.info(f"Early stopping triggered after {self.wait_count} epochs")

        return self.stopped


class OnlineLearner:
    """Online learning component for streaming data scenarios."""

    def __init__(
        self,
        model: AdaptiveEnsembleDetector,
        evaluator: PrequentialEvaluator,
        update_frequency: int = 100,
        evaluation_frequency: int = 500,
    ):
        """
        Initialize the online learner.

        Args:
            model: The adaptive ensemble model.
            evaluator: Prequential evaluator for streaming evaluation.
            update_frequency: How often to update the model (in samples).
            evaluation_frequency: How often to evaluate (in samples).
        """
        self.model = model
        self.evaluator = evaluator
        self.update_frequency = update_frequency
        self.evaluation_frequency = evaluation_frequency

        self.samples_seen = 0
        self.last_update = 0
        self.last_evaluation = 0

        # Performance tracking
        self.performance_history: List[Dict] = []

    def learn_batch(self, X: pd.DataFrame, y: pd.Series, metadata: Dict) -> Dict[str, float]:
        """
        Learn from a batch of streaming data.

        Args:
            X: Features.
            y: Labels.
            metadata: Batch metadata.

        Returns:
            Dictionary with learning statistics.
        """
        batch_size = len(X)
        self.samples_seen += batch_size

        # Make predictions before learning (prequential evaluation)
        predictions = self.model.predict_proba(X) if self.model.is_fitted else None

        # Evaluate if needed
        evaluation_results = {}
        if (
            self.samples_seen - self.last_evaluation >= self.evaluation_frequency
            and predictions is not None
        ):
            evaluation_results = self.evaluator.update(X, y, predictions, metadata)
            self.last_evaluation = self.samples_seen

        # Update model if needed
        if self.samples_seen - self.last_update >= self.update_frequency:
            self.model.partial_fit(X, y)
            self.last_update = self.samples_seen

        # Track performance
        performance_info = {
            "samples_seen": self.samples_seen,
            "batch_size": batch_size,
            "drift_type": metadata.get("drift_type", "none"),
            **evaluation_results,
        }
        self.performance_history.append(performance_info)

        return performance_info

    def get_performance_summary(self) -> Dict:
        """
        Get summary of online learning performance.

        Returns:
            Performance summary dictionary.
        """
        if not self.performance_history:
            return {}

        recent_performance = self.performance_history[-10:]  # Last 10 batches

        return {
            "total_samples_seen": self.samples_seen,
            "total_batches_processed": len(self.performance_history),
            "recent_accuracy": np.mean([p.get("accuracy", 0) for p in recent_performance]),
            "adaptation_summary": self.model.get_adaptation_summary(),
            "evaluator_summary": self.evaluator.get_summary(),
        }


class EnsembleTrainer:
    """
    Main trainer for the adaptive ensemble detector with MLflow integration.

    Handles both batch training and online learning scenarios.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        mlflow_config: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        """
        Initialize the ensemble trainer.

        Args:
            model_config: Configuration for the model.
            training_config: Configuration for training.
            mlflow_config: Configuration for MLflow tracking.
            random_state: Random seed for reproducibility.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.mlflow_config = mlflow_config or {}
        self.random_state = random_state

        # Training components
        self.model: Optional[AdaptiveEnsembleDetector] = None
        self.feature_processor: Optional[FeatureProcessor] = None
        self.evaluator: Optional[PrequentialEvaluator] = None
        self.online_learner: Optional[OnlineLearner] = None

        # Training state
        self.training_history: List[Dict] = []
        self.checkpoint_dir = Path(training_config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)

        # MLflow setup
        self._setup_mlflow()

    @staticmethod
    def _flatten_params(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dict for mlflow.log_params (which doesn't support nested dicts)."""
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                flat.update(EnsembleTrainer._flatten_params(v, key))
            elif isinstance(v, (list, tuple)):
                flat[key] = str(v)
            else:
                flat[key] = v
        return flat

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if self.mlflow_config.get("tracking_uri"):
            mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])

        experiment_name = self.mlflow_config.get(
            "experiment_name", "temporal_distribution_shift_detection"
        )
        try:
            mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            pass  # Experiment already exists

        mlflow.set_experiment(experiment_name)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> AdaptiveEnsembleDetector:
        """
        Train the adaptive ensemble detector.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Trained model.
        """
        logger.info("Starting ensemble training...")

        with mlflow.start_run():
            # Log parameters (flatten nested dicts for MLflow compatibility)
            mlflow.log_params(self._flatten_params(self.model_config, "model"))
            mlflow.log_params(self._flatten_params(self.training_config, "training"))

            # Split validation set if not provided
            if X_val is None or y_val is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=self.training_config.get("validation_split", 0.2),
                    random_state=self.random_state,
                    stratify=y_train,
                )

            # Initialize feature processor
            self.feature_processor = FeatureProcessor(
                scaler_type=self.training_config.get("scaler_type", "standard"),
                handle_missing=self.training_config.get("handle_missing", "median"),
                random_state=self.random_state,
            )

            # Process features
            logger.info("Processing features...")
            X_train_processed = self.feature_processor.fit_transform(X_train, y_train)
            X_val_processed = self.feature_processor.transform(X_val)

            # Initialize model
            self.model = AdaptiveEnsembleDetector(
                drift_detector_params=self.model_config.get("drift_detector", {}),
                reweighter_params=self.model_config.get("reweighter", {}),
                base_model_params=self.model_config.get("base_models", {}),
                random_state=self.random_state,
            )

            # Train model
            logger.info("Training ensemble models...")
            start_time = time.time()
            self.model.fit(X_train_processed, y_train)
            training_time = time.time() - start_time

            # Evaluate on validation set
            val_predictions = self.model.predict(X_val_processed)
            val_proba = self.model.predict_proba(X_val_processed)

            val_metrics = self._compute_metrics(y_val, val_predictions, val_proba)

            # Log metrics
            mlflow.log_metrics(val_metrics)
            mlflow.log_metric("training_time", training_time)
            mlflow.log_metric("n_features", X_train_processed.shape[1])
            mlflow.log_metric("n_train_samples", len(X_train_processed))

            # Log model
            mlflow.sklearn.log_model(self.model, "adaptive_ensemble_model")

            # Save checkpoint
            self._save_checkpoint("final_model")

            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Validation accuracy: {val_metrics['accuracy']:.3f}")

            return self.model

    def train_streaming(
        self,
        data_loader: DriftDataLoader,
        max_samples: Optional[int] = None,
        save_frequency: int = 1000,
    ) -> Dict[str, Any]:
        """
        Train using streaming data with drift simulation.

        Args:
            data_loader: Streaming data loader.
            max_samples: Maximum number of samples to process.
            save_frequency: How often to save checkpoints (in batches).

        Returns:
            Training summary dictionary.
        """
        logger.info("Starting streaming training...")

        with mlflow.start_run():
            # Log parameters (flatten nested dicts for MLflow compatibility)
            mlflow.log_params(self._flatten_params(self.model_config, "model"))
            mlflow.log_params(self._flatten_params(self.training_config, "training"))
            mlflow.log_params({"streaming": True, "max_samples": max_samples})

            # Initialize components
            if self.model is None:
                self.model = AdaptiveEnsembleDetector(
                    drift_detector_params=self.model_config.get("drift_detector", {}),
                    reweighter_params=self.model_config.get("reweighter", {}),
                    base_model_params=self.model_config.get("base_models", {}),
                    random_state=self.random_state,
                )

            if self.feature_processor is None:
                self.feature_processor = FeatureProcessor(
                    scaler_type=self.training_config.get("scaler_type", "standard"),
                    handle_missing=self.training_config.get("handle_missing", "median"),
                    random_state=self.random_state,
                )

            # Initialize evaluator for streaming evaluation
            from ..evaluation.metrics import PrequentialEvaluator

            self.evaluator = PrequentialEvaluator(
                window_size=self.training_config.get("eval_window_size", 1000),
                metrics=["accuracy", "f1", "drift_detection"],
            )

            # Initialize online learner
            self.online_learner = OnlineLearner(
                model=self.model,
                evaluator=self.evaluator,
                update_frequency=self.training_config.get("update_frequency", 100),
                evaluation_frequency=self.training_config.get("evaluation_frequency", 500),
            )

            # Training loop
            total_samples = 0
            batch_count = 0
            initial_training_done = False

            start_time = time.time()

            with tqdm(desc="Processing batches", unit="batch") as pbar:
                for batch_X, batch_y, metadata in data_loader:
                    if max_samples and total_samples >= max_samples:
                        break

                    # Process features
                    if not initial_training_done:
                        # Initial training on first batch
                        batch_X_processed = self.feature_processor.fit_transform(
                            batch_X, batch_y
                        )
                        self.model.fit(batch_X_processed, batch_y)
                        initial_training_done = True
                    else:
                        # Incremental processing
                        batch_X_processed = self.feature_processor.transform(batch_X)

                    # Online learning
                    batch_performance = self.online_learner.learn_batch(
                        batch_X_processed, batch_y, metadata
                    )

                    # Update counters
                    total_samples += len(batch_X)
                    batch_count += 1

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "samples": total_samples,
                            "accuracy": f"{batch_performance.get('accuracy', 0):.3f}",
                            "drift": metadata.get("drift_type", "none"),
                        }
                    )
                    pbar.update(1)

                    # Save checkpoint
                    if batch_count % save_frequency == 0:
                        self._save_checkpoint(f"batch_{batch_count}")

                    # Log batch metrics
                    if batch_count % 10 == 0:  # Log every 10 batches
                        for key, value in batch_performance.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"batch_{key}", value, step=batch_count)

            training_time = time.time() - start_time

            # Final evaluation and logging
            final_performance = self.online_learner.get_performance_summary()
            adaptation_summary = self.model.get_adaptation_summary()

            # Log final metrics
            mlflow.log_metrics(final_performance)
            mlflow.log_metrics(adaptation_summary)
            mlflow.log_metric("total_training_time", training_time)
            mlflow.log_metric("total_samples_processed", total_samples)
            mlflow.log_metric("total_batches_processed", batch_count)

            # Log model
            mlflow.sklearn.log_model(self.model, "streaming_adaptive_ensemble_model")

            # Save final checkpoint
            self._save_checkpoint("streaming_final")

            logger.info(f"Streaming training completed: {total_samples} samples in {training_time:.2f}s")

            return {
                "total_samples": total_samples,
                "total_batches": batch_count,
                "training_time": training_time,
                "final_performance": final_performance,
                "adaptation_summary": adaptation_summary,
            }

    def _compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities.

        Returns:
            Dictionary of metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        if y_proba is not None:
            from sklearn.metrics import roc_auc_score, log_loss

            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics["auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )

                metrics["log_loss"] = log_loss(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Could not compute AUC or log_loss: {e}")

        return metrics

    def _save_checkpoint(self, name: str) -> None:
        """
        Save model checkpoint.

        Args:
            name: Checkpoint name.
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"

        checkpoint_data = {
            "model": self.model,
            "feature_processor": self.feature_processor,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }

        try:
            joblib.dump(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Log checkpoint as artifact in MLflow
            mlflow.log_artifact(str(checkpoint_path))
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        try:
            checkpoint_data = joblib.load(checkpoint_path)

            self.model = checkpoint_data["model"]
            self.feature_processor = checkpoint_data["feature_processor"]
            self.model_config = checkpoint_data.get("model_config", self.model_config)
            self.training_config = checkpoint_data.get("training_config", self.training_config)

            logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            X: Input features.

        Returns:
            Tuple of (predictions, probabilities).
        """
        if self.model is None or self.feature_processor is None:
            raise ValueError("Model must be trained before prediction")

        X_processed = self.feature_processor.transform(X)
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)

        return predictions, probabilities

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.

        Returns:
            Model summary dictionary.
        """
        if self.model is None:
            return {}

        summary = {
            "model_type": "AdaptiveEnsembleDetector",
            "n_base_models": len(self.model.base_models),
            "base_model_names": self.model.model_names,
            "current_weights": self.model.get_model_importance(),
            "adaptation_summary": self.model.get_adaptation_summary(),
            "drift_history": self.model.get_drift_history(),
        }

        if self.online_learner:
            summary["online_learning_summary"] = self.online_learner.get_performance_summary()

        return summary