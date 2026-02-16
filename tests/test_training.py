"""
Test suite for training components.

Tests cover the ensemble trainer, online learner, and MLflow integration.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import mlflow

from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.training.trainer import (
    EnsembleTrainer,
    OnlineLearner,
    EarlyStoppingCallback,
)
from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.evaluation.metrics import (
    PrequentialEvaluator,
)
from tests.conftest import assert_is_fitted, create_batch_data


class TestEarlyStoppingCallback:
    """Test suite for EarlyStoppingCallback class."""

    def test_init(self):
        """Test EarlyStoppingCallback initialization."""
        callback = EarlyStoppingCallback(patience=5, min_delta=0.01)

        assert callback.patience == 5
        assert callback.min_delta == 0.01
        assert callback.best_score == float("-inf")
        assert callback.wait_count == 0
        assert not callback.stopped

    def test_improvement_detection(self):
        """Test detection of score improvements."""
        callback = EarlyStoppingCallback(patience=3, min_delta=0.01)

        # Initial score
        assert not callback(0.8)
        assert callback.best_score == 0.8
        assert callback.wait_count == 0

        # Improvement
        assert not callback(0.85)
        assert callback.best_score == 0.85
        assert callback.wait_count == 0

        # Small improvement (below min_delta)
        assert not callback(0.851)
        assert callback.best_score == 0.85  # Should not update
        assert callback.wait_count == 1

    def test_early_stopping_trigger(self):
        """Test early stopping trigger."""
        callback = EarlyStoppingCallback(patience=2, min_delta=0.01)

        # Set initial score
        assert not callback(0.8)

        # No improvement for patience epochs
        assert not callback(0.79)  # wait_count = 1
        assert not callback(0.78)  # wait_count = 2, should stop
        assert callback.stopped


class TestOnlineLearner:
    """Test suite for OnlineLearner class."""

    def test_init(self, adaptive_ensemble_detector):
        """Test OnlineLearner initialization."""
        evaluator = PrequentialEvaluator()
        learner = OnlineLearner(
            model=adaptive_ensemble_detector,
            evaluator=evaluator,
            update_frequency=50,
            evaluation_frequency=100,
        )

        assert learner.model == adaptive_ensemble_detector
        assert learner.evaluator == evaluator
        assert learner.update_frequency == 50
        assert learner.evaluation_frequency == 100
        assert learner.samples_seen == 0

    def test_learn_batch_unfitted_model(self, adaptive_ensemble_detector):
        """Test learning with unfitted model."""
        evaluator = PrequentialEvaluator()
        learner = OnlineLearner(adaptive_ensemble_detector, evaluator)

        # Create sample data
        X = pd.DataFrame(np.random.random((50, 5)), columns=[f"feature_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 50))
        metadata = {"batch_index": 0, "drift_type": "none"}

        # Should handle unfitted model gracefully
        result = learner.learn_batch(X, y, metadata)

        assert "samples_seen" in result
        assert result["samples_seen"] == 50

    def test_learn_batch_fitted_model(self, small_classification_data, adaptive_ensemble_detector):
        """Test learning with fitted model."""
        X, y = small_classification_data

        # Fit the model first
        adaptive_ensemble_detector.fit(X[:800], y[:800])

        evaluator = PrequentialEvaluator(window_size=200)
        learner = OnlineLearner(
            adaptive_ensemble_detector,
            evaluator,
            update_frequency=100,
            evaluation_frequency=100,
        )

        # Learn from batch
        batch_X = X[800:900]
        batch_y = y[800:900]
        metadata = {"batch_index": 0, "drift_type": "none"}

        result = learner.learn_batch(batch_X, batch_y, metadata)

        assert result["samples_seen"] == 100
        assert "batch_size" in result

    def test_periodic_updates(self, small_classification_data, adaptive_ensemble_detector):
        """Test periodic model updates."""
        X, y = small_classification_data

        # Fit the model first
        adaptive_ensemble_detector.fit(X[:500], y[:500])

        learner = OnlineLearner(
            adaptive_ensemble_detector,
            PrequentialEvaluator(),
            update_frequency=50,  # Update every 50 samples
            evaluation_frequency=100,
        )

        # Process multiple batches
        batch_size = 30
        for i in range(3):
            start_idx = 500 + i * batch_size
            end_idx = start_idx + batch_size

            batch_X = X[start_idx:end_idx]
            batch_y = y[start_idx:end_idx]
            metadata = {"batch_index": i, "drift_type": "none"}

            learner.learn_batch(batch_X, batch_y, metadata)

        # Should have triggered updates
        assert learner.samples_seen == 3 * batch_size
        assert len(learner.performance_history) == 3

    def test_performance_summary(self, adaptive_ensemble_detector):
        """Test performance summary generation."""
        learner = OnlineLearner(adaptive_ensemble_detector, PrequentialEvaluator())

        # Initially empty
        summary = learner.get_performance_summary()
        expected_keys = ["total_samples_seen", "total_batches_processed", "recent_accuracy"]
        for key in expected_keys:
            assert key in summary

        # Add some performance history
        learner.performance_history = [
            {"accuracy": 0.8, "samples_seen": 100},
            {"accuracy": 0.85, "samples_seen": 200},
            {"accuracy": 0.9, "samples_seen": 300},
        ]

        summary = learner.get_performance_summary()
        assert summary["total_batches_processed"] == 3
        assert "recent_accuracy" in summary


class TestEnsembleTrainer:
    """Test suite for EnsembleTrainer class."""

    def test_init(self, test_config):
        """Test EnsembleTrainer initialization."""
        trainer = EnsembleTrainer(
            model_config=test_config.model.__dict__,
            training_config=test_config.training.__dict__,
            mlflow_config=test_config.mlflow.__dict__,
        )

        assert trainer.model is None
        assert trainer.feature_processor is None
        assert isinstance(trainer.checkpoint_dir, Path)

    @patch("mlflow.create_experiment")
    @patch("mlflow.set_experiment")
    def test_mlflow_setup(self, mock_set_exp, mock_create_exp, test_config):
        """Test MLflow setup."""
        trainer = EnsembleTrainer(
            model_config=test_config.model.__dict__,
            training_config=test_config.training.__dict__,
            mlflow_config=test_config.mlflow.__dict__,
        )

        # Should have attempted to create and set experiment
        mock_create_exp.assert_called_once()
        mock_set_exp.assert_called_once()

    def test_fit_with_validation_split(self, small_classification_data, test_config, temp_dir):
        """Test fitting with automatic validation split."""
        X, y = small_classification_data

        # Update config for testing
        config_dict = test_config.__dict__
        config_dict["training"]["checkpoint_dir"] = str(temp_dir)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.log_metric"):
                        with patch("mlflow.sklearn.log_model"):
                            trainer = EnsembleTrainer(
                                model_config=config_dict["model"].__dict__,
                                training_config=config_dict["training"].__dict__,
                                mlflow_config=config_dict["mlflow"].__dict__,
                            )

                            model = trainer.fit(X, y)

                            assert_is_fitted(model)
                            assert trainer.feature_processor is not None
                            assert trainer.feature_processor.is_fitted

    def test_fit_with_provided_validation(self, small_classification_data, test_config, temp_dir):
        """Test fitting with provided validation data."""
        X, y = small_classification_data

        # Split data manually
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Update config for testing
        config_dict = test_config.__dict__
        config_dict["training"]["checkpoint_dir"] = str(temp_dir)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.log_metric"):
                        with patch("mlflow.sklearn.log_model"):
                            trainer = EnsembleTrainer(
                                model_config=config_dict["model"].__dict__,
                                training_config=config_dict["training"].__dict__,
                                mlflow_config=config_dict["mlflow"].__dict__,
                            )

                            model = trainer.fit(X_train, y_train, X_val, y_val)

                            assert_is_fitted(model)

    def test_streaming_training(self, streaming_data, test_config, temp_dir):
        """Test streaming training mode."""
        # Update config for testing
        config_dict = test_config.__dict__
        config_dict["training"]["checkpoint_dir"] = str(temp_dir)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.log_metric"):
                        with patch("mlflow.sklearn.log_model"):
                            trainer = EnsembleTrainer(
                                model_config=config_dict["model"].__dict__,
                                training_config=config_dict["training"].__dict__,
                                mlflow_config=config_dict["mlflow"].__dict__,
                            )

                            result = trainer.train_streaming(
                                data_loader=streaming_data,
                                max_samples=500,  # Limit for testing
                                save_frequency=2,  # Save frequently for testing
                            )

                            assert isinstance(result, dict)
                            assert "total_samples" in result
                            assert "total_batches" in result
                            assert "training_time" in result
                            assert result["total_samples"] <= 500

                            # Check that model was created and trained
                            assert trainer.model is not None
                            assert trainer.online_learner is not None

    def test_checkpoint_save_load(self, small_classification_data, test_config, temp_dir):
        """Test checkpoint saving and loading."""
        X, y = small_classification_data

        config_dict = test_config.__dict__
        config_dict["training"]["checkpoint_dir"] = str(temp_dir)

        # Train model
        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.log_metric"):
                        with patch("mlflow.sklearn.log_model"):
                            with patch("mlflow.log_artifact"):
                                trainer = EnsembleTrainer(
                                    model_config=config_dict["model"].__dict__,
                                    training_config=config_dict["training"].__dict__,
                                    mlflow_config=config_dict["mlflow"].__dict__,
                                )

                                trainer.fit(X[:800], y[:800])

                                # Save checkpoint manually
                                trainer._save_checkpoint("test_checkpoint")

                                checkpoint_path = temp_dir / "test_checkpoint.pkl"
                                assert checkpoint_path.exists()

                                # Create new trainer and load checkpoint
                                new_trainer = EnsembleTrainer(
                                    model_config=config_dict["model"].__dict__,
                                    training_config=config_dict["training"].__dict__,
                                    mlflow_config=config_dict["mlflow"].__dict__,
                                )

                                new_trainer.load_checkpoint(str(checkpoint_path))

                                assert new_trainer.model is not None
                                assert new_trainer.feature_processor is not None

                                # Test prediction with loaded model
                                predictions, probabilities = new_trainer.predict(X[800:])
                                assert len(predictions) == len(X) - 800

    def test_predict_without_model(self, small_classification_data, test_config):
        """Test prediction without trained model."""
        X, _ = small_classification_data

        trainer = EnsembleTrainer(
            model_config=test_config.model.__dict__,
            training_config=test_config.training.__dict__,
        )

        with pytest.raises(ValueError, match="Model must be trained"):
            trainer.predict(X)

    def test_compute_metrics(self, sample_predictions, test_config):
        """Test metrics computation."""
        trainer = EnsembleTrainer(
            model_config=test_config.model.__dict__,
            training_config=test_config.training.__dict__,
        )

        # Binary classification metrics
        y_true = pd.Series(np.random.randint(0, 2, 100))
        y_pred = np.random.randint(0, 2, 100)
        y_proba = np.random.random((100, 2))
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize

        metrics = trainer._compute_metrics(y_true, y_pred, y_proba)

        expected_metrics = ["accuracy", "precision", "recall", "f1", "auc", "log_loss"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

    def test_model_summary(self, small_classification_data, test_config, temp_dir):
        """Test model summary generation."""
        X, y = small_classification_data

        config_dict = test_config.__dict__
        config_dict["training"]["checkpoint_dir"] = str(temp_dir)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.log_metric"):
                        with patch("mlflow.sklearn.log_model"):
                            trainer = EnsembleTrainer(
                                model_config=config_dict["model"].__dict__,
                                training_config=config_dict["training"].__dict__,
                                mlflow_config=config_dict["mlflow"].__dict__,
                            )

                            trainer.fit(X, y)

                            summary = trainer.get_model_summary()

                            assert isinstance(summary, dict)
                            assert "model_type" in summary
                            assert "n_base_models" in summary
                            assert "base_model_names" in summary
                            assert "current_weights" in summary

    def test_model_summary_empty(self, test_config):
        """Test model summary with no trained model."""
        trainer = EnsembleTrainer(
            model_config=test_config.model.__dict__,
            training_config=test_config.training.__dict__,
        )

        summary = trainer.get_model_summary()
        assert summary == {}

    @patch("mlflow.log_metric")
    def test_mlflow_logging_during_streaming(self, mock_log_metric, streaming_data, test_config, temp_dir):
        """Test MLflow logging during streaming training."""
        config_dict = test_config.__dict__
        config_dict["training"]["checkpoint_dir"] = str(temp_dir)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.sklearn.log_model"):
                        trainer = EnsembleTrainer(
                            model_config=config_dict["model"].__dict__,
                            training_config=config_dict["training"].__dict__,
                            mlflow_config=config_dict["mlflow"].__dict__,
                        )

                        trainer.train_streaming(streaming_data, max_samples=300)

                        # Should have logged metrics during training
                        assert mock_log_metric.call_count > 0

    def test_invalid_checkpoint_loading(self, test_config, temp_dir):
        """Test loading invalid checkpoint."""
        trainer = EnsembleTrainer(
            model_config=test_config.model.__dict__,
            training_config=test_config.training.__dict__,
        )

        # Try to load non-existent checkpoint
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(str(temp_dir / "nonexistent.pkl"))

    def test_feature_processor_integration(self, data_with_missing_values, test_config, temp_dir):
        """Test feature processor integration."""
        X, y = data_with_missing_values

        config_dict = test_config.__dict__
        config_dict["training"]["checkpoint_dir"] = str(temp_dir)

        with patch("mlflow.start_run"):
            with patch("mlflow.log_params"):
                with patch("mlflow.log_metrics"):
                    with patch("mlflow.log_metric"):
                        with patch("mlflow.sklearn.log_model"):
                            trainer = EnsembleTrainer(
                                model_config=config_dict["model"].__dict__,
                                training_config=config_dict["training"].__dict__,
                                mlflow_config=config_dict["mlflow"].__dict__,
                            )

                            # Fit should handle missing values
                            model = trainer.fit(X, y)

                            assert_is_fitted(model)
                            assert trainer.feature_processor is not None

                            # Predict should also handle missing values
                            predictions, probabilities = trainer.predict(X[:100])
                            assert len(predictions) == 100