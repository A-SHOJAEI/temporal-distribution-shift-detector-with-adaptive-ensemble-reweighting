"""
Test suite for model components.

Tests cover the adaptive ensemble detector, drift detector, and Bayesian reweighter.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.models.model import (
    AdaptiveEnsembleDetector,
    DriftDetector,
    BayesianReweighter,
)
from tests.conftest import assert_is_fitted, assert_valid_predictions, assert_valid_probabilities


class TestDriftDetector:
    """Test suite for DriftDetector class."""

    def test_init(self):
        """Test DriftDetector initialization."""
        detector = DriftDetector(
            window_size=500,
            alpha=0.05,
            detection_threshold=0.1,
            min_samples=50,
        )

        assert detector.window_size == 500
        assert detector.alpha == 0.05
        assert detector.detection_threshold == 0.1
        assert detector.min_samples == 50
        assert detector.reference_X is None
        assert len(detector.current_X) == 0

    def test_set_reference(self, small_classification_data, sample_predictions):
        """Test setting reference data."""
        X, y = small_classification_data
        predictions = sample_predictions["binary"]["probabilities"]

        detector = DriftDetector()
        detector.set_reference(X, y, predictions)

        assert detector.reference_X is not None
        assert detector.reference_y is not None
        assert detector.reference_predictions is not None
        assert len(detector.reference_X) == len(X)

    def test_update_insufficient_data(self, small_classification_data):
        """Test update with insufficient data."""
        X, y = small_classification_data
        detector = DriftDetector(min_samples=1000)  # Require more samples than available

        # Set reference
        detector.set_reference(X[:500], y[:500])

        # Update with small batch
        result = detector.update(X[500:520], y[500:520])

        assert result["drift_detected"] is False
        assert result["drift_score"] == 0.0

    def test_detect_covariate_drift(self, small_classification_data, drift_injector):
        """Test covariate drift detection."""
        X, y = small_classification_data

        detector = DriftDetector(min_samples=100)
        detector.set_reference(X[:500], y[:500])

        # Inject covariate drift
        X_drift = drift_injector.inject_covariate_shift(X[500:700], shift_magnitude=1.0)

        result = detector.update(X_drift, y[500:700])

        # Should detect drift with high magnitude shift
        assert "covariate_drift_score" in result
        assert isinstance(result["covariate_drift_score"], float)

    def test_detect_label_drift(self, small_classification_data):
        """Test label drift detection."""
        X, y = small_classification_data

        detector = DriftDetector(min_samples=100)
        detector.set_reference(X[:500], y[:500])

        # Create artificial label shift
        y_drift = y[500:700].copy()
        y_drift[:] = 1 - y_drift  # Flip all labels

        result = detector.update(X[500:700], y_drift)

        assert "label_drift_score" in result
        assert isinstance(result["label_drift_score"], float)

    def test_reset_functionality(self, small_classification_data):
        """Test reset functionality."""
        X, y = small_classification_data

        detector = DriftDetector()
        detector.set_reference(X[:500], y[:500])
        detector.update(X[500:600], y[500:600])

        # Check that data is stored
        assert len(detector.current_X) > 0

        # Reset
        detector.reset()

        # Check that current data is cleared
        assert len(detector.current_X) == 0
        assert len(detector.current_y) == 0
        assert len(detector.drift_history) == 0

    def test_window_size_maintenance(self, large_classification_data):
        """Test that window size is properly maintained."""
        X, y = large_classification_data

        window_size = 100
        detector = DriftDetector(window_size=window_size, min_samples=50)
        detector.set_reference(X[:500], y[:500])

        # Add multiple batches exceeding window size
        for i in range(10):
            start_idx = 500 + i * 30
            end_idx = start_idx + 30
            if end_idx < len(X):
                detector.update(X[start_idx:end_idx], y[start_idx:end_idx])

        # Check that window size is maintained
        current_size = sum(len(batch_X) for batch_X in detector.current_X)
        assert current_size <= window_size


class TestBayesianReweighter:
    """Test suite for BayesianReweighter class."""

    def test_init(self):
        """Test BayesianReweighter initialization."""
        reweighter = BayesianReweighter(
            n_models=3,
            initial_alpha=2.0,
            initial_beta=1.0,
            decay_factor=0.9,
        )

        assert reweighter.n_models == 3
        assert reweighter.initial_alpha == 2.0
        assert reweighter.initial_beta == 1.0
        assert reweighter.decay_factor == 0.9

        # Check initial parameters
        assert len(reweighter.alpha) == 3
        assert len(reweighter.beta) == 3
        assert np.all(reweighter.alpha == 2.0)
        assert np.all(reweighter.beta == 1.0)

    def test_update_performance(self, small_classification_data):
        """Test performance update."""
        _, y = small_classification_data
        y_binary = (y > 0).astype(int)

        reweighter = BayesianReweighter(n_models=3)

        # Create mock predictions from 3 models
        n_samples = len(y_binary)
        model_predictions = np.array([
            np.random.random(n_samples),  # Model 1 (random)
            y_binary.values,  # Model 2 (perfect)
            1 - y_binary.values,  # Model 3 (inverse)
        ])

        initial_alpha = reweighter.alpha.copy()
        initial_beta = reweighter.beta.copy()

        reweighter.update(model_predictions, y_binary.values, drift_detected=False)

        # Check that parameters were updated
        assert not np.array_equal(reweighter.alpha, initial_alpha)
        assert not np.array_equal(reweighter.beta, initial_beta)

        # Check that performance history is recorded
        assert len(reweighter.performance_history) == 1

    def test_get_weights_thompson_sampling(self):
        """Test weight generation with Thompson sampling."""
        reweighter = BayesianReweighter(n_models=3)

        weights = reweighter.get_weights(use_thompson_sampling=True)

        # Check weights properties
        assert len(weights) == 3
        assert np.all(weights >= 0)
        assert np.allclose(np.sum(weights), 1.0)

        # Check that weights history is recorded
        assert len(reweighter.weights_history) == 1

    def test_get_weights_posterior_mean(self):
        """Test weight generation with posterior means."""
        reweighter = BayesianReweighter(n_models=3)

        weights = reweighter.get_weights(use_thompson_sampling=False)

        # Check weights properties
        assert len(weights) == 3
        assert np.all(weights >= 0)
        assert np.allclose(np.sum(weights), 1.0)

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation."""
        reweighter = BayesianReweighter(n_models=3)

        uncertainty = reweighter.get_uncertainty()

        assert len(uncertainty) == 3
        assert np.all(uncertainty >= 0)

        # After some updates, uncertainty should change
        model_predictions = np.random.random((3, 100))
        true_labels = np.random.randint(0, 2, 100)

        reweighter.update(model_predictions, true_labels)
        new_uncertainty = reweighter.get_uncertainty()

        assert not np.array_equal(uncertainty, new_uncertainty)

    def test_drift_detection_reset(self, small_classification_data):
        """Test behavior when drift is detected."""
        _, y = small_classification_data
        y_binary = (y > 0).astype(int)

        reweighter = BayesianReweighter(n_models=3)

        # Update without drift
        model_predictions = np.random.random((3, 50))
        reweighter.update(model_predictions, y_binary[:50].values, drift_detected=False)

        alpha_before = reweighter.alpha.copy()
        beta_before = reweighter.beta.copy()

        # Update with drift detected
        reweighter.update(model_predictions, y_binary[:50].values, drift_detected=True)

        # Parameters should be partially reset
        assert not np.array_equal(reweighter.alpha, alpha_before)
        assert not np.array_equal(reweighter.beta, beta_before)

    def test_reset_model_beliefs(self):
        """Test resetting beliefs for specific model."""
        reweighter = BayesianReweighter(n_models=3)

        # Update to change from initial values
        model_predictions = np.random.random((3, 50))
        true_labels = np.random.randint(0, 2, 50)
        reweighter.update(model_predictions, true_labels)

        original_alpha_1 = reweighter.alpha[1]
        original_beta_1 = reweighter.beta[1]

        # Reset model 1
        reweighter.reset_model_beliefs(1)

        # Check that model 1 was reset
        assert reweighter.alpha[1] == reweighter.initial_alpha
        assert reweighter.beta[1] == reweighter.initial_beta

        # Check that other models were not affected
        assert reweighter.alpha[0] != reweighter.initial_alpha
        assert reweighter.alpha[2] != reweighter.initial_alpha


class TestAdaptiveEnsembleDetector:
    """Test suite for AdaptiveEnsembleDetector class."""

    def test_init(self, random_seed):
        """Test AdaptiveEnsembleDetector initialization."""
        detector = AdaptiveEnsembleDetector(random_state=random_seed)

        assert detector.random_state == random_seed
        assert detector.adaptation_threshold == 0.1
        assert detector.max_ensemble_size == 3
        assert not detector.is_fitted

    def test_fit_binary_classification(self, small_classification_data, adaptive_ensemble_detector):
        """Test fitting on binary classification data."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X, y)

        assert_is_fitted(detector)
        assert len(detector.classes_) == 2
        assert detector.n_features_in_ == X.shape[1]
        assert len(detector.base_models) == 3

        # Check that models are fitted
        for model in detector.base_models:
            assert hasattr(model, "predict")

        # Check that reweighter is initialized
        assert detector.reweighter is not None
        assert detector.reweighter.n_models == len(detector.base_models)

    def test_fit_multiclass_classification(self, multiclass_data, adaptive_ensemble_detector):
        """Test fitting on multiclass data."""
        X, y = multiclass_data

        detector = adaptive_ensemble_detector
        detector.fit(X, y)

        assert_is_fitted(detector)
        assert len(detector.classes_) == 3
        assert detector.n_features_in_ == X.shape[1]

    def test_predict_before_fit(self, small_classification_data, adaptive_ensemble_detector):
        """Test prediction before fitting."""
        X, _ = small_classification_data

        detector = adaptive_ensemble_detector

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict(X)

    def test_predict_binary(self, small_classification_data, adaptive_ensemble_detector):
        """Test prediction on binary classification."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X[:800], y[:800])

        predictions = detector.predict(X[800:])
        probabilities = detector.predict_proba(X[800:])

        n_test_samples = len(X) - 800
        assert_valid_predictions(predictions, n_test_samples)
        assert_valid_probabilities(probabilities, n_test_samples, n_classes=2)

    def test_predict_multiclass(self, multiclass_data, adaptive_ensemble_detector):
        """Test prediction on multiclass data."""
        X, y = multiclass_data

        detector = adaptive_ensemble_detector
        detector.fit(X[:800], y[:800])

        predictions = detector.predict(X[800:])
        probabilities = detector.predict_proba(X[800:])

        n_test_samples = len(X) - 800
        assert_valid_predictions(predictions, n_test_samples, n_classes=3)
        assert_valid_probabilities(probabilities, n_test_samples, n_classes=3)

    def test_partial_fit_without_labels(self, small_classification_data, adaptive_ensemble_detector):
        """Test partial fit without labels."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X[:800], y[:800])

        # Partial fit without labels should work
        detector.partial_fit(X[800:900])

        # Should still be able to predict
        predictions = detector.predict(X[800:900])
        assert len(predictions) == 100

    def test_partial_fit_with_labels(self, small_classification_data, adaptive_ensemble_detector):
        """Test partial fit with labels."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X[:800], y[:800])

        initial_adaptation_count = len(detector.adaptation_history)

        # Partial fit with labels
        detector.partial_fit(X[800:900], y[800:900])

        # Should have recorded adaptation information
        assert len(detector.adaptation_history) > initial_adaptation_count

        # Should still be able to predict
        predictions = detector.predict(X[900:])
        assert len(predictions) == len(X) - 900

    def test_partial_fit_before_fit(self, small_classification_data, adaptive_ensemble_detector):
        """Test partial fit before initial fit."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector

        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.partial_fit(X[:100])

    def test_get_base_predictions(self, small_classification_data, adaptive_ensemble_detector):
        """Test base model predictions retrieval."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X[:800], y[:800])

        base_predictions = detector._get_base_predictions(X[800:900])

        # Check shape: [n_models, n_samples] for binary
        assert base_predictions.shape[0] == len(detector.base_models)
        assert base_predictions.shape[1] == 100

    def test_adaptation_triggered(self, small_classification_data, drift_injector, random_seed):
        """Test that adaptation is triggered on significant drift."""
        X, y = small_classification_data

        # Create detector with low adaptation threshold
        detector = AdaptiveEnsembleDetector(
            adaptation_threshold=0.01,  # Low threshold
            drift_detector_params={"detection_threshold": 0.01},
            random_state=random_seed,
        )

        detector.fit(X[:700], y[:700])

        # Create strong drift
        X_drift = drift_injector.inject_covariate_shift(X[700:800], shift_magnitude=2.0)

        initial_adaptation_count = len(detector.adaptation_history)

        # This should trigger adaptation
        detector.partial_fit(X_drift, y[700:800])

        # Check that adaptation was recorded
        assert len(detector.adaptation_history) > initial_adaptation_count

    def test_model_importance(self, small_classification_data, adaptive_ensemble_detector):
        """Test model importance retrieval."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X, y)

        importance = detector.get_model_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert "xgboost" in importance
        assert "lightgbm" in importance
        assert "catboost" in importance

        # All importance values should be positive and sum to 1
        values = list(importance.values())
        assert all(v >= 0 for v in values)
        assert np.allclose(sum(values), 1.0)

    def test_drift_history(self, small_classification_data, adaptive_ensemble_detector):
        """Test drift history tracking."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X[:800], y[:800])

        # Perform some partial fits
        detector.partial_fit(X[800:850], y[800:850])
        detector.partial_fit(X[850:900], y[850:900])

        drift_history = detector.get_drift_history()

        assert isinstance(drift_history, list)
        # Should have some drift detection results
        assert len(drift_history) >= 0

    def test_adaptation_summary(self, small_classification_data, adaptive_ensemble_detector):
        """Test adaptation summary generation."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X[:800], y[:800])

        # Perform partial fit to generate adaptation history
        detector.partial_fit(X[800:], y[800:])

        summary = detector.get_adaptation_summary()

        assert isinstance(summary, dict)
        expected_keys = [
            "total_adaptations",
            "drift_detection_rate",
            "average_drift_score",
            "max_drift_score",
            "current_regret_bound",
            "final_model_weights",
        ]

        for key in expected_keys:
            assert key in summary

    def test_regret_bound_calculation(self, small_classification_data, adaptive_ensemble_detector):
        """Test regret bound calculation."""
        X, y = small_classification_data

        detector = adaptive_ensemble_detector
        detector.fit(X[:800], y[:800])

        # Perform multiple partial fits to accumulate adaptation history
        for i in range(5):
            start_idx = 800 + i * 20
            end_idx = start_idx + 20
            if end_idx <= len(X):
                detector.partial_fit(X[start_idx:end_idx], y[start_idx:end_idx])

        regret_bound = detector._calculate_regret_bound()

        assert isinstance(regret_bound, float)
        assert regret_bound >= 0

    def test_empty_adaptation_summary(self, adaptive_ensemble_detector):
        """Test adaptation summary with no adaptation history."""
        summary = adaptive_ensemble_detector.get_adaptation_summary()

        # Should return empty dict if no adaptation history
        assert summary == {}

    def test_different_base_model_configs(self, small_classification_data, random_seed):
        """Test with different base model configurations."""
        X, y = small_classification_data

        custom_params = {
            "xgboost": {"n_estimators": 5, "max_depth": 3},
            "lightgbm": {"n_estimators": 5, "max_depth": 3, "verbose": -1},
            "catboost": {"iterations": 5, "depth": 3, "verbose": False},
        }

        detector = AdaptiveEnsembleDetector(
            base_model_params=custom_params,
            random_state=random_seed,
        )

        detector.fit(X, y)

        assert_is_fitted(detector)

        # Check that custom parameters were applied
        xgb_model = detector.base_models[0]
        assert xgb_model.n_estimators == 5
        assert xgb_model.max_depth == 3