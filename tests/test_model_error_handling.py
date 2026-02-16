"""
Test suite for error handling in model components.

Tests cover input validation and error conditions for all model components.
"""

import pytest
import numpy as np
import pandas as pd

from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.models.model import (
    AdaptiveEnsembleDetector,
    DriftDetector,
    BayesianReweighter,
)


class TestDriftDetectorErrorHandling:
    """Test error handling for DriftDetector class."""

    def test_init_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test negative window_size
        with pytest.raises(ValueError, match="window_size must be positive"):
            DriftDetector(window_size=-1)

        # Test zero window_size
        with pytest.raises(ValueError, match="window_size must be positive"):
            DriftDetector(window_size=0)

        # Test invalid alpha
        with pytest.raises(ValueError, match="alpha must be in"):
            DriftDetector(alpha=0)

        with pytest.raises(ValueError, match="alpha must be in"):
            DriftDetector(alpha=1.5)

        # Test negative detection_threshold
        with pytest.raises(ValueError, match="detection_threshold must be non-negative"):
            DriftDetector(detection_threshold=-0.1)

        # Test invalid min_samples
        with pytest.raises(ValueError, match="min_samples must be positive"):
            DriftDetector(min_samples=0)

    def test_set_reference_invalid_inputs(self):
        """Test set_reference with invalid inputs."""
        detector = DriftDetector()

        # Test empty DataFrame
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.set_reference(pd.DataFrame())

        # Test None input
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.set_reference(None)

        # Test wrong type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            detector.set_reference([[1, 2], [3, 4]])

        # Test mismatched lengths
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([0, 1])  # Wrong length

        with pytest.raises(ValueError, match="y length .* doesn't match X length"):
            detector.set_reference(X, y)

        # Test mismatched predictions length
        predictions = np.array([0.1, 0.2])  # Wrong length
        with pytest.raises(ValueError, match="predictions length .* doesn't match X length"):
            detector.set_reference(X, predictions=predictions)

    def test_update_without_reference(self):
        """Test update without setting reference data first."""
        detector = DriftDetector()
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

        with pytest.raises(RuntimeError, match="Reference data not set"):
            detector.update(X)

    def test_update_invalid_inputs(self, small_classification_data):
        """Test update with invalid inputs."""
        X, y = small_classification_data
        detector = DriftDetector()
        detector.set_reference(X[:100], y[:100])

        # Test empty DataFrame
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.update(pd.DataFrame())

        # Test wrong type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            detector.update([[1, 2], [3, 4]])

        # Test incompatible columns
        X_incompatible = pd.DataFrame({'new_feature': [1, 2, 3]})
        with pytest.raises(ValueError, match="X contains unknown columns"):
            detector.update(X_incompatible)

        # Test mismatched lengths
        X_batch = X[100:103]
        y_batch = y[100:102]  # Wrong length
        with pytest.raises(ValueError, match="y length .* doesn't match X length"):
            detector.update(X_batch, y_batch)


class TestBayesianReweighterErrorHandling:
    """Test error handling for BayesianReweighter class."""

    def test_init_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test negative n_models
        with pytest.raises(ValueError, match="n_models must be positive"):
            BayesianReweighter(n_models=-1)

        # Test zero n_models
        with pytest.raises(ValueError, match="n_models must be positive"):
            BayesianReweighter(n_models=0)

        # Test negative initial_alpha
        with pytest.raises(ValueError, match="initial_alpha must be non-negative"):
            BayesianReweighter(n_models=3, initial_alpha=-1.0)

        # Test negative initial_beta
        with pytest.raises(ValueError, match="initial_beta must be non-negative"):
            BayesianReweighter(n_models=3, initial_beta=-1.0)

        # Test invalid decay_factor
        with pytest.raises(ValueError, match="decay_factor must be in"):
            BayesianReweighter(n_models=3, decay_factor=0)

        with pytest.raises(ValueError, match="decay_factor must be in"):
            BayesianReweighter(n_models=3, decay_factor=1.5)

        # Test negative exploration_factor
        with pytest.raises(ValueError, match="exploration_factor must be non-negative"):
            BayesianReweighter(n_models=3, exploration_factor=-0.1)

    def test_update_invalid_inputs(self):
        """Test update with invalid inputs."""
        reweighter = BayesianReweighter(n_models=3)

        # Test empty predictions
        with pytest.raises(ValueError, match="model_predictions cannot be empty"):
            reweighter.update([], np.array([0, 1]))

        # Test empty labels
        with pytest.raises(ValueError, match="true_labels cannot be empty"):
            reweighter.update(np.random.random((3, 10)), np.array([]))

        # Test wrong number of models
        with pytest.raises(ValueError, match="Expected 3 model predictions, got 2"):
            predictions = np.random.random((2, 10))  # Only 2 models instead of 3
            labels = np.random.randint(0, 2, 10)
            reweighter.update(predictions, labels)

        # Test mismatched prediction lengths
        predictions = [
            np.random.random(10),
            np.random.random(8),  # Wrong length
            np.random.random(10)
        ]
        labels = np.random.randint(0, 2, 10)

        with pytest.raises(ValueError, match="Model 1 predictions length .* doesn't match"):
            reweighter.update(predictions, labels)

    def test_reset_model_beliefs_invalid_index(self):
        """Test resetting beliefs with invalid model index."""
        reweighter = BayesianReweighter(n_models=3)

        # Test non-integer index
        with pytest.raises(TypeError, match="model_idx must be an integer"):
            reweighter.reset_model_beliefs("invalid")

        # Test negative index
        with pytest.raises(IndexError, match="model_idx -1 is out of range"):
            reweighter.reset_model_beliefs(-1)

        # Test index too large
        with pytest.raises(IndexError, match="model_idx 5 is out of range"):
            reweighter.reset_model_beliefs(5)


class TestAdaptiveEnsembleDetectorErrorHandling:
    """Test error handling for AdaptiveEnsembleDetector class."""

    def test_init_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test negative adaptation_threshold
        with pytest.raises(ValueError, match="adaptation_threshold must be non-negative"):
            AdaptiveEnsembleDetector(adaptation_threshold=-0.1)

        # Test invalid max_ensemble_size
        with pytest.raises(ValueError, match="max_ensemble_size must be positive"):
            AdaptiveEnsembleDetector(max_ensemble_size=0)

        # Test invalid random_state type
        with pytest.raises(TypeError, match="random_state must be an integer"):
            AdaptiveEnsembleDetector(random_state="invalid")

    def test_fit_invalid_inputs(self):
        """Test fit with invalid inputs."""
        detector = AdaptiveEnsembleDetector()

        # Test empty X
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.fit(pd.DataFrame(), pd.Series([1, 2, 3]))

        # Test empty y
        with pytest.raises(ValueError, match="y cannot be empty"):
            detector.fit(pd.DataFrame({'col': [1, 2, 3]}), pd.Series([]))

        # Test wrong types
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            detector.fit([[1, 2], [3, 4]], pd.Series([0, 1]))

        with pytest.raises(TypeError, match="y must be a pandas Series"):
            detector.fit(pd.DataFrame({'col': [1, 2]}), [0, 1])

        # Test mismatched lengths
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([0, 1])  # Wrong length

        with pytest.raises(ValueError, match="X and y must have the same length"):
            detector.fit(X, y)

    def test_predict_invalid_inputs(self, small_classification_data):
        """Test predict with invalid inputs."""
        X, y = small_classification_data
        detector = AdaptiveEnsembleDetector()
        detector.fit(X[:100], y[:100])

        # Test empty X
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.predict(pd.DataFrame())

        # Test wrong type
        with pytest.raises(TypeError, match="X must be a pandas DataFrame"):
            detector.predict([[1, 2], [3, 4]])

        # Test wrong number of features
        X_wrong_features = pd.DataFrame({'new_feature': [1, 2, 3]})
        with pytest.raises(ValueError, match="X has .* features, expected"):
            detector.predict(X_wrong_features)

        # Test missing features
        X_missing_features = X[['feature_0']].copy()  # Only keep first feature
        with pytest.raises(ValueError, match="Missing features in X"):
            detector.predict(X_missing_features)

    def test_predict_proba_invalid_inputs(self, small_classification_data):
        """Test predict_proba with same validation as predict."""
        X, y = small_classification_data
        detector = AdaptiveEnsembleDetector()
        detector.fit(X[:100], y[:100])

        # Test empty X
        with pytest.raises(ValueError, match="X cannot be empty"):
            detector.predict_proba(pd.DataFrame())

    def test_partial_fit_before_fit(self, small_classification_data):
        """Test partial_fit before calling fit."""
        X, y = small_classification_data
        detector = AdaptiveEnsembleDetector()

        with pytest.raises(ValueError, match="Model must be fitted before partial_fit"):
            detector.partial_fit(X[:10])

    def test_fit_failure_recovery(self, small_classification_data):
        """Test that failed fit properly resets state."""
        X, y = small_classification_data

        # Create detector with invalid parameters that will cause fit to fail
        detector = AdaptiveEnsembleDetector(
            base_model_params={
                "xgboost": {"invalid_param": "invalid_value"}  # This should cause an error
            }
        )

        # Fit should fail and is_fitted should remain False
        with pytest.raises(RuntimeError, match="Failed to fit AdaptiveEnsembleDetector"):
            detector.fit(X, y)

        assert not detector.is_fitted

    def test_edge_case_single_sample_batches(self, small_classification_data):
        """Test handling of very small batch sizes."""
        X, y = small_classification_data
        detector = AdaptiveEnsembleDetector()
        detector.fit(X[:100], y[:100])

        # Should handle single sample batches gracefully
        single_sample = X[100:101]
        single_label = y[100:101]

        # This should not raise an error
        detector.partial_fit(single_sample, single_label)
        prediction = detector.predict(single_sample)

        assert len(prediction) == 1

    def test_error_propagation_from_components(self, small_classification_data):
        """Test that errors from components are properly propagated."""
        X, y = small_classification_data
        detector = AdaptiveEnsembleDetector()
        detector.fit(X[:100], y[:100])

        # Corrupt the drift detector to cause an error
        detector.drift_detector.reference_X = None

        with pytest.raises(Exception):  # Should propagate the error from drift detector
            detector.partial_fit(X[100:110], y[100:110])