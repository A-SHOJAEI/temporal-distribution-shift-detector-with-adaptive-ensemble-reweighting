"""
Test fixtures for the temporal distribution shift detector test suite.

This module provides common test fixtures and utilities used across
all test modules.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.models.model import (
    AdaptiveEnsembleDetector,
    DriftDetector,
    BayesianReweighter,
)
from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.loader import (
    DataLoader,
    DriftDataLoader,
)
from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.preprocessing import (
    FeatureProcessor,
    DriftInjector,
)
from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.utils.config import Config


@pytest.fixture
def random_seed():
    """Random seed for reproducible tests."""
    return 42


@pytest.fixture
def small_classification_data(random_seed):
    """Generate small classification dataset for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=random_seed,
    )

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def large_classification_data(random_seed):
    """Generate larger classification dataset for testing."""
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=random_seed,
    )

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def multiclass_data(random_seed):
    """Generate multiclass classification dataset for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=random_seed,
    )

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    return X_df, y_series


@pytest.fixture
def data_with_missing_values(small_classification_data, random_seed):
    """Create dataset with missing values for preprocessing tests."""
    X, y = small_classification_data

    # Randomly introduce missing values
    np.random.seed(random_seed)
    missing_mask = np.random.random(X.shape) < 0.1  # 10% missing

    X_missing = X.copy()
    X_missing[missing_mask] = np.nan

    return X_missing, y


@pytest.fixture
def drift_schedule():
    """Sample drift schedule for testing."""
    return {
        5: "covariate_shift",
        15: "concept_drift",
        25: "label_shift",
        35: "combined",
    }


@pytest.fixture
def streaming_data(small_classification_data, drift_schedule, random_seed):
    """Create streaming data loader for testing."""
    X, y = small_classification_data

    data_loader = DriftDataLoader(
        X=X,
        y=y,
        batch_size=100,
        drift_schedule=drift_schedule,
        random_state=random_seed,
    )

    return data_loader


@pytest.fixture
def drift_detector():
    """Create drift detector for testing."""
    return DriftDetector(
        window_size=500,
        alpha=0.05,
        detection_threshold=0.1,
        min_samples=50,
    )


@pytest.fixture
def bayesian_reweighter():
    """Create Bayesian reweighter for testing."""
    return BayesianReweighter(
        n_models=3,
        initial_alpha=1.0,
        initial_beta=1.0,
        decay_factor=0.95,
        exploration_factor=0.1,
    )


@pytest.fixture
def feature_processor(random_seed):
    """Create feature processor for testing."""
    return FeatureProcessor(
        scaler_type="standard",
        handle_missing="median",
        random_state=random_seed,
    )


@pytest.fixture
def drift_injector(random_seed):
    """Create drift injector for testing."""
    return DriftInjector(random_state=random_seed)


@pytest.fixture
def adaptive_ensemble_detector(random_seed):
    """Create adaptive ensemble detector for testing."""
    return AdaptiveEnsembleDetector(
        drift_detector_params={
            "window_size": 500,
            "detection_threshold": 0.1,
        },
        reweighter_params={
            "initial_alpha": 1.0,
            "initial_beta": 1.0,
        },
        base_model_params={
            "xgboost": {"n_estimators": 10, "random_state": random_seed},
            "lightgbm": {"n_estimators": 10, "random_state": random_seed, "verbose": -1},
            "catboost": {"iterations": 10, "random_seed": random_seed, "verbose": False},
        },
        random_state=random_seed,
    )


@pytest.fixture
def test_config(random_seed):
    """Create test configuration."""
    config = Config()
    config.random_seed = random_seed

    # Reduce model complexity for faster tests
    config.model.base_models.xgboost["n_estimators"] = 10
    config.model.base_models.lightgbm["n_estimators"] = 10
    config.model.base_models.catboost["iterations"] = 10

    # Reduce window sizes for faster tests
    config.model.drift_detector.window_size = 500
    config.evaluation.prequential_window_size = 500

    # Set test batch size
    config.data.batch_size = 100

    return config


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_predictions(random_seed):
    """Generate sample predictions for testing."""
    np.random.seed(random_seed)

    # Binary predictions
    binary_proba = np.random.random(100)
    binary_pred = (binary_proba > 0.5).astype(int)

    # Multiclass predictions
    n_samples, n_classes = 100, 3
    multiclass_proba = np.random.random((n_samples, n_classes))
    multiclass_proba = multiclass_proba / multiclass_proba.sum(axis=1, keepdims=True)
    multiclass_pred = np.argmax(multiclass_proba, axis=1)

    return {
        "binary": {
            "predictions": binary_pred,
            "probabilities": binary_proba,
        },
        "multiclass": {
            "predictions": multiclass_pred,
            "probabilities": multiclass_proba,
        }
    }


@pytest.fixture(autouse=True)
def set_random_seeds(random_seed):
    """Set random seeds for all relevant libraries."""
    np.random.seed(random_seed)

    # Set environment variable for reproducible hashing
    import os
    os.environ["PYTHONHASHSEED"] = str(random_seed)


# Utility functions for tests

def assert_is_fitted(estimator):
    """Assert that an estimator is fitted."""
    assert hasattr(estimator, "is_fitted"), "Estimator should have is_fitted attribute"
    assert estimator.is_fitted, "Estimator should be fitted"


def assert_valid_predictions(predictions, n_samples, n_classes=None):
    """Assert that predictions are valid."""
    assert len(predictions) == n_samples, f"Expected {n_samples} predictions, got {len(predictions)}"

    if n_classes is not None:
        assert all(0 <= pred < n_classes for pred in predictions), \
            f"Predictions should be in range [0, {n_classes})"
    else:
        # Binary classification
        assert all(pred in [0, 1] for pred in predictions), \
            "Binary predictions should be 0 or 1"


def assert_valid_probabilities(probabilities, n_samples, n_classes=None):
    """Assert that probabilities are valid."""
    assert len(probabilities) == n_samples, \
        f"Expected {n_samples} probability vectors, got {len(probabilities)}"

    if n_classes is not None:
        # Multiclass
        assert probabilities.shape == (n_samples, n_classes), \
            f"Expected shape ({n_samples}, {n_classes}), got {probabilities.shape}"

        # Check probabilities sum to 1
        prob_sums = probabilities.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-6), \
            "Probabilities should sum to 1"
    else:
        # Binary classification
        assert probabilities.ndim == 1 or probabilities.shape[1] == 2, \
            "Binary probabilities should be 1D or have 2 columns"

    # Check probability values are valid
    assert np.all(probabilities >= 0), "Probabilities should be non-negative"
    assert np.all(probabilities <= 1), "Probabilities should be <= 1"


def create_batch_data(X, y, batch_size=100, n_batches=5, random_state=42):
    """Create batched data for streaming tests."""
    np.random.seed(random_state)

    batches = []
    total_samples = len(X)
    indices = np.random.permutation(total_samples)

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_samples)

        if start_idx >= total_samples:
            break

        batch_indices = indices[start_idx:end_idx]
        batch_X = X.iloc[batch_indices].reset_index(drop=True)
        batch_y = y.iloc[batch_indices].reset_index(drop=True)

        metadata = {
            "batch_index": i,
            "drift_type": "none",
            "start_idx": start_idx,
            "end_idx": end_idx,
        }

        batches.append((batch_X, batch_y, metadata))

    return batches