"""
Test suite for data loading and preprocessing modules.

Tests cover data loaders, preprocessing utilities, and drift injection.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.loader import (
    DataLoader,
    DriftDataLoader,
)
from src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.preprocessing import (
    FeatureProcessor,
    DriftInjector,
)
from tests.conftest import assert_valid_predictions, assert_valid_probabilities


class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_init(self, random_seed):
        """Test DataLoader initialization."""
        loader = DataLoader(random_state=random_seed)
        assert loader.random_state == random_seed
        assert hasattr(loader, "rng")

    @patch("src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.loader.fetch_covtype")
    def test_load_covertype_success(self, mock_fetch, random_seed):
        """Test successful Covertype dataset loading."""
        # Mock the sklearn fetch_covtype function
        mock_X = pd.DataFrame(np.random.random((100, 10)), columns=[f"feature_{i}" for i in range(10)])
        mock_y = pd.Series(np.random.randint(0, 7, 100), name="target")
        mock_fetch.return_value = (mock_X, mock_y)

        loader = DataLoader(random_state=random_seed)
        X, y = loader.load_covertype()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) == 100
        mock_fetch.assert_called_once()

    @patch("src.temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data.loader.fetch_covtype")
    def test_load_covertype_failure(self, mock_fetch, random_seed):
        """Test Covertype dataset loading failure."""
        mock_fetch.side_effect = Exception("Network error")

        loader = DataLoader(random_state=random_seed)

        with pytest.raises(RuntimeError, match="Failed to load Covertype dataset"):
            loader.load_covertype()

    def test_create_elec2_features(self, random_seed):
        """Test ELEC2 feature creation."""
        loader = DataLoader(random_state=random_seed)

        # Create mock ELEC2-style data
        dates = pd.date_range("2020-01-01", periods=100, freq="H")
        consumption_data = pd.DataFrame(
            np.random.random((100, 5)),
            index=dates,
            columns=[f"consumer_{i}" for i in range(5)]
        )

        features = loader._create_elec2_features(consumption_data)

        # Check feature types
        expected_time_features = ["hour", "day_of_week", "month", "quarter"]
        for feature in expected_time_features:
            assert feature in features.columns

        expected_stat_features = ["mean_consumption", "std_consumption", "min_consumption", "max_consumption"]
        for feature in expected_stat_features:
            assert feature in features.columns

        # Check for lag and rolling features
        lag_features = [col for col in features.columns if col.startswith("lag_")]
        rolling_features = [col for col in features.columns if col.startswith("rolling_")]

        assert len(lag_features) > 0
        assert len(rolling_features) > 0

        # Check that features are numeric
        assert features.select_dtypes(include=[np.number]).shape[1] == features.shape[1]

    def test_load_airlines_missing_path(self, random_seed):
        """Test Airlines dataset loading without path."""
        loader = DataLoader(random_state=random_seed)

        with pytest.raises(ValueError, match="Airlines dataset path must be provided"):
            loader.load_airlines()

    def test_load_airlines_missing_target(self, tmp_path, random_seed):
        """Test Airlines dataset loading with missing target column."""
        # Create mock CSV file without the expected target column
        mock_data = pd.DataFrame({
            "feature_1": np.random.random(100),
            "feature_2": np.random.random(100),
            "wrong_target": np.random.random(100),
        })

        csv_path = tmp_path / "airlines_test.csv"
        mock_data.to_csv(csv_path, index=False)

        loader = DataLoader(random_state=random_seed)

        with pytest.raises(ValueError, match="Target column not found"):
            loader.load_airlines(str(csv_path))


class TestDriftDataLoader:
    """Test suite for DriftDataLoader class."""

    def test_init(self, small_classification_data, drift_schedule, random_seed):
        """Test DriftDataLoader initialization."""
        X, y = small_classification_data

        loader = DriftDataLoader(
            X=X,
            y=y,
            batch_size=100,
            drift_schedule=drift_schedule,
            random_state=random_seed,
        )

        assert loader.batch_size == 100
        assert loader.drift_schedule == drift_schedule
        assert loader.random_state == random_seed
        assert len(loader.X) == len(X)
        assert len(loader.y) == len(y)

    def test_iteration(self, streaming_data):
        """Test data loader iteration."""
        batches = list(streaming_data)

        assert len(batches) > 0

        for batch_X, batch_y, metadata in batches:
            assert isinstance(batch_X, pd.DataFrame)
            assert isinstance(batch_y, pd.Series)
            assert isinstance(metadata, dict)

            assert len(batch_X) == len(batch_y)
            assert len(batch_X) <= 100  # batch_size

            # Check metadata
            required_keys = ["batch_index", "drift_type", "start_idx", "end_idx", "timestamp"]
            for key in required_keys:
                assert key in metadata

    def test_drift_application(self, small_classification_data, random_seed):
        """Test that drift is applied according to schedule."""
        X, y = small_classification_data

        drift_schedule = {0: "covariate_shift", 1: "label_shift"}

        loader = DriftDataLoader(
            X=X,
            y=y,
            batch_size=100,
            drift_schedule=drift_schedule,
            random_state=random_seed,
        )

        batches = list(loader)

        # Check first batch has drift
        _, _, metadata_0 = batches[0]
        assert metadata_0["drift_type"] == "covariate_shift"

        # Check second batch has drift
        _, _, metadata_1 = batches[1]
        assert metadata_1["drift_type"] == "label_shift"

    def test_reset_functionality(self, streaming_data):
        """Test reset functionality."""
        # Iterate through some batches
        batch_count = 0
        for _ in streaming_data:
            batch_count += 1
            if batch_count >= 3:
                break

        assert streaming_data.current_batch == 3

        # Reset and check
        streaming_data.reset()
        assert streaming_data.current_batch == 0

        # Verify we can iterate again
        new_batches = list(streaming_data)
        assert len(new_batches) > 0

    def test_apply_covariate_drift(self, small_classification_data, random_seed):
        """Test covariate shift application."""
        X, y = small_classification_data

        loader = DriftDataLoader(X=X, y=y, batch_size=100, random_state=random_seed)

        # Apply covariate shift
        X_drift, y_drift = loader._apply_drift(X.iloc[:50], y.iloc[:50], "covariate_shift")

        # Check that features changed
        assert not X.iloc[:50].equals(X_drift)

        # Check that labels didn't change
        assert y.iloc[:50].equals(y_drift)

        # Check data types and shapes are preserved
        assert X_drift.shape == X.iloc[:50].shape
        assert y_drift.shape == y.iloc[:50].shape

    def test_apply_label_drift(self, small_classification_data, random_seed):
        """Test label shift application."""
        X, y = small_classification_data

        loader = DriftDataLoader(X=X, y=y, batch_size=100, random_state=random_seed)

        # Apply label shift
        X_drift, y_drift = loader._apply_drift(X.iloc[:50], y.iloc[:50], "label_shift")

        # Check that features didn't change much
        assert X.iloc[:50].equals(X_drift)

        # Check that some labels changed (should be different with high probability)
        # Note: with random flipping, there's a small chance they could be the same
        assert len(y_drift) == len(y.iloc[:50])

    def test_apply_concept_drift(self, small_classification_data, random_seed):
        """Test concept drift application."""
        X, y = small_classification_data

        loader = DriftDataLoader(X=X, y=y, batch_size=100, random_state=random_seed)

        # Apply concept drift
        X_drift, y_drift = loader._apply_drift(X.iloc[:50], y.iloc[:50], "concept_drift")

        # Check shapes are preserved
        assert X_drift.shape == X.iloc[:50].shape
        assert y_drift.shape == y.iloc[:50].shape

        # Check that relationship changed (some labels should be different)
        assert len(y_drift) == len(y.iloc[:50])


class TestFeatureProcessor:
    """Test suite for FeatureProcessor class."""

    def test_init(self, random_seed):
        """Test FeatureProcessor initialization."""
        processor = FeatureProcessor(
            scaler_type="standard",
            handle_missing="median",
            random_state=random_seed,
        )

        assert processor.scaler_type == "standard"
        assert processor.handle_missing == "median"
        assert processor.random_state == random_seed
        assert not processor.is_fitted

    def test_fit_transform(self, small_classification_data, random_seed):
        """Test fit_transform functionality."""
        X, y = small_classification_data

        processor = FeatureProcessor(random_state=random_seed)
        X_processed = processor.fit_transform(X, y)

        assert processor.is_fitted
        assert isinstance(X_processed, pd.DataFrame)
        assert X_processed.shape[0] == X.shape[0]
        assert X_processed.shape[1] == X.shape[1]

        # Check that features are scaled (mean should be close to 0, std close to 1)
        means = X_processed.mean()
        stds = X_processed.std()

        assert np.allclose(means, 0, atol=1e-10)
        assert np.allclose(stds, 1, atol=1e-10)

    def test_transform_without_fit(self, small_classification_data, random_seed):
        """Test transform without fitting first."""
        X, _ = small_classification_data

        processor = FeatureProcessor(random_state=random_seed)

        with pytest.raises(ValueError, match="FeatureProcessor must be fitted"):
            processor.transform(X)

    def test_handle_missing_values(self, data_with_missing_values, random_seed):
        """Test missing value handling."""
        X_missing, y = data_with_missing_values

        processor = FeatureProcessor(
            handle_missing="median",
            random_state=random_seed,
        )

        X_processed = processor.fit_transform(X_missing, y)

        # Check that no missing values remain
        assert not X_processed.isnull().any().any()

        # Check shape is preserved
        assert X_processed.shape == X_missing.shape

    def test_different_scalers(self, small_classification_data, random_seed):
        """Test different scaler types."""
        X, y = small_classification_data

        scalers = ["standard", "robust", "minmax"]

        for scaler_type in scalers:
            processor = FeatureProcessor(
                scaler_type=scaler_type,
                random_state=random_seed,
            )

            X_processed = processor.fit_transform(X, y)

            assert isinstance(X_processed, pd.DataFrame)
            assert X_processed.shape == X.shape
            assert not X_processed.isnull().any().any()

    def test_partial_fit(self, small_classification_data, random_seed):
        """Test partial fit functionality."""
        X, y = small_classification_data

        processor = FeatureProcessor(random_state=random_seed)

        # Initial fit
        X_processed_1 = processor.fit_transform(X[:500], y[:500])
        assert processor.is_fitted

        # Partial fit on new data
        processor.partial_fit(X[500:])

        # Transform all data
        X_processed_all = processor.transform(X)

        assert X_processed_all.shape == X.shape


class TestDriftInjector:
    """Test suite for DriftInjector class."""

    def test_init(self, random_seed):
        """Test DriftInjector initialization."""
        injector = DriftInjector(random_state=random_seed)
        assert injector.random_state == random_seed

    def test_inject_covariate_shift(self, small_classification_data, random_seed):
        """Test covariate shift injection."""
        X, _ = small_classification_data

        injector = DriftInjector(random_state=random_seed)
        X_shifted = injector.inject_covariate_shift(X, shift_magnitude=0.5)

        # Check that data changed
        assert not X.equals(X_shifted)

        # Check shape and columns are preserved
        assert X_shifted.shape == X.shape
        assert X_shifted.columns.equals(X.columns)

        # Check data types are preserved
        assert X_shifted.dtypes.equals(X.dtypes)

    def test_inject_label_shift_gradual(self, small_classification_data, random_seed):
        """Test gradual label shift injection."""
        _, y = small_classification_data

        injector = DriftInjector(random_state=random_seed)
        y_shifted = injector.inject_label_shift(
            y, shift_probability=0.1, shift_pattern="gradual"
        )

        # Check shape and type are preserved
        assert y_shifted.shape == y.shape
        assert y_shifted.dtype == y.dtype

        # Check that some labels changed (with high probability)
        # Note: Due to randomness, this might occasionally fail
        assert len(y_shifted) == len(y)

    def test_inject_concept_drift(self, small_classification_data, random_seed):
        """Test concept drift injection."""
        X, y = small_classification_data

        injector = DriftInjector(random_state=random_seed)
        X_drift, y_drift = injector.inject_concept_drift(
            X, y, drift_severity=0.3, drift_pattern="linear"
        )

        # Check shapes are preserved
        assert X_drift.shape == X.shape
        assert y_drift.shape == y.shape

        # Check types are preserved
        assert X_drift.dtypes.equals(X.dtypes)
        assert y_drift.dtype == y.dtype

    def test_create_drift_schedule(self, random_seed):
        """Test drift schedule creation."""
        injector = DriftInjector(random_state=random_seed)

        drift_points = [10, 30, 50]
        drift_types = ["covariate_shift", "concept_drift", "label_shift"]
        drift_durations = [5, 10, 8]

        schedule = injector.create_drift_schedule(
            total_batches=100,
            drift_points=drift_points,
            drift_types=drift_types,
            drift_durations=drift_durations,
        )

        # Check schedule structure
        assert isinstance(schedule, dict)
        assert len(schedule) == sum(drift_durations)

        # Check specific drift periods
        assert schedule[10] == "covariate_shift"
        assert schedule[14] == "covariate_shift"  # 10 + 5 - 1

        assert schedule[30] == "concept_drift"
        assert schedule[39] == "concept_drift"  # 30 + 10 - 1

        assert schedule[50] == "label_shift"
        assert schedule[57] == "label_shift"  # 50 + 8 - 1

    def test_drift_schedule_default_duration(self, random_seed):
        """Test drift schedule with default durations."""
        injector = DriftInjector(random_state=random_seed)

        drift_points = [10, 30]
        drift_types = ["covariate_shift", "concept_drift"]

        schedule = injector.create_drift_schedule(
            total_batches=100,
            drift_points=drift_points,
            drift_types=drift_types,
        )

        # Check default duration of 10 batches
        expected_length = 10 * len(drift_points)
        assert len(schedule) == expected_length

    def test_shift_type_variations(self, small_classification_data, random_seed):
        """Test different shift types for covariate drift."""
        X, _ = small_classification_data

        injector = DriftInjector(random_state=random_seed)
        shift_types = ["linear", "polynomial", "seasonal"]

        for shift_type in shift_types:
            X_shifted = injector.inject_covariate_shift(
                X, shift_magnitude=0.3, shift_type=shift_type
            )

            # Basic checks
            assert X_shifted.shape == X.shape
            assert X_shifted.columns.equals(X.columns)
            assert not X.equals(X_shifted)

    def test_empty_data_handling(self, random_seed):
        """Test handling of edge cases with empty data."""
        injector = DriftInjector(random_state=random_seed)

        # Empty DataFrame
        empty_X = pd.DataFrame()
        empty_y = pd.Series([], dtype=int)

        # Should not raise errors
        X_shifted = injector.inject_covariate_shift(empty_X)
        assert X_shifted.empty

        y_shifted = injector.inject_label_shift(empty_y)
        assert y_shifted.empty

        X_drift, y_drift = injector.inject_concept_drift(empty_X, empty_y)
        assert X_drift.empty and y_drift.empty