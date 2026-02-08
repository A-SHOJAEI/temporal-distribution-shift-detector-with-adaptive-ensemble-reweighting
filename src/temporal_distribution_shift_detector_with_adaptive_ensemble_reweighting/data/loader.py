"""
Data loading utilities for temporal distribution shift detection.

This module provides data loaders for various datasets including Covertype,
ELEC2, and Airlines datasets with support for streaming data and temporal drift injection.
"""

import logging
import os
import urllib.request
from typing import Dict, Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class DataLoader:
    """Base data loader for streaming tabular datasets."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the data loader.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def load_covertype(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the Covertype dataset from sklearn.

        Returns:
            Tuple of (features, labels) as DataFrame and Series.

        Raises:
            RuntimeError: If dataset loading fails.
        """
        try:
            logger.info("Loading Covertype dataset...")
            data = fetch_covtype(return_X_y=True, as_frame=True)
            X, y = data
            logger.info(f"Loaded Covertype dataset: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
        except Exception as e:
            raise RuntimeError(f"Failed to load Covertype dataset: {e}")

    def load_elec2(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the ELEC2 (Electricity Market) dataset.

        Args:
            data_path: Path to ELEC2 dataset. If None, downloads from UCI repository.

        Returns:
            Tuple of (features, labels) as DataFrame and Series.

        Raises:
            RuntimeError: If dataset loading fails.
        """
        try:
            if data_path is None:
                # Download from UCI repository
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt"
                data_path = "elec2_data.txt"
                if not os.path.exists(data_path):
                    logger.info("Downloading ELEC2 dataset...")
                    urllib.request.urlretrieve(url, data_path)

            logger.info("Loading ELEC2 dataset...")
            # Load and process ELEC2 data
            df = pd.read_csv(
                data_path,
                sep=";",
                decimal=",",
                parse_dates=[0],
                index_col=0,
            )

            # Create binary classification target (high vs low consumption)
            consumption_median = df.median(axis=1)
            y = (consumption_median > consumption_median.median()).astype(int)

            # Create features from time series data
            X = self._create_elec2_features(df)

            logger.info(f"Loaded ELEC2 dataset: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y

        except Exception as e:
            raise RuntimeError(f"Failed to load ELEC2 dataset: {e}")

    def _create_elec2_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from ELEC2 time series data.

        Args:
            df: Raw ELEC2 DataFrame.

        Returns:
            Feature DataFrame.
        """
        features = pd.DataFrame(index=df.index)

        # Time-based features
        features["hour"] = df.index.hour
        features["day_of_week"] = df.index.dayofweek
        features["month"] = df.index.month
        features["quarter"] = df.index.quarter

        # Statistical features
        features["mean_consumption"] = df.mean(axis=1)
        features["std_consumption"] = df.std(axis=1)
        features["min_consumption"] = df.min(axis=1)
        features["max_consumption"] = df.max(axis=1)

        # Lag features
        for lag in [1, 2, 7, 24]:
            features[f"lag_{lag}_mean"] = df.mean(axis=1).shift(lag)

        # Rolling statistics
        for window in [7, 24, 168]:  # Week, day, week in hours
            features[f"rolling_{window}_mean"] = (
                df.mean(axis=1).rolling(window=window).mean()
            )
            features[f"rolling_{window}_std"] = (
                df.mean(axis=1).rolling(window=window).std()
            )

        # Drop rows with NaN values
        features = features.dropna()

        return features

    def load_airlines(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the Airlines Delay dataset.

        Args:
            data_path: Path to Airlines dataset file.

        Returns:
            Tuple of (features, labels) as DataFrame and Series.

        Raises:
            RuntimeError: If dataset loading fails.
        """
        if data_path is None:
            raise ValueError("Airlines dataset path must be provided")

        try:
            logger.info("Loading Airlines dataset...")
            df = pd.read_csv(data_path)

            # Extract features and target
            target_col = "ArrDelay"
            if target_col not in df.columns:
                # Try alternative column names
                for alt_col in ["arr_delay", "arrival_delay", "delay"]:
                    if alt_col in df.columns:
                        target_col = alt_col
                        break
                else:
                    raise ValueError("Target column not found in Airlines dataset")

            # Binary classification: delayed (>15 min) vs on-time
            y = (df[target_col] > 15).astype(int)

            # Select and process features
            feature_cols = [
                col for col in df.columns
                if col not in [target_col, "ArrDelay", "arr_delay", "arrival_delay", "delay"]
            ]
            X = df[feature_cols].copy()

            # Encode categorical variables
            for col in X.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            # Handle missing values
            X = X.fillna(X.median())

            logger.info(f"Loaded Airlines dataset: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y

        except Exception as e:
            raise RuntimeError(f"Failed to load Airlines dataset: {e}")


class DriftDataLoader:
    """Data loader that provides streaming data with temporal drift patterns."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        batch_size: int = 1000,
        drift_schedule: Optional[Dict[int, str]] = None,
        random_state: int = 42,
    ):
        """
        Initialize the drift data loader.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            batch_size: Size of data batches.
            drift_schedule: Dictionary mapping batch indices to drift types.
            random_state: Random seed for reproducibility.
        """
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.batch_size = batch_size
        self.drift_schedule = drift_schedule or {}
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.current_batch = 0
        self.total_batches = len(self.X) // batch_size

        logger.info(
            f"Initialized DriftDataLoader with {len(self.X)} samples, "
            f"batch_size={batch_size}, total_batches={self.total_batches}"
        )

    def __iter__(self) -> Iterator[Tuple[pd.DataFrame, pd.Series, Dict]]:
        """
        Iterate over data batches with drift information.

        Yields:
            Tuple of (batch_X, batch_y, metadata) where metadata contains drift info.
        """
        self.current_batch = 0
        return self

    def __next__(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Get the next data batch.

        Returns:
            Tuple of (batch_X, batch_y, metadata).

        Raises:
            StopIteration: When all batches have been processed.
        """
        if self.current_batch >= self.total_batches:
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.X))

        batch_X = self.X.iloc[start_idx:end_idx].copy()
        batch_y = self.y.iloc[start_idx:end_idx].copy()

        # Apply drift if scheduled
        drift_type = self.drift_schedule.get(self.current_batch, "none")
        if drift_type != "none":
            batch_X, batch_y = self._apply_drift(batch_X, batch_y, drift_type)

        metadata = {
            "batch_index": self.current_batch,
            "drift_type": drift_type,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "timestamp": pd.Timestamp.now(),
        }

        self.current_batch += 1
        return batch_X, batch_y, metadata

    def _apply_drift(
        self, X: pd.DataFrame, y: pd.Series, drift_type: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply synthetic drift to a data batch.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            drift_type: Type of drift to apply.

        Returns:
            Tuple of modified (X, y).
        """
        X_drift = X.copy()
        y_drift = y.copy()

        if drift_type == "covariate_shift":
            # Shift feature distributions
            for col in X_drift.select_dtypes(include=[np.number]).columns:
                shift_magnitude = self.rng.normal(0, 0.1 * X_drift[col].std())
                X_drift[col] = X_drift[col] + shift_magnitude

        elif drift_type == "label_shift":
            # Change label distribution
            flip_prob = 0.1
            flip_mask = self.rng.random(len(y_drift)) < flip_prob
            y_drift[flip_mask] = 1 - y_drift[flip_mask]

        elif drift_type == "concept_drift":
            # Change relationship between features and labels
            # Gradually flip labels for samples with specific feature patterns
            feature_col = X_drift.columns[0]
            threshold = X_drift[feature_col].median()
            flip_mask = X_drift[feature_col] > threshold
            flip_prob = self.rng.random(flip_mask.sum()) < 0.3
            y_drift.loc[flip_mask] = y_drift.loc[flip_mask] ^ flip_prob

        elif drift_type == "combined":
            # Apply multiple types of drift
            X_drift, y_drift = self._apply_drift(X_drift, y_drift, "covariate_shift")
            X_drift, y_drift = self._apply_drift(X_drift, y_drift, "concept_drift")

        logger.debug(f"Applied {drift_type} drift to batch")
        return X_drift, y_drift

    def reset(self) -> None:
        """Reset the iterator to the beginning."""
        self.current_batch = 0

    def get_drift_schedule(self) -> Dict[int, str]:
        """
        Get the current drift schedule.

        Returns:
            Dictionary mapping batch indices to drift types.
        """
        return self.drift_schedule.copy()

    def update_drift_schedule(self, schedule: Dict[int, str]) -> None:
        """
        Update the drift schedule.

        Args:
            schedule: New drift schedule.
        """
        self.drift_schedule = schedule
        logger.info(f"Updated drift schedule: {schedule}")