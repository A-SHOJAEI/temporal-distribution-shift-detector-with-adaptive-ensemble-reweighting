"""
Data preprocessing utilities for temporal distribution shift detection.

This module provides tools for feature engineering, drift injection, and data
preprocessing specific to streaming data scenarios.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """
    Feature processor for streaming tabular data with support for online updates.

    Handles scaling, encoding, and feature engineering for temporal data.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        handle_missing: str = "median",
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize the feature processor.

        Args:
            scaler_type: Type of scaler ('standard', 'robust', 'minmax').
            handle_missing: Strategy for missing values ('median', 'mean', 'mode').
            feature_selection: Feature selection method ('variance', 'mutual_info').
            n_features: Number of features to select.
            random_state: Random seed for reproducibility.
        """
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.random_state = random_state

        self.is_fitted = False
        self.feature_names_in_ = None
        self.pipeline = None
        self.selected_features_ = None

        self._create_pipeline()

    def _create_pipeline(self) -> None:
        """Create the preprocessing pipeline."""
        steps = []

        # Imputation
        if self.handle_missing == "median":
            imputer = SimpleImputer(strategy="median")
        elif self.handle_missing == "mean":
            imputer = SimpleImputer(strategy="mean")
        else:
            imputer = SimpleImputer(strategy="most_frequent")

        steps.append(("imputer", imputer))

        # Scaling
        if self.scaler_type == "standard":
            scaler = StandardScaler()
        elif self.scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = MinMaxScaler()

        steps.append(("scaler", scaler))

        self.pipeline = Pipeline(steps)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureProcessor":
        """
        Fit the feature processor on training data.

        Args:
            X: Training features.
            y: Training labels (used for feature selection).

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting FeatureProcessor on {X.shape[0]} samples")

        self.feature_names_in_ = X.columns.tolist()

        # Fit the pipeline
        self.pipeline.fit(X)

        # Feature selection if requested
        if self.feature_selection is not None and y is not None:
            self._fit_feature_selection(X, y)

        self.is_fitted = True
        logger.info("FeatureProcessor fitting completed")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted processor.

        Args:
            X: Features to transform.

        Returns:
            Transformed features.

        Raises:
            ValueError: If processor is not fitted.
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")

        # Transform using pipeline
        X_transformed = self.pipeline.transform(X)

        # Convert back to DataFrame
        feature_names = self.feature_names_in_
        if self.selected_features_ is not None:
            feature_names = [feature_names[i] for i in self.selected_features_]
            X_transformed = X_transformed[:, self.selected_features_]

        X_df = pd.DataFrame(
            X_transformed, columns=feature_names, index=X.index
        )

        return X_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the processor and transform features in one step.

        Args:
            X: Training features.
            y: Training labels.

        Returns:
            Transformed features.
        """
        return self.fit(X, y).transform(X)

    def partial_fit(self, X: pd.DataFrame) -> "FeatureProcessor":
        """
        Incrementally fit the processor on new data (online learning).

        Args:
            X: New batch of features.

        Returns:
            Self for method chaining.
        """
        if not self.is_fitted:
            # First batch - perform full fit
            return self.fit(X)

        # For StandardScaler and other scalers that support partial_fit
        scaler = self.pipeline.named_steps["scaler"]
        if hasattr(scaler, "partial_fit"):
            scaler.partial_fit(X)
        else:
            logger.warning(f"Scaler {type(scaler)} does not support partial_fit")

        return self

    def _fit_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit feature selection on the data.

        Args:
            X: Training features.
            y: Training labels.
        """
        from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

        if self.feature_selection == "variance":
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X)
            self.selected_features_ = selector.get_support(indices=True)

        elif self.feature_selection == "mutual_info":
            if self.n_features is None:
                self.n_features = min(50, X.shape[1])

            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            # Select top features
            top_indices = np.argsort(mi_scores)[::-1][: self.n_features]
            self.selected_features_ = sorted(top_indices)

        logger.info(
            f"Selected {len(self.selected_features_)} features using {self.feature_selection}"
        )

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self.feature_selection == "mutual_info" and hasattr(self, "_mi_scores"):
            feature_names = [self.feature_names_in_[i] for i in self.selected_features_]
            return dict(zip(feature_names, self._mi_scores[self.selected_features_]))
        return None


class DriftInjector:
    """
    Utility class for injecting synthetic distribution shifts into data streams.

    Supports various types of drift including covariate shift, label shift,
    and concept drift with configurable severity and timing.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the drift injector.

        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def inject_covariate_shift(
        self,
        X: pd.DataFrame,
        shift_features: Optional[List[str]] = None,
        shift_magnitude: float = 0.5,
        shift_type: str = "linear",
    ) -> pd.DataFrame:
        """
        Inject covariate shift into features.

        Args:
            X: Input features.
            shift_features: List of feature names to shift. If None, shifts all numeric features.
            shift_magnitude: Magnitude of the shift (standard deviations).
            shift_type: Type of shift ('linear', 'polynomial', 'seasonal').

        Returns:
            Features with covariate shift applied.
        """
        X_shifted = X.copy()

        if shift_features is None:
            shift_features = X.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Injecting covariate shift to {len(shift_features)} features")

        for feature in shift_features:
            if feature not in X.columns:
                continue

            feature_std = X[feature].std()
            shift_amount = shift_magnitude * feature_std

            if shift_type == "linear":
                # Linear drift over time
                drift_trend = np.linspace(0, shift_amount, len(X))
                X_shifted[feature] = X[feature] + drift_trend

            elif shift_type == "polynomial":
                # Polynomial drift
                time_points = np.linspace(0, 1, len(X))
                drift_trend = shift_amount * (time_points ** 2)
                X_shifted[feature] = X[feature] + drift_trend

            elif shift_type == "seasonal":
                # Seasonal drift pattern
                frequency = len(X) // 4  # 4 seasons
                time_points = np.arange(len(X))
                drift_trend = shift_amount * np.sin(2 * np.pi * time_points / frequency)
                X_shifted[feature] = X[feature] + drift_trend

            else:
                # Sudden shift
                shift_point = len(X) // 2
                X_shifted.iloc[shift_point:, X_shifted.columns.get_loc(feature)] += shift_amount

        return X_shifted

    def inject_label_shift(
        self,
        y: pd.Series,
        shift_probability: float = 0.1,
        shift_pattern: str = "gradual",
    ) -> pd.Series:
        """
        Inject label shift by changing label distribution.

        Args:
            y: Original labels.
            shift_probability: Probability of label flipping.
            shift_pattern: Pattern of shift ('gradual', 'sudden', 'cyclical').

        Returns:
            Labels with shift applied.
        """
        y_shifted = y.copy()

        logger.info(f"Injecting label shift with pattern: {shift_pattern}")

        if shift_pattern == "gradual":
            # Gradually increase flip probability
            flip_probs = np.linspace(0, shift_probability, len(y))
            flip_mask = self.rng.random(len(y)) < flip_probs
            y_shifted[flip_mask] = 1 - y_shifted[flip_mask]

        elif shift_pattern == "sudden":
            # Sudden shift at midpoint
            shift_point = len(y) // 2
            flip_mask = self.rng.random(len(y) - shift_point) < shift_probability
            indices_to_flip = y_shifted.index[shift_point:][flip_mask]
            y_shifted[indices_to_flip] = 1 - y_shifted[indices_to_flip]

        elif shift_pattern == "cyclical":
            # Cyclical pattern of label shifts
            cycle_length = len(y) // 4
            for i in range(len(y)):
                cycle_position = (i % cycle_length) / cycle_length
                current_flip_prob = shift_probability * np.sin(2 * np.pi * cycle_position)
                if self.rng.random() < abs(current_flip_prob):
                    y_shifted.iloc[i] = 1 - y_shifted.iloc[i]

        return y_shifted

    def inject_concept_drift(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        drift_features: Optional[List[str]] = None,
        drift_severity: float = 0.3,
        drift_pattern: str = "linear",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Inject concept drift by changing feature-label relationships.

        Args:
            X: Input features.
            y: Original labels.
            drift_features: Features involved in concept drift.
            drift_severity: Severity of the concept change.
            drift_pattern: Pattern of drift evolution.

        Returns:
            Tuple of (features, modified_labels).
        """
        if drift_features is None:
            # Select most important features
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            drift_features = numeric_features[:min(3, len(numeric_features))]

        logger.info(f"Injecting concept drift using features: {drift_features}")

        X_drift = X.copy()
        y_drift = y.copy()

        # Create concept drift by modifying decision boundary
        for i, feature in enumerate(drift_features):
            if feature not in X.columns:
                continue

            feature_values = X[feature].values
            feature_threshold = np.median(feature_values)

            if drift_pattern == "linear":
                # Linear change in decision boundary
                drift_strengths = np.linspace(0, drift_severity, len(X))
            else:
                # Sudden change
                drift_strengths = np.concatenate([
                    np.zeros(len(X) // 2),
                    np.full(len(X) - len(X) // 2, drift_severity)
                ])

            # Apply concept drift
            for idx in range(len(X)):
                if feature_values[idx] > feature_threshold:
                    if self.rng.random() < drift_strengths[idx]:
                        y_drift.iloc[idx] = 1 - y_drift.iloc[idx]

        return X_drift, y_drift

    def create_drift_schedule(
        self,
        total_batches: int,
        drift_points: List[int],
        drift_types: List[str],
        drift_durations: Optional[List[int]] = None,
    ) -> Dict[int, str]:
        """
        Create a schedule for drift injection across batches.

        Args:
            total_batches: Total number of batches.
            drift_points: Batch indices where drift starts.
            drift_types: Types of drift at each point.
            drift_durations: Duration of each drift in batches.

        Returns:
            Dictionary mapping batch indices to drift types.
        """
        if drift_durations is None:
            drift_durations = [10] * len(drift_points)  # Default 10 batches

        schedule = {}

        for point, drift_type, duration in zip(drift_points, drift_types, drift_durations):
            for batch_idx in range(point, min(point + duration, total_batches)):
                schedule[batch_idx] = drift_type

        logger.info(f"Created drift schedule with {len(schedule)} drift batches")
        return schedule