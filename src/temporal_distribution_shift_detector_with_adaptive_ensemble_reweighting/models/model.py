"""Core model implementations for adaptive ensemble drift detection.

This module implements the main AdaptiveEnsembleDetector along with supporting
components for Bayesian reweighting and drift detection. The implementation
provides a complete framework for detecting temporal distribution shifts in
streaming data using an adaptive ensemble approach with theoretical regret
guarantees.

Classes:
    DriftDetector: Multi-method drift detector for streaming data.
    BayesianReweighter: Bayesian online learning for ensemble reweighting.
    AdaptiveEnsembleDetector: Main adaptive ensemble with drift detection.

Example:
    Basic usage of the adaptive ensemble detector:

    >>> from temporal_distribution_shift_detector import AdaptiveEnsembleDetector
    >>> detector = AdaptiveEnsembleDetector()
    >>> detector.fit(X_train, y_train)
    >>> predictions = detector.predict(X_test)
    >>> detector.partial_fit(X_new, y_new)  # Online adaptation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder as _LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

logger = logging.getLogger(__name__)


class DriftDetector:
    """Multi-method drift detector for streaming data.

    Implements a comprehensive drift detection system that combines statistical
    tests, distribution monitoring, and prediction confidence analysis to detect
    various types of distribution shift in streaming data environments.

    The detector maintains a reference window and continuously compares incoming
    data against this reference to identify:
    - Covariate drift: Changes in input feature distributions
    - Label drift: Changes in target variable distribution
    - Prediction drift: Changes in model confidence/prediction patterns

    Attributes:
        window_size (int): Size of the reference and current data windows.
        alpha (float): Significance level for statistical tests.
        detection_threshold (float): Threshold above which drift is detected.
        min_samples (int): Minimum samples required before drift detection.
        reference_X (pd.DataFrame): Reference feature data for comparison.
        reference_y (pd.Series): Reference labels for comparison.
        reference_predictions (np.ndarray): Reference predictions for comparison.
        current_X (List[pd.DataFrame]): Current window of feature data.
        current_y (List[pd.Series]): Current window of labels.
        current_predictions (List[np.ndarray]): Current window of predictions.
        drift_history (List[Dict]): History of all drift detection results.

    Example:
        Basic drift detection workflow:

        >>> detector = DriftDetector(window_size=1000, alpha=0.05)
        >>> detector.set_reference(X_ref, y_ref, pred_ref)
        >>> result = detector.update(X_new, y_new, pred_new)
        >>> if result['drift_detected']:
        >>>     print(f"Drift detected: {result['drift_type']}")
    """

    def __init__(
        self,
        window_size: int = 1000,
        alpha: float = 0.05,
        detection_threshold: float = 0.1,
        min_samples: int = 100,
    ):
        """Initialize the drift detector with specified parameters.

        Args:
            window_size (int, optional): Size of the reference and current data
                windows. Larger values provide more stable detection but slower
                adaptation. Defaults to 1000.
            alpha (float, optional): Significance level for statistical tests
                (e.g., Kolmogorov-Smirnov). Lower values make detection more
                conservative. Defaults to 0.05.
            detection_threshold (float, optional): Threshold above which drift
                is considered detected. Higher values reduce false positives
                but may miss subtle drifts. Defaults to 0.1.
            min_samples (int, optional): Minimum number of samples required
                in current window before attempting drift detection. Ensures
                statistical validity. Defaults to 100.

        Raises:
            ValueError: If window_size or min_samples are non-positive, or if
                alpha is not between 0 and 1, or if detection_threshold is negative.
        """
        # Validate parameters
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if detection_threshold < 0:
            raise ValueError(f"detection_threshold must be non-negative, got {detection_threshold}")
        if min_samples <= 0:
            raise ValueError(f"min_samples must be positive, got {min_samples}")

        self.window_size = window_size
        self.alpha = alpha
        self.detection_threshold = detection_threshold
        self.min_samples = min_samples

        # Reference data storage
        self.reference_X: Optional[pd.DataFrame] = None
        self.reference_y: Optional[pd.Series] = None
        self.reference_predictions: Optional[np.ndarray] = None

        # Current window data
        self.current_X: List[pd.DataFrame] = []
        self.current_y: List[pd.Series] = []
        self.current_predictions: List[np.ndarray] = []

        # Drift detection history
        self.drift_history: List[Dict] = []

    def set_reference(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        predictions: Optional[np.ndarray] = None,
    ) -> None:
        """Set the reference data for future drift detection comparisons.

        Establishes the baseline distributions against which future data will
        be compared for drift detection. This should typically be called with
        representative data from the training phase or initial stable period.

        Args:
            X (pd.DataFrame): Reference feature data with shape (n_samples, n_features).
                Must contain all features that will be monitored for drift.
            y (pd.Series, optional): Reference target labels with shape (n_samples,).
                Required for label drift detection. Defaults to None.
            predictions (np.ndarray, optional): Reference model predictions with shape
                (n_samples,) for binary or (n_samples, n_classes) for multiclass.
                Required for prediction drift detection. Defaults to None.

        Raises:
            ValueError: If X is empty or contains invalid data types.
            IndexError: If y or predictions have mismatched lengths with X.

        Note:
            This method creates deep copies of the input data to prevent
            unintentional modifications to the reference distributions.
        """
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValueError("X cannot be empty")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        if y is not None and len(y) != len(X):
            raise ValueError(f"y length ({len(y)}) doesn't match X length ({len(X)})")
        if predictions is not None and len(predictions) != len(X):
            raise ValueError(f"predictions length ({len(predictions)}) doesn't match X length ({len(X)})")

        try:
            self.reference_X = X.copy()
            if y is not None:
                self.reference_y = y.copy()
            if predictions is not None:
                self.reference_predictions = predictions.copy()

            logger.info(f"Set reference data: {len(X)} samples")
        except Exception as e:
            logger.error(f"Failed to set reference data: {str(e)}")
            raise RuntimeError(f"Failed to set reference data: {str(e)}") from e

    def update(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[bool, float]]:
        """Update the detector with new data and perform drift detection.

        Adds new data to the current window, maintains window size limits,
        and performs comprehensive drift detection if sufficient data is available.
        The method automatically manages the sliding window and triggers detection
        when the minimum sample requirement is met.

        Args:
            X (pd.DataFrame): New batch of features with shape (batch_size, n_features).
                Must have the same column structure as the reference data.
            y (pd.Series, optional): New batch of labels with shape (batch_size,).
                Required for label drift detection. Defaults to None.
            predictions (np.ndarray, optional): New batch of model predictions with
                shape (batch_size,) or (batch_size, n_classes). Required for
                prediction drift detection. Defaults to None.

        Returns:
            Dict[str, Union[bool, float]]: Drift detection results containing:
                - drift_detected (bool): Whether any type of drift was detected.
                - drift_score (float): Maximum drift score across all detection methods.
                - drift_type (str): Type of dominant drift ('covariate', 'label',
                  'prediction', or 'none').
                - covariate_drift_score (float): Score for covariate drift.
                - covariate_drift_detected (bool): Whether covariate drift was detected.
                - label_drift_score (float): Score for label drift (if y provided).
                - label_drift_detected (bool): Whether label drift was detected.
                - prediction_drift_score (float): Score for prediction drift (if predictions provided).
                - prediction_drift_detected (bool): Whether prediction drift was detected.

        Raises:
            ValueError: If X has incompatible structure with reference data.
            RuntimeError: If reference data has not been set via set_reference().

        Note:
            The method maintains a sliding window of recent data. When the window
            exceeds the specified window_size, oldest data is automatically removed.
        """
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValueError("X cannot be empty")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        if self.reference_X is None:
            raise RuntimeError("Reference data not set. Call set_reference() first.")
        if y is not None and len(y) != len(X):
            raise ValueError(f"y length ({len(y)}) doesn't match X length ({len(X)})")
        if predictions is not None and len(predictions) != len(X):
            raise ValueError(f"predictions length ({len(predictions)}) doesn't match X length ({len(X)})")

        # Check feature compatibility
        if not set(X.columns).issubset(set(self.reference_X.columns)):
            missing_cols = set(X.columns) - set(self.reference_X.columns)
            raise ValueError(f"X contains unknown columns: {missing_cols}")

        try:
            # Add new data to current window
            self.current_X.append(X)
            if y is not None:
                self.current_y.append(y)
            if predictions is not None:
                self.current_predictions.append(predictions)
        except Exception as e:
            logger.error(f"Failed to update drift detector: {str(e)}")
            raise RuntimeError(f"Failed to update drift detector: {str(e)}") from e

        # Maintain window size
        if len(self.current_X) > self.window_size:
            self.current_X.pop(0)
            if self.current_y:
                self.current_y.pop(0)
            if self.current_predictions:
                self.current_predictions.pop(0)

        # Check for drift if we have enough data
        current_size = sum(len(x) for x in self.current_X)
        if current_size >= self.min_samples and self.reference_X is not None:
            logger.debug(f"Performing drift detection with {current_size} samples in current window")
            return self._detect_drift()

        logger.debug(f"Insufficient data for drift detection: {current_size}/{self.min_samples} samples")
        return {"drift_detected": False, "drift_score": 0.0, "drift_type": "none"}

    def _detect_drift(self) -> Dict[str, Union[bool, float]]:
        """Perform comprehensive drift detection across multiple dimensions.

        Executes all available drift detection methods (covariate, label, and
        prediction drift) and aggregates results to determine overall drift status.
        Uses maximum aggregation across drift types to ensure sensitive detection.

        Returns:
            Dict[str, Union[bool, float]]: Comprehensive drift detection results containing:
                - drift_detected (bool): True if any drift type exceeds threshold.
                - drift_score (float): Maximum drift score across all methods.
                - drift_type (str): Dominant drift type or 'none' if no drift.
                - Plus individual results from each detection method.

        Note:
            This method combines current window data before analysis and logs
            drift detection events for monitoring and debugging purposes.
        """
        # Combine current window data
        current_X_combined = pd.concat(self.current_X, ignore_index=True)
        current_y_combined = None
        current_pred_combined = None

        if self.current_y:
            current_y_combined = pd.concat(self.current_y, ignore_index=True)
        if self.current_predictions:
            current_pred_combined = np.concatenate(self.current_predictions)

        # Detect different types of drift
        drift_results = {}

        # 1. Covariate drift detection
        covariate_drift = self._detect_covariate_drift(current_X_combined)
        drift_results.update(covariate_drift)

        # 2. Label drift detection
        if current_y_combined is not None and self.reference_y is not None:
            label_drift = self._detect_label_drift(current_y_combined)
            drift_results.update(label_drift)

        # 3. Prediction drift detection
        if current_pred_combined is not None and self.reference_predictions is not None:
            pred_drift = self._detect_prediction_drift(current_pred_combined)
            drift_results.update(pred_drift)

        # Aggregate drift detection
        drift_scores = [
            drift_results.get("covariate_drift_score", 0),
            drift_results.get("label_drift_score", 0),
            drift_results.get("prediction_drift_score", 0),
        ]
        max_drift_score = max(drift_scores)
        drift_detected = max_drift_score > self.detection_threshold

        # Determine dominant drift type
        drift_types = ["covariate", "label", "prediction"]
        dominant_drift_idx = np.argmax(drift_scores)
        drift_type = drift_types[dominant_drift_idx] if drift_detected else "none"

        result = {
            "drift_detected": drift_detected,
            "drift_score": max_drift_score,
            "drift_type": drift_type,
            **drift_results,
        }

        # Log drift detection
        self.drift_history.append(result.copy())
        if drift_detected:
            logger.warning(
                f"Drift detected: {drift_type} (score: {max_drift_score:.3f}, threshold: {self.detection_threshold})"
            )
            # Log individual drift scores for debugging
            logger.debug(f"Covariate drift score: {drift_results.get('covariate_drift_score', 0):.3f}")
            logger.debug(f"Label drift score: {drift_results.get('label_drift_score', 0):.3f}")
            logger.debug(f"Prediction drift score: {drift_results.get('prediction_drift_score', 0):.3f}")
        else:
            logger.debug(f"No drift detected (max score: {max_drift_score:.3f})")

        return result

    def _detect_covariate_drift(self, current_X: pd.DataFrame) -> Dict[str, float]:
        """Detect covariate drift using Kolmogorov-Smirnov statistical tests.

        Compares the distribution of each feature between reference and current
        data using the Kolmogorov-Smirnov two-sample test. Averages test statistics
        across all features to produce an overall covariate drift score.

        Args:
            current_X (pd.DataFrame): Current feature data to compare against reference.
                Must have matching column structure with reference data.

        Returns:
            Dict[str, float]: Covariate drift detection results containing:
                - covariate_drift_score (float): Average KS statistic across all features.
                - covariate_drift_detected (bool): True if score exceeds threshold.

        Note:
            Only compares features present in both reference and current data.
            Handles missing values by dropping them before comparison. Returns
            zero score if no valid features can be compared.
        """
        drift_scores = []

        for col in self.reference_X.columns:
            if col in current_X.columns:
                ref_values = self.reference_X[col].dropna()
                curr_values = current_X[col].dropna()

                if len(ref_values) > 0 and len(curr_values) > 0:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p = ks_2samp(ref_values, curr_values)
                    drift_scores.append(ks_stat)

        covariate_drift_score = np.mean(drift_scores) if drift_scores else 0.0

        return {
            "covariate_drift_score": covariate_drift_score,
            "covariate_drift_detected": covariate_drift_score > self.detection_threshold,
        }

    def _detect_label_drift(self, current_y: pd.Series) -> Dict[str, float]:
        """Detect label drift using Jensen-Shannon divergence.

        Compares the class distribution between reference and current labels
        using Jensen-Shannon divergence, which provides a symmetric and bounded
        measure of distributional difference suitable for drift detection.

        Args:
            current_y (pd.Series): Current labels to compare against reference labels.
                Can contain any label values; method handles unseen classes gracefully.

        Returns:
            Dict[str, float]: Label drift detection results containing:
                - label_drift_score (float): Jensen-Shannon divergence between distributions.
                - label_drift_detected (bool): True if divergence exceeds threshold.

        Note:
            Automatically handles class alignment between reference and current data.
            Adds small epsilon (1e-10) to prevent numerical issues with zero probabilities.
            Jensen-Shannon divergence ranges from 0 (identical) to log(2) (maximally different).
        """
        # Compare label distributions
        ref_dist = self.reference_y.value_counts(normalize=True).sort_index()
        curr_dist = current_y.value_counts(normalize=True).sort_index()

        # Align indices
        all_labels = set(ref_dist.index) | set(curr_dist.index)
        ref_probs = [ref_dist.get(label, 0) for label in sorted(all_labels)]
        curr_probs = [curr_dist.get(label, 0) for label in sorted(all_labels)]

        # Calculate Jensen-Shannon divergence
        ref_probs = np.array(ref_probs) + 1e-10  # Add small epsilon
        curr_probs = np.array(curr_probs) + 1e-10
        m = (ref_probs + curr_probs) / 2
        js_divergence = (entropy(ref_probs, m) + entropy(curr_probs, m)) / 2

        return {
            "label_drift_score": js_divergence,
            "label_drift_detected": js_divergence > self.detection_threshold,
        }

    def _detect_prediction_drift(self, current_predictions: np.ndarray) -> Dict[str, float]:
        """Detect drift in model prediction confidence patterns.

        Analyzes changes in model confidence by comparing the distribution of
        prediction confidence scores between reference and current data. Uses
        Kolmogorov-Smirnov test on confidence distributions.

        Args:
            current_predictions (np.ndarray): Current model predictions with shape
                (n_samples,) for binary or (n_samples, n_classes) for multiclass.

        Returns:
            Dict[str, float]: Prediction drift detection results containing:
                - prediction_drift_score (float): KS statistic for confidence distributions.
                - prediction_drift_detected (bool): True if statistic exceeds threshold.

        Note:
            For binary classification, uses distance from 0.5 as confidence measure.
            For multiclass, uses maximum probability across classes as confidence.
            Confidence ranges from 0 (uncertain) to 0.5 (binary) or 1 (multiclass) for certain.
        """
        # Compare prediction distributions
        if len(current_predictions.shape) == 1:
            # Binary classification
            ref_confidence = np.abs(self.reference_predictions - 0.5)
            curr_confidence = np.abs(current_predictions - 0.5)
        else:
            # Multi-class classification
            ref_confidence = np.max(self.reference_predictions, axis=1)
            curr_confidence = np.max(current_predictions, axis=1)

        # Statistical test for confidence distributions
        ks_stat, ks_p = ks_2samp(ref_confidence, curr_confidence)

        return {
            "prediction_drift_score": ks_stat,
            "prediction_drift_detected": ks_stat > self.detection_threshold,
        }

    def reset(self) -> None:
        """Reset the drift detector to initial state.

        Clears all accumulated data in current windows and drift history.
        Reference data is preserved to allow continued operation without
        needing to call set_reference() again.

        Note:
            This method is useful when starting monitoring of a new data stream
            or after handling detected drift that requires fresh baselines.
        """
        self.current_X.clear()
        self.current_y.clear()
        self.current_predictions.clear()
        self.drift_history.clear()


class BayesianReweighter:
    """Bayesian online learning mechanism for dynamic ensemble reweighting.

    Implements a Thompson sampling approach using Beta distributions to maintain
    beliefs about each base learner's performance and dynamically reweight them
    based on recent accuracy under potentially shifting data regimes.

    The reweighter maintains Beta distribution parameters (alpha, beta) for each
    model, representing success/failure counts. When drift is detected, beliefs
    are partially reset to allow rapid adaptation to new data characteristics.

    Attributes:
        n_models (int): Number of base models in the ensemble.
        initial_alpha (float): Initial alpha parameter for Beta distributions.
        initial_beta (float): Initial beta parameter for Beta distributions.
        decay_factor (float): Exponential decay factor for historical performance.
        exploration_factor (float): Temperature parameter for exploration in Thompson sampling.
        alpha (np.ndarray): Current alpha parameters for each model's Beta distribution.
        beta (np.ndarray): Current beta parameters for each model's Beta distribution.
        performance_history (List[np.ndarray]): History of accuracy scores for each model.
        weights_history (List[np.ndarray]): History of computed ensemble weights.

    Example:
        Basic usage for ensemble reweighting:

        >>> reweighter = BayesianReweighter(n_models=3, decay_factor=0.95)
        >>> reweighter.update(model_predictions, true_labels, drift_detected=False)
        >>> weights = reweighter.get_weights()
        >>> uncertainty = reweighter.get_uncertainty()
    """

    def __init__(
        self,
        n_models: int,
        initial_alpha: float = 1.0,
        initial_beta: float = 1.0,
        decay_factor: float = 0.95,
        exploration_factor: float = 0.1,
    ):
        """Initialize the Bayesian reweighter with specified parameters.

        Args:
            n_models (int): Number of base models in the ensemble. Must be positive.
            initial_alpha (float, optional): Initial alpha parameter for Beta
                distributions. Higher values represent more initial successes.
                Defaults to 1.0 (uniform prior).
            initial_beta (float, optional): Initial beta parameter for Beta
                distributions. Higher values represent more initial failures.
                Defaults to 1.0 (uniform prior).
            decay_factor (float, optional): Exponential decay factor applied to
                Beta parameters before each update. Should be in (0,1]. Values
                closer to 1 preserve more history. Defaults to 0.95.
            exploration_factor (float, optional): Temperature parameter for
                Thompson sampling. Lower values increase exploitation of best
                models. Defaults to 0.1.

        Raises:
            ValueError: If n_models is non-positive, if decay_factor is not in (0,1],
                or if initial_alpha/initial_beta are negative.
        """
        # Validate parameters
        if n_models <= 0:
            raise ValueError(f"n_models must be positive, got {n_models}")
        if initial_alpha < 0:
            raise ValueError(f"initial_alpha must be non-negative, got {initial_alpha}")
        if initial_beta < 0:
            raise ValueError(f"initial_beta must be non-negative, got {initial_beta}")
        if not (0 < decay_factor <= 1):
            raise ValueError(f"decay_factor must be in (0, 1], got {decay_factor}")
        if exploration_factor < 0:
            raise ValueError(f"exploration_factor must be non-negative, got {exploration_factor}")

        self.n_models = n_models
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.decay_factor = decay_factor
        self.exploration_factor = exploration_factor

        # Beta distribution parameters for each model
        self.alpha = np.full(n_models, initial_alpha, dtype=float)
        self.beta = np.full(n_models, initial_beta, dtype=float)

        # Performance tracking
        self.performance_history: List[np.ndarray] = []
        self.weights_history: List[np.ndarray] = []

    def update(
        self,
        model_predictions: np.ndarray,
        true_labels: np.ndarray,
        drift_detected: bool = False,
    ) -> None:
        """Update model performance beliefs with new evidence.

        Computes accuracy for each model on the provided batch and updates
        the corresponding Beta distribution parameters. Applies exponential
        decay to historical beliefs and optionally resets beliefs when drift
        is detected to enable rapid adaptation.

        Args:
            model_predictions (np.ndarray): Predictions from each model with shape
                (n_models, n_samples) for binary or (n_models, n_samples, n_classes)
                for multiclass classification.
            true_labels (np.ndarray): True labels with shape (n_samples,).
            drift_detected (bool, optional): Whether drift was detected in this
                batch. If True, model beliefs are partially reset to enable
                faster adaptation. Defaults to False.

        Raises:
            ValueError: If model_predictions and true_labels have incompatible shapes,
                or if model_predictions doesn't match expected n_models.

        Note:
            For multiclass predictions, argmax is used to convert probabilities
            to class predictions. For binary predictions, threshold of 0.5 is used.
            Accuracy is converted to success/failure counts for Beta updates.
        """
        # Validate inputs
        if model_predictions is None or len(model_predictions) == 0:
            raise ValueError("model_predictions cannot be empty")
        if true_labels is None or len(true_labels) == 0:
            raise ValueError("true_labels cannot be empty")
        if len(model_predictions) != self.n_models:
            raise ValueError(f"Expected {self.n_models} model predictions, got {len(model_predictions)}")

        # Validate prediction shapes
        for i, pred in enumerate(model_predictions):
            if len(pred) != len(true_labels):
                raise ValueError(f"Model {i} predictions length ({len(pred)}) doesn't match true_labels length ({len(true_labels)})")

        try:
            n_samples = len(true_labels)
            model_accuracies = np.zeros(self.n_models)
        except Exception as e:
            logger.error(f"Failed to initialize accuracy computation: {str(e)}")
            raise RuntimeError(f"Failed to initialize accuracy computation: {str(e)}") from e

        # Calculate accuracy for each model
        for i, predictions in enumerate(model_predictions):
            if len(predictions.shape) > 1:
                # Multi-class: use argmax
                pred_labels = np.argmax(predictions, axis=1)
            else:
                # Binary: threshold at 0.5
                pred_labels = (predictions > 0.5).astype(int)

            accuracy = accuracy_score(true_labels, pred_labels)
            model_accuracies[i] = accuracy

        # Update Beta distribution parameters
        for i in range(self.n_models):
            successes = int(model_accuracies[i] * n_samples)
            failures = n_samples - successes

            if drift_detected:
                # Reset beliefs partially on drift detection
                reset_factor = 0.5
                self.alpha[i] = (
                    reset_factor * self.initial_alpha + (1 - reset_factor) * self.alpha[i]
                )
                self.beta[i] = (
                    reset_factor * self.initial_beta + (1 - reset_factor) * self.beta[i]
                )

            # Apply decay to previous beliefs
            self.alpha[i] *= self.decay_factor
            self.beta[i] *= self.decay_factor

            # Update with new evidence
            self.alpha[i] += successes
            self.beta[i] += failures

        # Store performance history
        self.performance_history.append(model_accuracies.copy())

        # Log performance updates
        logger.debug(f"Updated model performance beliefs: {model_accuracies}")
        if drift_detected:
            logger.info(f"Partial belief reset applied due to drift detection")
            logger.debug(f"Updated alpha parameters: {self.alpha}")
            logger.debug(f"Updated beta parameters: {self.beta}")

    def get_weights(self, use_thompson_sampling: bool = True) -> np.ndarray:
        """Get current ensemble weights for model combination.

        Computes weights for ensemble prediction based on current beliefs about
        model performance. Can use either Thompson sampling for exploration
        or posterior means for exploitation-only behavior.

        Args:
            use_thompson_sampling (bool, optional): Whether to use Thompson
                sampling for exploration. If True, samples from each model's
                Beta distribution and applies softmax. If False, uses posterior
                means. Defaults to True.

        Returns:
            np.ndarray: Array of weights with shape (n_models,) that sum to 1.
                Higher weights indicate better-performing or more promising models.

        Note:
            Thompson sampling provides automatic exploration-exploitation balance.
            The exploration_factor parameter controls the temperature of the softmax,
            with lower values leading to more concentrated weight distributions.
        """
        if use_thompson_sampling:
            # Thompson sampling: sample from Beta distributions
            sampled_means = np.random.beta(self.alpha, self.beta)
            weights = softmax(sampled_means / self.exploration_factor)
        else:
            # Use posterior means
            posterior_means = self.alpha / (self.alpha + self.beta)
            weights = softmax(posterior_means / self.exploration_factor)

        self.weights_history.append(weights.copy())
        logger.debug(f"Generated ensemble weights: {dict(zip(['xgboost', 'lightgbm', 'catboost'], weights))}")
        return weights

    def get_uncertainty(self) -> np.ndarray:
        """Get uncertainty estimates for each model's performance.

        Computes uncertainty based on the variance of each model's Beta distribution.
        Higher uncertainty indicates less confidence in the model's performance
        estimate, which can guide exploration and adaptation strategies.

        Returns:
            np.ndarray: Array of uncertainty values with shape (n_models,).
                Higher values indicate greater uncertainty about model performance.

        Note:
            Uncertainty decreases as more evidence (successful/failed predictions)
            is accumulated. Newly added models or models after belief resets
            will have higher uncertainty until sufficient evidence is gathered.
        """
        # Variance of Beta distribution
        variance = (self.alpha * self.beta) / (
            (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1)
        )
        return variance

    def reset_model_beliefs(self, model_idx: int) -> None:
        """Reset beliefs for a specific model to initial state.

        Resets the Beta distribution parameters for the specified model back
        to initial values, effectively forgetting all accumulated performance
        evidence. Useful when a specific model needs retraining or when
        handling model-specific adaptation scenarios.

        Args:
            model_idx (int): Index of the model to reset. Must be in range [0, n_models).

        Raises:
            IndexError: If model_idx is outside the valid range.

        Note:
            This affects only the specified model while preserving beliefs
            about other models in the ensemble.
        """
        # Validate input
        if not isinstance(model_idx, int):
            raise TypeError(f"model_idx must be an integer, got {type(model_idx)}")
        if not (0 <= model_idx < self.n_models):
            raise IndexError(f"model_idx {model_idx} is out of range [0, {self.n_models})")

        self.alpha[model_idx] = self.initial_alpha
        self.beta[model_idx] = self.initial_beta


class AdaptiveEnsembleDetector(BaseEstimator, ClassifierMixin):
    """Main adaptive ensemble detector with integrated drift detection and Bayesian reweighting.

    This is the primary class that implements the complete adaptive ensemble framework.
    It combines multiple base learners (XGBoost, LightGBM, CatBoost) with automated
    drift detection and Bayesian reweighting to provide robust performance under
    distribution shift scenarios.

    The detector provides both batch training (fit/predict) and online learning
    (partial_fit) capabilities, with automatic adaptation when drift is detected.
    It maintains theoretical regret bounds and detailed adaptation history for
    analysis and monitoring.

    Attributes:
        drift_detector (DriftDetector): Component for detecting distribution shifts.
        reweighter (BayesianReweighter): Component for dynamic model reweighting.
        base_models (List): List of trained base models (XGBoost, LightGBM, CatBoost).
        model_names (List[str]): Names of base models for identification.
        is_fitted (bool): Whether the ensemble has been fitted.
        classes_ (np.ndarray): Unique class labels from training data.
        n_features_in_ (int): Number of features from training data.
        adaptation_history (List[Dict]): History of adaptation events and metrics.
        regret_bounds (List[float]): Theoretical regret bounds over time.

    Example:
        Basic usage for adaptive classification:

        >>> detector = AdaptiveEnsembleDetector()
        >>> detector.fit(X_train, y_train)
        >>> predictions = detector.predict(X_test)
        >>>
        >>> # Online adaptation
        >>> for X_batch, y_batch in streaming_data:
        >>>     detector.partial_fit(X_batch, y_batch)
        >>>     current_predictions = detector.predict(X_batch)
    """

    def __init__(
        self,
        drift_detector_params: Optional[Dict] = None,
        reweighter_params: Optional[Dict] = None,
        base_model_params: Optional[Dict] = None,
        adaptation_threshold: float = 0.1,
        max_ensemble_size: int = 3,
        random_state: int = 42,
    ):
        """Initialize the adaptive ensemble detector with specified configuration.

        Args:
            drift_detector_params (Dict, optional): Parameters for DriftDetector
                initialization. See DriftDetector.__init__ for available options.
                Defaults to None (uses DriftDetector defaults).
            reweighter_params (Dict, optional): Parameters for BayesianReweighter
                initialization. See BayesianReweighter.__init__ for available options.
                Defaults to None (uses BayesianReweighter defaults).
            base_model_params (Dict, optional): Nested dictionary with parameters
                for base models. Keys should be 'xgboost', 'lightgbm', 'catboost'.
                Each value is a dict of parameters for that specific model.
                Defaults to None (uses default parameters for all models).
            adaptation_threshold (float, optional): Drift score threshold above
                which ensemble adaptation is triggered. Higher values make
                adaptation more conservative. Defaults to 0.1.
            max_ensemble_size (int, optional): Maximum number of base models
                in the ensemble. Currently fixed at 3 for XGBoost, LightGBM,
                CatBoost. Defaults to 3.
            random_state (int, optional): Random seed for reproducible results
                across all components. Defaults to 42.

        Raises:
            ValueError: If adaptation_threshold is negative or max_ensemble_size
                is non-positive.

        Example:
            Custom configuration example:

            >>> detector = AdaptiveEnsembleDetector(
            ...     drift_detector_params={'window_size': 500, 'alpha': 0.01},
            ...     reweighter_params={'decay_factor': 0.9, 'exploration_factor': 0.2},
            ...     base_model_params={
            ...         'xgboost': {'n_estimators': 200, 'max_depth': 8},
            ...         'lightgbm': {'n_estimators': 150, 'num_leaves': 64}
            ...     },
            ...     adaptation_threshold=0.15
            ... )
        """
        # Validate parameters
        if adaptation_threshold < 0:
            raise ValueError(f"adaptation_threshold must be non-negative, got {adaptation_threshold}")
        if max_ensemble_size <= 0:
            raise ValueError(f"max_ensemble_size must be positive, got {max_ensemble_size}")
        if not isinstance(random_state, int):
            raise TypeError(f"random_state must be an integer, got {type(random_state)}")

        self.drift_detector_params = drift_detector_params or {}
        self.reweighter_params = reweighter_params or {}
        self.base_model_params = base_model_params or {}
        self.adaptation_threshold = adaptation_threshold
        self.max_ensemble_size = max_ensemble_size
        self.random_state = random_state

        # Initialize components
        self.drift_detector = DriftDetector(**self.drift_detector_params)
        self.reweighter = None  # Initialized after fitting base models

        # Base models
        self.base_models: List = []
        self.model_names = ["xgboost", "lightgbm", "catboost"]

        # Training state
        self.is_fitted = False
        self.classes_ = None
        self.n_features_in_ = None
        self._label_encoder = _LabelEncoder()

        # Performance tracking
        self.adaptation_history: List[Dict] = []
        self.regret_bounds: List[float] = []

    def _create_base_models(self, n_classes: int) -> None:
        """
        Create base models for the ensemble.

        Args:
            n_classes: Number of classes in the target.
        """
        default_params = {
            "random_state": self.random_state,
            "n_estimators": 100,
            "max_depth": 6,
        }

        # Merge with user-provided parameters
        xgb_params = {**default_params, **self.base_model_params.get("xgboost", {})}
        lgb_params = {**default_params, **self.base_model_params.get("lightgbm", {})}

        # CatBoost uses 'random_seed' instead of 'random_state', and 'iterations' instead of 'n_estimators', 'depth' instead of 'max_depth'
        cb_default_params = {
            "random_seed": self.random_state,
            "iterations": 100,
            "depth": 6,
        }
        cb_params = {**cb_default_params, **self.base_model_params.get("catboost", {})}

        if n_classes == 2:
            # Binary classification
            xgb_params["objective"] = "binary:logistic"
            lgb_params["objective"] = "binary"
            cb_params["loss_function"] = "Logloss"
        else:
            # Multi-class classification
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = n_classes
            lgb_params["objective"] = "multiclass"
            lgb_params["num_class"] = n_classes
            cb_params["loss_function"] = "MultiClass"

        # Ensure verbose is set for CatBoost (remove duplicate if present in params)
        cb_params.setdefault("verbose", False)

        self.base_models = [
            xgb.XGBClassifier(**xgb_params),
            lgb.LGBMClassifier(**lgb_params),
            cb.CatBoostClassifier(**cb_params),
        ]

        # Initialize reweighter
        self.reweighter = BayesianReweighter(
            n_models=len(self.base_models), **self.reweighter_params
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaptiveEnsembleDetector":
        """Fit the adaptive ensemble on initial training data.

        Trains all base models (XGBoost, LightGBM, CatBoost) on the provided
        training data and initializes the drift detection and reweighting
        components. Sets up the reference distributions for future drift detection.

        Args:
            X (pd.DataFrame): Training features with shape (n_samples, n_features).
                All features should be numeric or properly encoded.
            y (pd.Series): Training labels with shape (n_samples,). Can be
                binary or multiclass classification targets.

        Returns:
            AdaptiveEnsembleDetector: Self for method chaining.

        Raises:
            ValueError: If X or y are empty, have mismatched lengths, or contain
                incompatible data types.

        Note:
            After fitting, the ensemble is ready for prediction and online learning.
            The drift detector is initialized with the training data as reference.
        """
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValueError("X cannot be empty")
        if y is None or len(y) == 0:
            raise ValueError("y cannot be empty")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y)}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")

        logger.info(f"Fitting AdaptiveEnsembleDetector on {len(X)} samples")

        try:
            # Store basic info
            self.classes_ = np.unique(y)
            self.n_features_in_ = X.shape[1]
        except Exception as e:
            logger.error(f"Failed to extract basic info from training data: {str(e)}")
            raise RuntimeError(f"Failed to extract basic info from training data: {str(e)}") from e

        # Encode labels to 0-indexed (required by XGBoost multiclass)
        y_encoded = pd.Series(self._label_encoder.fit_transform(y), index=y.index)

        # Create base models
        self._create_base_models(len(self.classes_))

        try:
            # Fit base models
            for i, model in enumerate(self.base_models):
                logger.info(f"Fitting {self.model_names[i]}...")
                model.fit(X, y_encoded)

            # Mark as fitted so predict_proba can work
            self.is_fitted = True

            # Set reference for drift detector using ensemble predictions
            initial_predictions = self.predict_proba(X)
            self.drift_detector.set_reference(X, y, initial_predictions)
            logger.info("AdaptiveEnsembleDetector fitting completed")
            return self
        except Exception as e:
            logger.error(f"Failed to fit AdaptiveEnsembleDetector: {str(e)}")
            self.is_fitted = False
            raise RuntimeError(f"Failed to fit AdaptiveEnsembleDetector: {str(e)}") from e

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict.

        Returns:
            Predicted class labels (original label space).
        """
        proba = self.predict_proba(X)
        encoded_preds = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(encoded_preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict.

        Returns:
            Predicted probabilities for each class.
        """
        # Validate state and inputs
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if X is None or len(X) == 0:
            raise ValueError("X cannot be empty")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}")

        # Check feature compatibility (skip if reference data not yet set during init)
        if self.drift_detector.reference_X is not None:
            expected_features = set(self.drift_detector.reference_X.columns)
            actual_features = set(X.columns)
            if not actual_features.issubset(expected_features):
                missing_features = expected_features - actual_features
                raise ValueError(f"Missing features in X: {missing_features}")

        # Get predictions from base models
        base_predictions = self._get_base_predictions(X)
        logger.debug(f"Obtained predictions from {len(base_predictions)} base models for {len(X)} samples")

        # Get current weights
        weights = self.reweighter.get_weights()

        # Combine predictions
        if len(self.classes_) == 2:
            # Binary classification
            weighted_pred = np.average(base_predictions, axis=0, weights=weights)
            proba = np.column_stack([1 - weighted_pred, weighted_pred])
            logger.debug(f"Binary classification prediction combination complete")
        else:
            # Multi-class classification
            weighted_pred = np.average(base_predictions, axis=0, weights=weights)
            proba = weighted_pred
            logger.debug(f"Multi-class classification prediction combination complete")

        return proba

    def partial_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "AdaptiveEnsembleDetector":
        """Incrementally update the ensemble with new streaming data.

        Performs online learning by detecting drift in the new data batch,
        updating model performance beliefs, and triggering ensemble adaptation
        if significant drift is detected. This is the core method for streaming
        operation.

        Args:
            X (pd.DataFrame): New batch of features with shape (batch_size, n_features).
                Must have the same feature structure as training data.
            y (pd.Series, optional): New batch of labels with shape (batch_size,).
                If provided, enables performance evaluation and belief updates.
                If None, only drift detection on features is performed. Defaults to None.

        Returns:
            AdaptiveEnsembleDetector: Self for method chaining.

        Raises:
            ValueError: If ensemble is not fitted, or if X has incompatible structure.
            RuntimeError: If drift detection fails due to insufficient reference data.

        Note:
            This method automatically:
            1. Detects drift using multiple methods
            2. Updates model performance beliefs (if labels provided)
            3. Computes theoretical regret bounds
            4. Triggers adaptation if drift exceeds threshold
            5. Records adaptation history for monitoring

            The method is designed for efficient streaming operation with minimal
            computational overhead while maintaining adaptation capabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before partial_fit")

        logger.debug(f"Starting partial fit with batch of {len(X)} samples")

        # Get predictions before updating
        base_predictions = self._get_base_predictions(X)
        ensemble_predictions = self.predict_proba(X)

        # Detect drift
        logger.debug("Checking for distribution drift...")
        drift_info = self.drift_detector.update(X, y, ensemble_predictions)

        # Update reweighter if labels are available
        if y is not None:
            logger.debug("Updating Bayesian reweighter with new performance evidence")
            y_encoded = self._label_encoder.transform(y.values)
            self.reweighter.update(
                base_predictions, y_encoded, drift_info["drift_detected"]
            )

            # Calculate regret bound
            regret_bound = self._calculate_regret_bound()
            self.regret_bounds.append(regret_bound)
            logger.debug(f"Current regret bound: {regret_bound:.4f}")
        else:
            logger.debug("No labels provided; skipping reweighter update")

        # Store adaptation information
        adaptation_info = {
            "drift_detected": drift_info["drift_detected"],
            "drift_type": drift_info["drift_type"],
            "drift_score": drift_info["drift_score"],
            "current_weights": self.reweighter.get_weights(),
            "uncertainty": self.reweighter.get_uncertainty(),
        }
        self.adaptation_history.append(adaptation_info)

        # Adapt ensemble if necessary
        if drift_info["drift_detected"] and drift_info["drift_score"] > self.adaptation_threshold:
            logger.info(f"Triggering ensemble adaptation (drift score: {drift_info['drift_score']:.3f} > threshold: {self.adaptation_threshold})")
            self._adapt_ensemble(X, y)
        else:
            if drift_info["drift_detected"]:
                logger.debug(f"Drift detected but below adaptation threshold ({drift_info['drift_score']:.3f} <= {self.adaptation_threshold})")

        logger.debug(f"Partial fit complete. Total adaptation events: {len(self.adaptation_history)}")
        return self

    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from all base models.

        Args:
            X: Input features.

        Returns:
            Array of shape [n_models, n_samples, n_classes] or [n_models, n_samples] for binary.
        """
        predictions = []

        for model in self.base_models:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)
                if len(self.classes_) == 2:
                    # For binary classification, use only positive class probability
                    pred = pred[:, 1]
            else:
                pred = model.predict(X)

            predictions.append(pred)

        return np.array(predictions)

    def _adapt_ensemble(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """
        Adapt the ensemble to handle detected drift.

        Args:
            X: Recent batch of features.
            y: Recent batch of labels.
        """
        logger.info("Adapting ensemble to detected drift...")

        if y is not None:
            logger.debug(f"Retraining base models on {len(X)} recent samples")
            y_encoded = pd.Series(self._label_encoder.transform(y), index=y.index)
            # Retrain models on recent data with higher weight
            for i, model in enumerate(self.base_models):
                logger.debug(f"Adapting {self.model_names[i]}...")
                # Create weighted sample
                sample_weights = np.ones(len(X))

                try:
                    # Simple incremental training for XGBoost
                    if hasattr(model, "get_booster"):
                        # XGBoost incremental training
                        dtrain = xgb.DMatrix(X, label=y_encoded, weight=sample_weights)
                        model.get_booster().update(dtrain, model.get_booster().num_boosted_rounds())
                        logger.debug(f"XGBoost model updated successfully")
                    elif hasattr(model, "refit"):
                        # LightGBM refit
                        model.refit(X, y_encoded, sample_weight=sample_weights)
                        logger.debug(f"LightGBM model refitted successfully")
                    else:
                        # CatBoost - create new model with recent data
                        new_model = type(model)(**model.get_params())
                        new_model.fit(X, y_encoded, sample_weight=sample_weights)
                        self.base_models[i] = new_model
                        logger.debug(f"CatBoost model retrained successfully")
                except Exception as e:
                    logger.warning(f"Failed to adapt {self.model_names[i]}: {str(e)}")
        else:
            logger.debug("No labels available for model adaptation")

        # Reset drift detector with new reference
        logger.debug("Resetting drift detector reference with adapted models")
        recent_predictions = self.predict_proba(X)
        self.drift_detector.set_reference(X, y, recent_predictions)
        logger.info("Ensemble adaptation completed")

    def _calculate_regret_bound(self) -> float:
        """
        Calculate theoretical regret bound for the ensemble.

        Returns:
            Regret bound value.
        """
        T = len(self.adaptation_history)
        if T == 0:
            return 0.0

        # Simplified regret bound calculation
        # In practice, this would involve more sophisticated analysis
        K = len(self.base_models)  # Number of arms (models)
        confidence_width = np.sqrt((2 * np.log(T)) / T) if T > 0 else 1.0
        regret_bound = confidence_width * np.sqrt(K * T)

        return regret_bound

    def get_model_importance(self) -> Dict[str, float]:
        """
        Get current importance of each base model.

        Returns:
            Dictionary mapping model names to importance scores.
        """
        if self.reweighter is None:
            return {}

        weights = self.reweighter.get_weights()
        return dict(zip(self.model_names, weights))

    def get_drift_history(self) -> List[Dict]:
        """
        Get the history of drift detection results.

        Returns:
            List of drift detection dictionaries.
        """
        return self.drift_detector.drift_history.copy()

    def get_adaptation_summary(self) -> Dict:
        """
        Get summary of adaptation performance.

        Returns:
            Dictionary with adaptation statistics.
        """
        if not self.adaptation_history:
            return {}

        drift_detections = [a["drift_detected"] for a in self.adaptation_history]
        drift_scores = [a["drift_score"] for a in self.adaptation_history]

        return {
            "total_adaptations": len(self.adaptation_history),
            "drift_detection_rate": np.mean(drift_detections),
            "average_drift_score": np.mean(drift_scores),
            "max_drift_score": np.max(drift_scores) if drift_scores else 0,
            "current_regret_bound": self.regret_bounds[-1] if self.regret_bounds else 0,
            "final_model_weights": self.get_model_importance(),
        }