"""Custom neural network components and attention mechanisms for ensemble models.

This module provides specialized neural network components used in the adaptive
ensemble framework, including:
- Attention mechanisms for feature importance weighting
- Custom loss functions for drift-aware training
- Neural network modules for model confidence estimation
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)


class FeatureAttention(nn.Module):
    """Attention mechanism for dynamic feature importance weighting.

    This component learns to assign importance weights to features based on
    the current data distribution, allowing the model to adapt its focus
    as the distribution shifts over time.

    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Dimension of hidden representation
        num_heads (int): Number of attention heads for multi-head attention
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention components
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing attention weights and weighted features.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            weighted_features: Attention-weighted features
            attention_weights: Computed attention weights for interpretability
        """
        batch_size = x.size(0)

        # Compute query, key, value projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.hidden_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Output projection
        output = self.output_proj(attended)

        # Residual connection and layer normalization
        weighted_features = self.layer_norm(x + output)

        return weighted_features, attention_weights


class ConfidenceEstimator(nn.Module):
    """Neural network for estimating prediction confidence under drift.

    This component learns to predict the reliability of base learner predictions
    based on features of the input and historical performance patterns. It helps
    the ensemble adapt by down-weighting unreliable predictions during drift.

    Args:
        input_dim (int): Number of input features
        num_base_learners (int): Number of base learners in the ensemble
    """

    def __init__(self, input_dim: int, num_base_learners: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.num_base_learners = num_base_learners

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Per-learner confidence heads
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output confidence in [0, 1]
            )
            for _ in range(num_base_learners)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute confidence scores for each base learner.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            confidence_scores: Confidence for each learner, shape (batch_size, num_base_learners)
        """
        # Extract shared features
        features = self.feature_extractor(x)

        # Compute per-learner confidence
        confidences = [head(features) for head in self.confidence_heads]
        confidence_scores = torch.cat(confidences, dim=-1)

        return confidence_scores


class DriftAwareLoss(nn.Module):
    """Custom loss function that adapts based on detected drift.

    This loss function combines standard cross-entropy with a drift penalty term
    that encourages the model to maintain stable predictions on reference data
    while adapting to new patterns.

    Args:
        drift_penalty_weight (float): Weight for drift penalty term
        smoothing (float): Label smoothing factor
    """

    def __init__(self, drift_penalty_weight: float = 0.1, smoothing: float = 0.1):
        super().__init__()
        self.drift_penalty_weight = drift_penalty_weight
        self.smoothing = smoothing

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reference_predictions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute drift-aware loss.

        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            reference_predictions: Predictions on reference data for drift penalty

        Returns:
            Total loss value
        """
        # Standard cross-entropy with label smoothing
        ce_loss = F.cross_entropy(predictions, targets, label_smoothing=self.smoothing)

        # Add drift penalty if reference predictions provided
        if reference_predictions is not None:
            # KL divergence between current and reference predictions
            drift_penalty = F.kl_div(
                F.log_softmax(predictions, dim=-1),
                F.softmax(reference_predictions, dim=-1),
                reduction='batchmean'
            )
            total_loss = ce_loss + self.drift_penalty_weight * drift_penalty
        else:
            total_loss = ce_loss

        return total_loss


class AdaptiveFeatureSelector(BaseEstimator, TransformerMixin):
    """Sklearn-compatible feature selector that adapts to distribution drift.

    This component dynamically selects the most relevant features based on
    their stability and predictive power under the current data distribution.
    It can be used as a preprocessing step in the ensemble pipeline.

    Args:
        n_features (int): Number of features to select
        stability_weight (float): Weight for feature stability vs predictive power
        window_size (int): Size of window for computing feature statistics
    """

    def __init__(
        self,
        n_features: int = 20,
        stability_weight: float = 0.3,
        window_size: int = 1000
    ):
        self.n_features = n_features
        self.stability_weight = stability_weight
        self.window_size = window_size
        self.selected_features_ = None
        self.feature_scores_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AdaptiveFeatureSelector':
        """Fit the feature selector.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values (optional)

        Returns:
            self
        """
        n_samples, n_total_features = X.shape

        # Compute feature statistics
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)

        # Compute feature stability (inverse of coefficient of variation)
        stability_scores = 1 / (1 + feature_stds / (feature_means + 1e-8))

        # Compute predictive power (correlation with target if provided)
        if y is not None:
            # For classification, use point-biserial correlation approximation
            predictive_scores = np.abs([np.corrcoef(X[:, i], y)[0, 1]
                                       for i in range(n_total_features)])
        else:
            # Use variance as proxy for predictive power
            predictive_scores = feature_stds

        # Combine scores
        self.feature_scores_ = (
            self.stability_weight * stability_scores +
            (1 - self.stability_weight) * predictive_scores
        )

        # Select top features
        self.selected_features_ = np.argsort(self.feature_scores_)[-self.n_features:]

        logger.info(f"Selected {self.n_features} features based on adaptive scoring")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by selecting features.

        Args:
            X: Data to transform

        Returns:
            Transformed data with selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")

        return X[:, self.selected_features_]

    def get_feature_importance(self) -> np.ndarray:
        """Get importance scores for all features.

        Returns:
            Feature importance scores
        """
        return self.feature_scores_
