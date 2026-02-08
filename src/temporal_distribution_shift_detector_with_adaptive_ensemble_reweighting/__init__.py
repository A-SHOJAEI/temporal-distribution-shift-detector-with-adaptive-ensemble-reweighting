"""
Temporal Distribution Shift Detector with Adaptive Ensemble Reweighting.

A research framework for detecting and adapting to distribution shift in streaming
tabular data through a novel ensemble approach with theoretical guarantees.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .models.model import AdaptiveEnsembleDetector
from .training.trainer import EnsembleTrainer
from .evaluation.metrics import DriftMetrics

__all__ = [
    "AdaptiveEnsembleDetector",
    "EnsembleTrainer",
    "DriftMetrics",
]