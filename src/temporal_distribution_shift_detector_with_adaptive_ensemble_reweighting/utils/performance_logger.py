"""Performance logging utilities for monitoring system behavior.

This module provides utilities for structured performance logging, timing,
and monitoring of key metrics during training and evaluation.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

# Get dedicated performance logger
perf_logger = logging.getLogger("performance")


class PerformanceTracker:
    """Tracks and logs performance metrics throughout the application lifecycle."""

    def __init__(self):
        """Initialize the performance tracker."""
        self.metrics: Dict[str, Any] = {}
        self.timing_data: Dict[str, list] = {}
        self.counters: Dict[str, int] = {}

    def log_metric(self, name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None) -> None:
        """Log a performance metric.

        Args:
            name (str): Metric name.
            value (Union[float, int]): Metric value.
            tags (Optional[Dict[str, str]]): Additional tags for the metric.
        """
        metric_data = {
            "metric": name,
            "value": value,
            "timestamp": time.time()
        }
        if tags:
            metric_data.update(tags)

        self.metrics[name] = metric_data
        perf_logger.info(f"METRIC: {name}={value}" + (f" tags={tags}" if tags else ""))

    def log_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Log timing information.

        Args:
            name (str): Operation name.
            duration (float): Duration in seconds.
            tags (Optional[Dict[str, str]]): Additional tags for the timing.
        """
        if name not in self.timing_data:
            self.timing_data[name] = []

        self.timing_data[name].append(duration)

        timing_info = {
            "operation": name,
            "duration_seconds": duration,
            "timestamp": time.time()
        }
        if tags:
            timing_info.update(tags)

        perf_logger.info(f"TIMING: {name} took {duration:.4f}s" + (f" tags={tags}" if tags else ""))

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter.

        Args:
            name (str): Counter name.
            value (int): Increment value. Defaults to 1.
        """
        self.counters[name] = self.counters.get(name, 0) + value
        perf_logger.info(f"COUNTER: {name}={self.counters[name]} (+{value})")

    def get_timing_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get timing statistics for an operation.

        Args:
            name (str): Operation name.

        Returns:
            Optional[Dict[str, float]]: Statistics dictionary or None if no data.
        """
        if name not in self.timing_data or not self.timing_data[name]:
            return None

        durations = self.timing_data[name]
        return {
            "count": len(durations),
            "total": sum(durations),
            "mean": np.mean(durations),
            "median": np.median(durations),
            "min": min(durations),
            "max": max(durations),
            "std": np.std(durations)
        }

    def log_timing_summary(self, name: str) -> None:
        """Log summary statistics for timing data.

        Args:
            name (str): Operation name.
        """
        stats = self.get_timing_stats(name)
        if stats:
            perf_logger.info(f"TIMING_SUMMARY: {name} - "
                           f"count={stats['count']}, "
                           f"mean={stats['mean']:.4f}s, "
                           f"median={stats['median']:.4f}s, "
                           f"min={stats['min']:.4f}s, "
                           f"max={stats['max']:.4f}s, "
                           f"std={stats['std']:.4f}s")

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.

        Returns:
            Dict[str, Any]: Performance summary.
        """
        timing_summaries = {}
        for name in self.timing_data:
            timing_summaries[name] = self.get_timing_stats(name)

        return {
            "metrics": self.metrics,
            "timing_summaries": timing_summaries,
            "counters": self.counters,
            "timestamp": time.time()
        }


# Global performance tracker instance
_performance_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance.

    Returns:
        PerformanceTracker: Global performance tracker.
    """
    return _performance_tracker


@contextmanager
def timer(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for timing operations.

    Args:
        operation_name (str): Name of the operation being timed.
        tags (Optional[Dict[str, str]]): Additional tags for the timing.

    Example:
        >>> with timer("model_training", tags={"model": "xgboost"}):
        >>>     model.fit(X, y)
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time
        _performance_tracker.log_timing(operation_name, duration, tags)


def timed(operation_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function calls.

    Args:
        operation_name (Optional[str]): Name of the operation. If None, uses function name.
        tags (Optional[Dict[str, str]]): Additional tags for the timing.

    Returns:
        Callable: Decorated function.

    Example:
        >>> @timed("model_prediction")
        >>> def predict(self, X):
        >>>     return self.model.predict(X)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            with timer(name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_model_performance(
    model_name: str,
    metrics: Dict[str, float],
    data_info: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None
) -> None:
    """Log model performance metrics.

    Args:
        model_name (str): Name of the model.
        metrics (Dict[str, float]): Performance metrics.
        data_info (Optional[Dict[str, Any]]): Information about the data.
        model_info (Optional[Dict[str, Any]]): Information about the model.
    """
    tags = {"model": model_name}
    if data_info:
        tags.update(data_info)
    if model_info:
        tags.update(model_info)

    for metric_name, metric_value in metrics.items():
        _performance_tracker.log_metric(f"model_{metric_name}", metric_value, tags)


def log_drift_detection(
    drift_detected: bool,
    drift_type: str,
    drift_score: float,
    threshold: float,
    batch_size: int
) -> None:
    """Log drift detection event.

    Args:
        drift_detected (bool): Whether drift was detected.
        drift_type (str): Type of drift detected.
        drift_score (float): Drift detection score.
        threshold (float): Detection threshold.
        batch_size (int): Size of the batch processed.
    """
    tags = {
        "drift_type": drift_type,
        "batch_size": str(batch_size)
    }

    _performance_tracker.log_metric("drift_score", drift_score, tags)
    _performance_tracker.log_metric("drift_threshold", threshold, tags)

    if drift_detected:
        _performance_tracker.increment_counter("drift_detections_total")
        _performance_tracker.increment_counter(f"drift_detections_{drift_type}")
        perf_logger.info(f"DRIFT_DETECTED: type={drift_type}, score={drift_score:.4f}, threshold={threshold}")
    else:
        perf_logger.debug(f"DRIFT_CHECK: type={drift_type}, score={drift_score:.4f}, threshold={threshold}")


def log_adaptation_event(
    adaptation_triggered: bool,
    drift_score: float,
    adaptation_threshold: float,
    model_weights: Dict[str, float],
    uncertainty: Optional[Dict[str, float]] = None
) -> None:
    """Log ensemble adaptation event.

    Args:
        adaptation_triggered (bool): Whether adaptation was triggered.
        drift_score (float): Drift score that triggered (or didn't trigger) adaptation.
        adaptation_threshold (float): Threshold for adaptation.
        model_weights (Dict[str, float]): Current model weights.
        uncertainty (Optional[Dict[str, float]]): Model uncertainty estimates.
    """
    if adaptation_triggered:
        _performance_tracker.increment_counter("adaptations_total")
        perf_logger.info(f"ADAPTATION: triggered=True, drift_score={drift_score:.4f}, "
                        f"threshold={adaptation_threshold}, weights={model_weights}")
    else:
        perf_logger.debug(f"ADAPTATION: triggered=False, drift_score={drift_score:.4f}, "
                         f"threshold={adaptation_threshold}")

    # Log individual model weights
    for model_name, weight in model_weights.items():
        _performance_tracker.log_metric(f"weight_{model_name}", weight, {"event": "adaptation"})

    # Log uncertainty if available
    if uncertainty:
        for model_name, unc in uncertainty.items():
            _performance_tracker.log_metric(f"uncertainty_{model_name}", unc, {"event": "adaptation"})


def log_batch_processing(
    batch_size: int,
    processing_time: float,
    has_labels: bool,
    adaptation_triggered: bool = False
) -> None:
    """Log batch processing information.

    Args:
        batch_size (int): Size of the processed batch.
        processing_time (float): Time taken to process the batch.
        has_labels (bool): Whether the batch included labels.
        adaptation_triggered (bool): Whether adaptation was triggered.
    """
    throughput = batch_size / processing_time if processing_time > 0 else 0

    tags = {
        "has_labels": str(has_labels),
        "adaptation_triggered": str(adaptation_triggered)
    }

    _performance_tracker.log_metric("batch_size", batch_size, tags)
    _performance_tracker.log_metric("batch_processing_time", processing_time, tags)
    _performance_tracker.log_metric("batch_throughput", throughput, tags)
    _performance_tracker.increment_counter("batches_processed")

    perf_logger.info(f"BATCH: size={batch_size}, time={processing_time:.4f}s, "
                    f"throughput={throughput:.1f} samples/s, labels={has_labels}")