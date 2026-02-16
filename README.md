# Temporal Distribution Shift Detector with Adaptive Ensemble Reweighting

A research framework for detecting and adapting to distribution shift in streaming tabular data through a novel ensemble approach that continuously monitors feature importance drift, prediction confidence degradation, and label distribution changes. The system implements a Bayesian online learning mechanism that dynamically reweights base learners based on their recent performance on detected data regimes, with theoretical guarantees on regret bounds under non-stationary distributions.

## Key Features

- **Multi-method drift detection** combining statistical tests and performance monitoring
- **Bayesian ensemble reweighting** with Thompson sampling for exploration-exploitation
- **Adaptive learning** without explicit retraining triggers
- **Theoretical regret bounds** for non-stationary environments
- **Production-ready** with MLflow integration and comprehensive testing

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting import AdaptiveEnsembleDetector
from temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting.data import DataLoader

# Load data
loader = DataLoader()
X, y = loader.load_covertype()

# Initialize detector
detector = AdaptiveEnsembleDetector(
    drift_detector_params={"detection_threshold": 0.1},
    reweighter_params={"exploration_factor": 0.1}
)

# Train initial model
detector.fit(X[:1000], y[:1000])

# Stream processing with adaptation
for i in range(1000, len(X), 100):
    batch_X = X[i:i+100]
    batch_y = y[i:i+100]

    # Make predictions
    predictions = detector.predict(batch_X)

    # Update with new data (detects drift and adapts)
    detector.partial_fit(batch_X, batch_y)
```

### Training Script

```bash
# Batch training
python scripts/train.py --config configs/default.yaml

# Streaming training with drift simulation
python scripts/train.py --config configs/default.yaml --streaming --max-samples 10000
```

### Evaluation Script

```bash
# Comprehensive evaluation
python scripts/evaluate.py --config configs/default.yaml --n-trials 5 --plot --save-results
```

## Results

**Dataset:** Covertype (581,012 samples, 54 features, 7 classes)

| Metric | Value |
|--------|-------|
| Validation Accuracy | 0.787 |
| Test Accuracy | 0.758 |

### Ensemble Weights (Adaptive Reweighting)

| Base Learner | Weight |
|--------------|--------|
| XGBoost | 70.7% |
| LightGBM | 16.4% |
| CatBoost | 12.9% |

The model dynamically reweights ensemble members based on detected distribution drift, allowing the system to adapt to temporal shifts in the data stream without explicit retraining triggers.

## Core Components

### AdaptiveEnsembleDetector
Main framework combining XGBoost, LightGBM, and CatBoost with dynamic reweighting.

### DriftDetector
Multi-method drift detection using KS tests, distribution monitoring, and prediction confidence analysis.

### BayesianReweighter
Thompson sampling-based ensemble weighting with theoretical regret guarantees.

### PrequentialEvaluator
Test-then-train evaluation for streaming scenarios.

## Architecture

```
├── src/temporal_distribution_shift_detector_with_adaptive_ensemble_reweighting/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Core ensemble and drift detection models
│   ├── training/       # Training pipeline with MLflow integration
│   ├── evaluation/     # Metrics and analysis tools
│   └── utils/          # Configuration and utilities
├── tests/              # Comprehensive test suite
├── scripts/            # Training and evaluation scripts
├── notebooks/          # Exploration and analysis notebooks
└── configs/            # Configuration files
```

## Methodology

This work introduces a novel approach to handling temporal distribution shifts through adaptive ensemble reweighting with Bayesian online learning. Unlike traditional drift detectors that trigger explicit model retraining, our method continuously adapts ensemble weights based on recent performance patterns across detected data regimes.

The key innovation is a Thompson sampling-based Bayesian reweighting mechanism that maintains Beta distributions over each base learner's success probability. As data streams through the system, the drift detector (combining KS tests, prediction confidence monitoring, and label distribution analysis) identifies regime changes. The reweighter then updates its beliefs about each learner's effectiveness in the current regime and samples new weights accordingly, enabling both exploration of underperforming models and exploitation of currently effective ones.

This approach provides theoretical regret bounds in non-stationary environments while avoiding the computational overhead and catastrophic forgetting risks associated with full model retraining. The ensemble naturally adapts to covariate shifts, label shifts, and concept drift without requiring labeled data for each incoming batch.

## Research Contributions

1. **Novel ensemble reweighting** approach using Bayesian online learning with Thompson sampling
2. **Theoretical regret bounds** for non-stationary streaming environments
3. **Multi-modal drift detection** combining statistical tests and performance monitoring
4. **Adaptation without retraining** avoiding catastrophic forgetting and computational overhead

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{temporal_distribution_shift_detector,
  title={Temporal Distribution Shift Detector with Adaptive Ensemble Reweighting},
  author={Alireza Shojaei},
  year={2026},
  url={https://github.com/A-SHOJAEI/temporal-distribution-shift-detector-with-adaptive-ensemble-reweighting}
}
```
