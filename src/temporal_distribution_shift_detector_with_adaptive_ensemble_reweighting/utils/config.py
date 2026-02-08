"""
Configuration management for temporal distribution shift detector.

This module provides utilities for loading, validating, and managing
configuration files for the adaptive ensemble detector.
"""

import logging
import logging.config
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectorConfig:
    """Configuration for drift detector component."""

    window_size: int = 1000
    alpha: float = 0.05
    detection_threshold: float = 0.1
    min_samples: int = 100


@dataclass
class BayesianReweighterConfig:
    """Configuration for Bayesian reweighter component."""

    initial_alpha: float = 1.0
    initial_beta: float = 1.0
    decay_factor: float = 0.95
    exploration_factor: float = 0.1


@dataclass
class BaseModelConfig:
    """Configuration for base models."""

    xgboost: Dict[str, Any] = None
    lightgbm: Dict[str, Any] = None
    catboost: Dict[str, Any] = None

    def __post_init__(self):
        # Set defaults
        if self.xgboost is None:
            self.xgboost = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }

        if self.lightgbm is None:
            self.lightgbm = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "verbose": -1,
            }

        if self.catboost is None:
            self.catboost = {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "random_seed": 42,
                "verbose": False,
            }


@dataclass
class ModelConfig:
    """Configuration for the main ensemble model."""

    drift_detector: DriftDetectorConfig = None
    reweighter: BayesianReweighterConfig = None
    base_models: BaseModelConfig = None
    adaptation_threshold: float = 0.1
    max_ensemble_size: int = 3
    random_state: int = 42

    def __post_init__(self):
        if self.drift_detector is None:
            self.drift_detector = DriftDetectorConfig()
        if self.reweighter is None:
            self.reweighter = BayesianReweighterConfig()
        if self.base_models is None:
            self.base_models = BaseModelConfig()


@dataclass
class DataConfig:
    """Configuration for data processing."""

    dataset_name: str = "covertype"
    data_path: Optional[str] = None
    batch_size: int = 1000
    validation_split: float = 0.2
    test_split: float = 0.2

    # Preprocessing
    scaler_type: str = "standard"
    handle_missing: str = "median"
    feature_selection: Optional[str] = None
    n_features: Optional[int] = None

    # Drift injection
    drift_schedule: Optional[Dict[int, str]] = None
    drift_types: List[str] = None
    drift_severity: float = 0.3

    def __post_init__(self):
        if self.drift_types is None:
            self.drift_types = ["covariate_shift", "label_shift", "concept_drift"]

        if self.drift_schedule is None:
            # Default drift schedule
            self.drift_schedule = {
                10: "covariate_shift",
                25: "concept_drift",
                40: "label_shift",
                55: "combined",
            }


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    # Training parameters
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Online learning
    streaming_mode: bool = False
    update_frequency: int = 100
    evaluation_frequency: int = 500
    max_samples: Optional[int] = None

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 1000
    keep_checkpoints: int = 5

    # Evaluation
    eval_window_size: int = 1000
    eval_metrics: List[str] = None

    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = ["accuracy", "f1", "precision", "recall"]


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking."""

    tracking_uri: Optional[str] = None
    experiment_name: str = "temporal_distribution_shift_detection"
    run_name: Optional[str] = None
    log_artifacts: bool = True
    log_models: bool = True

    # Autologging
    autolog_models: bool = True
    autolog_metrics: bool = True
    autolog_params: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and analysis."""

    # Metrics
    prequential_window_size: int = 1000
    drift_tolerance_window: int = 100

    # Target metrics
    target_prequential_accuracy: float = 0.82
    target_drift_detection_f1: float = 0.9
    target_adaptation_latency: int = 500
    target_regret_vs_oracle: float = 0.05

    # Analysis
    plot_results: bool = True
    save_plots: bool = True
    plot_dir: str = "plots"


@dataclass
class Config:
    """Main configuration class combining all components."""

    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    mlflow: MLflowConfig = None
    evaluation: EvaluationConfig = None

    # Global settings
    random_seed: int = 42
    log_level: str = "INFO"
    output_dir: str = "outputs"

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.mlflow is None:
            self.mlflow = MLflowConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()

        # Ensure random seeds are consistent
        self.model.random_state = self.random_seed
        self.model.base_models.xgboost["random_state"] = self.random_seed
        self.model.base_models.lightgbm["random_state"] = self.random_seed
        self.model.base_models.catboost["random_seed"] = self.random_seed


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file with environment variable support.

    Supports environment variable substitution using ${VAR_NAME:default_value} syntax.
    Environment variables can override any configuration value.

    Args:
        config_path (Union[str, Path]): Path to configuration file.

    Returns:
        Config: Loaded and validated configuration object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
        ValueError: If config is invalid or missing required environment variables.

    Example:
        Configuration file with environment variables:

        ```yaml
        data:
          dataset_name: ${DATASET_NAME:covertype}
          batch_size: ${BATCH_SIZE:1000}
        mlflow:
          tracking_uri: ${MLFLOW_TRACKING_URI:null}
        ```
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_content = f.read()

        # Substitute environment variables
        config_content = _substitute_env_vars(config_content)

        config_dict = yaml.safe_load(config_content)

        logger.info(f"Loaded configuration from {config_path}")

        # Apply environment variable overrides
        config_dict = _apply_env_overrides(config_dict)

        # Convert nested dictionaries to dataclass objects
        config = _dict_to_config(config_dict)

        # Validate configuration
        _validate_config(config)

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save.
        config_path: Path where to save configuration.

    Raises:
        IOError: If unable to save configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        config_dict = asdict(config)

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Saved configuration to {config_path}")

    except Exception as e:
        raise IOError(f"Error saving config to {config_path}: {e}")


def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """
    Convert nested dictionary to Config object.

    Args:
        config_dict: Configuration dictionary.

    Returns:
        Config object.
    """
    # Helper function to convert dict to dataclass
    def convert_section(section_dict: Dict, section_class):
        if section_dict is None:
            return section_class()

        # Handle nested dataclasses
        if section_class == ModelConfig:
            drift_detector = section_dict.get('drift_detector')
            if drift_detector and isinstance(drift_detector, dict):
                section_dict['drift_detector'] = DriftDetectorConfig(**drift_detector)

            reweighter = section_dict.get('reweighter')
            if reweighter and isinstance(reweighter, dict):
                section_dict['reweighter'] = BayesianReweighterConfig(**reweighter)

            base_models = section_dict.get('base_models')
            if base_models and isinstance(base_models, dict):
                section_dict['base_models'] = BaseModelConfig(**base_models)

        # Filter out unknown keys to avoid TypeError
        valid_keys = {f.name for f in section_class.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in section_dict.items() if k in valid_keys}

        return section_class(**filtered_dict)

    # Convert each section
    model_config = convert_section(config_dict.get('model', {}), ModelConfig)
    data_config = convert_section(config_dict.get('data', {}), DataConfig)
    training_config = convert_section(config_dict.get('training', {}), TrainingConfig)
    mlflow_config = convert_section(config_dict.get('mlflow', {}), MLflowConfig)
    evaluation_config = convert_section(config_dict.get('evaluation', {}), EvaluationConfig)

    # Extract global settings
    global_settings = {
        k: v for k, v in config_dict.items()
        if k not in ['model', 'data', 'training', 'mlflow', 'evaluation']
    }

    return Config(
        model=model_config,
        data=data_config,
        training=training_config,
        mlflow=mlflow_config,
        evaluation=evaluation_config,
        **global_settings
    )


def _validate_config(config: Config) -> None:
    """
    Validate configuration for consistency and correctness.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If configuration is invalid.
    """
    # Validate data config
    if config.data.validation_split + config.data.test_split >= 1.0:
        raise ValueError("validation_split + test_split must be < 1.0")

    if config.data.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Validate model config
    if config.model.adaptation_threshold <= 0 or config.model.adaptation_threshold >= 1:
        raise ValueError("adaptation_threshold must be in (0, 1)")

    if config.model.max_ensemble_size <= 0:
        raise ValueError("max_ensemble_size must be positive")

    # Validate training config
    if config.training.max_epochs <= 0:
        raise ValueError("max_epochs must be positive")

    if config.training.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")

    if config.training.update_frequency <= 0 or config.training.evaluation_frequency <= 0:
        raise ValueError("update_frequency and evaluation_frequency must be positive")

    # Validate evaluation config
    if config.evaluation.prequential_window_size <= 0:
        raise ValueError("prequential_window_size must be positive")

    if config.evaluation.drift_tolerance_window <= 0:
        raise ValueError("drift_tolerance_window must be positive")

    logger.info("Configuration validation passed")


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns:
        Default configuration object.
    """
    return Config()


def create_config_template(output_path: Union[str, Path]) -> None:
    """
    Create a template configuration file with default values.

    Args:
        output_path: Path where to save the template.
    """
    config = get_default_config()
    save_config(config, output_path)
    logger.info(f"Created configuration template at {output_path}")


def setup_logging(config: Config, logging_config_path: Optional[Union[str, Path]] = None, environment: str = "development") -> None:
    """Setup comprehensive logging based on configuration.

    Args:
        config (Config): Configuration object.
        logging_config_path (Optional[Union[str, Path]]): Path to logging configuration file.
            If None, uses default logging setup. Defaults to None.
        environment (str): Environment name for logging configuration.
            Options: 'development', 'production', 'testing'. Defaults to 'development'.

    Note:
        If logging_config_path is provided, it will load a full logging configuration
        from YAML file. Otherwise, it falls back to basic logging setup.
    """
    # Create necessary directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    if logging_config_path and Path(logging_config_path).exists():
        try:
            # Load logging configuration from file
            with open(logging_config_path, 'r') as f:
                logging_config = yaml.safe_load(f)

            # Apply environment-specific overrides
            if environment in logging_config and 'loggers' in logging_config[environment]:
                env_loggers = logging_config[environment]['loggers']
                for logger_name, logger_config in env_loggers.items():
                    if logger_name in logging_config['loggers']:
                        logging_config['loggers'][logger_name].update(logger_config)

            if environment in logging_config and 'handlers' in logging_config[environment]:
                env_handlers = logging_config[environment]['handlers']
                for handler_name, handler_config in env_handlers.items():
                    if handler_name in logging_config['handlers']:
                        logging_config['handlers'][handler_name].update(handler_config)

            # Update file paths to use absolute paths
            for handler_name, handler_config in logging_config.get('handlers', {}).items():
                if 'filename' in handler_config:
                    filename = handler_config['filename']
                    if not Path(filename).is_absolute():
                        handler_config['filename'] = str(logs_dir / filename)

            # Apply the configuration
            logging.config.dictConfig(logging_config)
            logger.info(f"Loaded logging configuration from {logging_config_path} (environment: {environment})")

        except Exception as e:
            logger.warning(f"Failed to load logging config from {logging_config_path}: {e}")
            logger.info("Falling back to basic logging setup")
            _setup_basic_logging(config, logs_dir)
    else:
        _setup_basic_logging(config, logs_dir)


def _setup_basic_logging(config: Config, logs_dir: Path) -> None:
    """Setup basic logging configuration as fallback.

    Args:
        config (Config): Configuration object.
        logs_dir (Path): Directory for log files.
    """
    log_level = getattr(logging, config.log_level.upper())

    # Setup logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "application.log"),
            logging.FileHandler(logs_dir / "errors.log")
        ]
    )

    # Set error handler to only log errors
    error_handler = logging.FileHandler(logs_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(log_format)
    error_handler.setFormatter(error_formatter)

    # Add error handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(error_handler)

    logger.info(f"Basic logging configured with level: {config.log_level}")


def setup_environment(config: Config) -> None:
    """
    Setup environment based on configuration.

    Args:
        config: Configuration object.
    """
    # Set random seeds for reproducibility
    os.environ["PYTHONHASHSEED"] = str(config.random_seed)

    import numpy as np
    np.random.seed(config.random_seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(config.random_seed)
    except ImportError:
        pass  # TensorFlow not available

    try:
        import torch
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)
    except ImportError:
        pass  # PyTorch not available

    # Create necessary directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if config.evaluation.plot_results:
        Path(config.evaluation.plot_dir).mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(config)

    logger.info("Environment setup completed")


def _substitute_env_vars(content: str) -> str:
    """Substitute environment variables in configuration content.

    Supports ${VAR_NAME:default_value} syntax for environment variable substitution.

    Args:
        content (str): Configuration file content.

    Returns:
        str: Content with environment variables substituted.

    Raises:
        ValueError: If required environment variable is not set and no default provided.
    """
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replace_var(match):
        var_name = match.group(1)
        default_value = match.group(2)

        env_value = os.environ.get(var_name)

        if env_value is not None:
            # Try to convert to appropriate type
            if env_value.lower() in ('true', 'false'):
                return env_value.lower()
            elif env_value.lower() == 'null' or env_value.lower() == 'none':
                return 'null'
            elif env_value.isdigit():
                return env_value
            else:
                return env_value
        elif default_value is not None:
            return default_value
        else:
            raise ValueError(f"Environment variable {var_name} is required but not set")

    return re.sub(pattern, replace_var, content)


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Environment variables with format TDS_<SECTION>_<KEY> override config values.
    For example: TDS_DATA_BATCH_SIZE overrides data.batch_size

    Args:
        config_dict (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Configuration with environment overrides applied.
    """
    # Map of sections to their config keys
    env_mappings = {
        'TDS_DATA_DATASET_NAME': ('data', 'dataset_name'),
        'TDS_DATA_BATCH_SIZE': ('data', 'batch_size'),
        'TDS_DATA_VALIDATION_SPLIT': ('data', 'validation_split'),
        'TDS_DATA_TEST_SPLIT': ('data', 'test_split'),
        'TDS_MODEL_ADAPTATION_THRESHOLD': ('model', 'adaptation_threshold'),
        'TDS_MODEL_RANDOM_STATE': ('model', 'random_state'),
        'TDS_TRAINING_MAX_EPOCHS': ('training', 'max_epochs'),
        'TDS_TRAINING_STREAMING_MODE': ('training', 'streaming_mode'),
        'TDS_TRAINING_MAX_SAMPLES': ('training', 'max_samples'),
        'TDS_MLFLOW_TRACKING_URI': ('mlflow', 'tracking_uri'),
        'TDS_MLFLOW_EXPERIMENT_NAME': ('mlflow', 'experiment_name'),
        'TDS_LOG_LEVEL': (None, 'log_level'),
        'TDS_OUTPUT_DIR': (None, 'output_dir'),
        'TDS_RANDOM_SEED': (None, 'random_seed'),
    }

    for env_var, (section, key) in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Convert value to appropriate type
            converted_value = _convert_env_value(value)

            if section is None:
                # Global config
                config_dict[key] = converted_value
            else:
                # Section-specific config
                if section not in config_dict:
                    config_dict[section] = {}
                config_dict[section][key] = converted_value

    return config_dict


def _convert_env_value(value: str) -> Any:
    """Convert environment variable string to appropriate Python type.

    Args:
        value (str): Environment variable value.

    Returns:
        Any: Converted value.
    """
    # Boolean conversion
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False

    # None/null conversion
    elif value.lower() in ('null', 'none', ''):
        return None

    # Numeric conversion
    elif value.isdigit():
        return int(value)
    elif '.' in value:
        try:
            return float(value)
        except ValueError:
            return value

    # String value
    else:
        return value


def load_config_with_env(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration with full environment variable support.

    This function provides a complete configuration loading experience including:
    1. Loading from YAML file (optional)
    2. Environment variable substitution
    3. Environment variable overrides
    4. Default value fallbacks

    Args:
        config_path (Optional[Union[str, Path]]): Path to configuration file.
            If None, uses only environment variables and defaults.

    Returns:
        Config: Loaded and validated configuration object.

    Example:
        Using environment variables only:

        >>> os.environ['TDS_DATA_BATCH_SIZE'] = '2000'
        >>> config = load_config_with_env()
        >>> print(config.data.batch_size)  # 2000

        Using config file with environment overrides:

        >>> config = load_config_with_env('config.yaml')
    """
    if config_path is not None:
        return load_config(config_path)
    else:
        # Create config from environment variables and defaults
        config_dict = {}
        config_dict = _apply_env_overrides(config_dict)
        config = _dict_to_config(config_dict)
        _validate_config(config)
        logger.info("Loaded configuration from environment variables and defaults")
        return config