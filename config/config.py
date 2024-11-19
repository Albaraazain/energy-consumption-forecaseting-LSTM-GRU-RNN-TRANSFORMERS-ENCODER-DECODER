# config/config.py

import torch
from pathlib import Path
from typing import Dict, Any

def get_device() -> torch.device:
    """Get the device to use for training"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

DEVICE = get_device()

class DataConfig:
    """Data configuration"""
    DATASET_PATH = "Datasets/DUQ_hourly.csv"
    INPUT_SEQUENCE_LENGTH = 168  # 1 week of hourly data
    TARGET_SEQUENCE_LENGTH = 48  # 2 days of predictions
    BATCH_SIZE = 32
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    NUM_WORKERS = 0  # Set to 0 for Windows
    PIN_MEMORY = True
    DROP_LAST = True

class TransformerConfig:
    """Transformer model configuration"""
    input_size = 1
    dec_seq_len = DataConfig.TARGET_SEQUENCE_LENGTH
    dim_val = 128  # Model dimension
    n_encoder_layers = 4
    n_decoder_layers = 4
    n_heads = 8
    dropout_encoder = 0.2
    dropout_decoder = 0.2
    dropout_pos_enc = 0.1
    dim_feedforward_encoder = 512
    dim_feedforward_decoder = 512
    num_predicted_features = 1
    learning_rate = 1e-4
    epochs = 100

class EncoderConfig:
    """Encoder-only model configuration"""
    num_layers = 4
    D = 128  # Model dimension
    H = 8    # Number of heads
    hidden_mlp_dim = 512
    inp_features = 1
    out_features = 1
    dropout_rate = 0.2
    learning_rate = 1e-4
    epochs = 100

class DecoderConfig:
    """Decoder-only model configuration"""
    num_layers = 4
    D = 128  # Model dimension
    H = 8    # Number of heads
    hidden_mlp_dim = 512
    inp_features = 1
    out_features = 1
    dropout_rate = 0.2
    learning_rate = 1e-4
    epochs = 100

class CNNConfig:
    """CNN model configuration"""
    hidden_size = 128
    kernel_size = 5
    padding = 2
    drop_rate = 0.1
    num_layers = 3
    learning_rate = 1e-4
    epochs = 100

class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    seed = 42
    batch_size = 32
    epochs = 100
    learning_rate = 1e-4

    # Early stopping and scheduling
    early_stopping_patience = 30
    scheduler_patience = 5
    scheduler_factor = 0.5
    min_learning_rate = 1e-6

    # Optimizer settings
    weight_decay = 0.01
    gradient_clip_val = 1.0
    warmup_steps = 4000
    beta1 = 0.9
    beta2 = 0.98
    epsilon = 1e-9

    # Loss settings
    mse_weight = 0.5
    mae_weight = 0.5

    # Data splits
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15

    # Checkpoint settings
    save_every_n_epochs = 10
    max_checkpoints = 5
    keep_best_n_checkpoints = 3

    # Logging settings
    log_every_n_steps = 100
    log_every_n_epochs = 1

    # Validation settings
    validate_every_n_steps = 500
    eval_batch_size = 64

    # Misc settings
    num_workers = 0  # For data loading
    pin_memory = True
    drop_last = True

    # Directories
    checkpoint_dir = Path("checkpoints")
    log_dir = Path("logs")
    results_dir = Path("results")
    model_dir = Path("models")

    # Create directories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Model architecture parameters
    input_sequence_length = 168  # 1 week of hourly data
    target_sequence_length = 48  # 2 days predictions
    hidden_size = 128
    num_layers = 4
    num_heads = 8
    dropout = 0.1

    # Training features
    use_teacher_forcing = True
    teacher_forcing_ratio = 0.5
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0

    # Regularization
    l1_lambda = 0.0
    l2_lambda = 0.01
    dropout_rate = 0.1

    # Learning rate scheduling
    use_warmup = True
    warmup_steps = 4000
    lr_scheduler_type = "OneCycleLR"  # or "ReduceLROnPlateau"
    lr_scheduler_params = {
        "OneCycleLR": {
            "max_lr": learning_rate,
            "pct_start": 0.3,
            "anneal_strategy": "cos",
            "cycle_momentum": True,
            "base_momentum": 0.85,
            "max_momentum": 0.95,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
        },
        "ReduceLROnPlateau": {
            "mode": "min",
            "factor": scheduler_factor,
            "patience": scheduler_patience,
            "verbose": True,
            "min_lr": min_learning_rate,
        }
    }

    # Early stopping configuration
    early_stopping_params = {
        "patience": early_stopping_patience,
        "min_delta": 1e-6,
        "mode": "min",
        "verbose": True
    }

    # Loss function configuration
    loss_params = {
        "mse_weight": mse_weight,
        "mae_weight": mae_weight,
        "use_huber_loss": False,
        "huber_delta": 1.0
    }

    # Metrics to track
    metrics = [
        "mse",
        "rmse",
        "mae",
        "mape",
        "smape",
        "r2_score"
    ]

    # Validation metrics
    validation_metrics = {
        "primary": "val_loss",
        "secondary": ["val_rmse", "val_mae"]
    }

    # Inference settings
    inference_batch_size = 32
    use_sliding_window = True
    sliding_window_size = input_sequence_length
    sliding_window_stride = target_sequence_length

    # Experiment tracking
    use_wandb = False
    wandb_project = "energy_forecasting"
    wandb_entity = None

    # Reproducibility settings
    deterministic = True
    benchmark = False

    @classmethod
    def get_scheduler_params(cls, scheduler_type: str) -> dict:
        """Get parameters for specific scheduler type"""
        return cls.lr_scheduler_params[scheduler_type]

    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        for directory in [cls.checkpoint_dir, cls.log_dir,
                          cls.results_dir, cls.model_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def to_dict(cls) -> dict:
        """Convert configuration to dictionary"""
        return {key: value for key, value in cls.__dict__.items()
                if not key.startswith('__')}
# Combined configurations for easy access
model_configs: Dict[str, Dict[str, Any]] = {
    'transformer': {
        'input_size': TransformerConfig.input_size,
        'dec_seq_len': TransformerConfig.dec_seq_len,
        'dim_val': TransformerConfig.dim_val,
        'n_encoder_layers': TransformerConfig.n_encoder_layers,
        'n_decoder_layers': TransformerConfig.n_decoder_layers,
        'n_heads': TransformerConfig.n_heads,
        'dropout_encoder': TransformerConfig.dropout_encoder,
        'dropout_decoder': TransformerConfig.dropout_decoder,
        'dropout_pos_enc': TransformerConfig.dropout_pos_enc,
        'dim_feedforward_encoder': TransformerConfig.dim_feedforward_encoder,
        'dim_feedforward_decoder': TransformerConfig.dim_feedforward_decoder,
        'num_predicted_features': TransformerConfig.num_predicted_features,
        'learning_rate': TransformerConfig.learning_rate,
        'epochs': TransformerConfig.epochs
    },
    'encoder': {
        'num_layers': EncoderConfig.num_layers,
        'D': EncoderConfig.D,
        'H': EncoderConfig.H,
        'hidden_mlp_dim': EncoderConfig.hidden_mlp_dim,
        'inp_features': EncoderConfig.inp_features,
        'out_features': EncoderConfig.out_features,
        'dropout_rate': EncoderConfig.dropout_rate,
        'learning_rate': EncoderConfig.learning_rate,
        'epochs': EncoderConfig.epochs
    },
    'decoder': {
        'num_layers': DecoderConfig.num_layers,
        'D': DecoderConfig.D,
        'H': DecoderConfig.H,
        'hidden_mlp_dim': DecoderConfig.hidden_mlp_dim,
        'inp_features': DecoderConfig.inp_features,
        'out_features': DecoderConfig.out_features,
        'dropout_rate': DecoderConfig.dropout_rate,
        'learning_rate': DecoderConfig.learning_rate,
        'epochs': DecoderConfig.epochs
    },
    'cnn': {
        'hidden_size': CNNConfig.hidden_size,
        'kernel_size': CNNConfig.kernel_size,
        'padding': CNNConfig.padding,
        'drop_rate': CNNConfig.drop_rate,
        'num_layers': CNNConfig.num_layers,
        'learning_rate': CNNConfig.learning_rate,
        'epochs': CNNConfig.epochs
    }
}

# Optional: Environment variables that can override configurations
import os

def get_env_var(var_name: str, default: Any) -> Any:
    """Get environment variable with default value"""
    return os.environ.get(var_name, default)

# Environment overrides
ENV_OVERRIDES = {
    'BATCH_SIZE': int(get_env_var('BATCH_SIZE', DataConfig.BATCH_SIZE)),
    'LEARNING_RATE': float(get_env_var('LEARNING_RATE', TrainingConfig.learning_rate)),
    'NUM_WORKERS': int(get_env_var('NUM_WORKERS', DataConfig.NUM_WORKERS)),
    'EPOCHS': int(get_env_var('EPOCHS', TransformerConfig.epochs))
}

# Apply environment overrides
for key, value in ENV_OVERRIDES.items():
    if key in locals():
        locals()[key] = value

# Validation functions
def validate_config():
    """Validate configuration settings"""
    assert DataConfig.TRAIN_SPLIT + DataConfig.VAL_SPLIT + DataConfig.TEST_SPLIT == 1.0, \
        "Data splits must sum to 1.0"
    assert DataConfig.INPUT_SEQUENCE_LENGTH > 0, \
        "Input sequence length must be positive"
    assert DataConfig.TARGET_SEQUENCE_LENGTH > 0, \
        "Target sequence length must be positive"
    assert DataConfig.BATCH_SIZE > 0, \
        "Batch size must be positive"

    for model_name, config in model_configs.items():
        assert 'learning_rate' in config, \
            f"Learning rate missing for {model_name}"
        assert 'epochs' in config, \
            f"Epochs missing for {model_name}"

# Run validation
validate_config()