import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import yaml
from typing import Optional
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import configurations
from config.config import (
    DEVICE,
    TransformerConfig,
    TrainingConfig,
    DataConfig
)

# Import model and utilities
from models.attention.transformer import Transformer
from utils.data_utils import create_dataloaders
from utils.training_utils import ModelTrainer


def setup_logging(model_name: str) -> logging.Logger:
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{model_name}_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_transformer(config: dict) -> Transformer:
    """Create transformer model from configuration"""
    return Transformer(
        input_size=config['input_size'],
        dec_seq_len=config['dec_seq_len'],
        num_predicted_features=config['num_predicted_features'],
        d_model=config['dim_val'],
        nhead=config['n_heads'],
        num_encoder_layers=config['n_encoder_layers'],
        num_decoder_layers=config['n_decoder_layers'],
        dim_feedforward=config['dim_feedforward_encoder'],
        dropout=config['dropout_encoder']
    )


def save_training_summary(trainer: ModelTrainer, config: dict, metrics: dict) -> None:
    """Save training summary including configuration and metrics"""
    # Convert numpy values to Python native types
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    summary = {
        'config': config,
        'metrics': convert_to_native_types(metrics),
        'training_history': {
            'train_losses': [float(loss) for loss in trainer.train_losses],
            'val_losses': [float(loss) for loss in trainer.val_losses],
            'best_val_loss': float(trainer.best_val_loss)
        }
    }

    summary_path = Path(f'outputs/transformer/results/training_summary.yaml')
    with open(summary_path, 'w') as f:
        yaml.dump(convert_to_native_types(summary), f)



def train_transformer(data_path: str = DataConfig.DATASET_PATH) -> Optional[ModelTrainer]:
    """Main training function for transformer model"""
    # Set random seeds for reproducibility
    torch.manual_seed(TrainingConfig.seed)
    np.random.seed(TrainingConfig.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TrainingConfig.seed)

    # Setup logging
    logger = setup_logging('transformer')
    logger.info(f"Starting transformer training")
    logger.info(f"Using device: {DEVICE}")

    # Create model configuration dictionary
    config = {
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
    }
    logger.info(f"Configuration: {config}")

    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            batch_size=TrainingConfig.batch_size,
            input_sequence_length=DataConfig.INPUT_SEQUENCE_LENGTH,
            target_sequence_length=DataConfig.TARGET_SEQUENCE_LENGTH,
            num_workers=DataConfig.NUM_WORKERS
        )
        logger.info("Data loaders created successfully")

        # Create model
        model = create_transformer(config)
        logger.info(f"Model created successfully: {model.__class__.__name__}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=config['learning_rate'],
            model_name='transformer'
        )

        # Train model
        logger.info("Starting training...")
        trainer.train(epochs=config['epochs'])
        logger.info("Training completed")

        # Evaluate model
        logger.info("Evaluating model...")
        metrics = trainer.evaluate()
        logger.info("\nTest Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.6f}")

        # Save training summary
        save_training_summary(trainer, config, metrics)
        logger.info("Training summary saved")

        return trainer

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return None


def main() -> None:
    """Main execution function"""
    try:
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Train transformer model
        trainer = train_transformer()
        if trainer is not None:
            print("\nTraining completed successfully!")
            print(f"Results saved in 'outputs/transformer' directory")
        else:
            print("\nTraining failed!")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
