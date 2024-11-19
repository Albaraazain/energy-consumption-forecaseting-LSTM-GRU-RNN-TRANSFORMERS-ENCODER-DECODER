import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

from config.config import (
    DEVICE, INPUT_SEQUENCE_LENGTH, TARGET_SEQUENCE_LENGTH,
    BATCH_SIZE, model_configs
)
from models.attention.decoder import Decoder
from models.attention.encoder import Encoder
from models.attention.transformer import Transformer
from models.cnn.cnn_model import CNNForecastNet
from utils.training_utils import ModelTrainer
from utils.data_utils import create_data_loaders

def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train energy consumption forecasting models')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['decoder', 'encoder', 'transformer', 'cnn'],
        help='Type of model to train'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/models_config.yaml',
        help='Path to model configuration file'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to dataset'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs to train'
    )
    return parser

def load_model_config(model_type: str, config_path: str) -> Dict[str, Any]:
    """Load model configuration from file or use default"""
    try:
        with open(config_path, 'r') as f:
            configs = yaml.safe_load(f)
            return configs.get(model_type, model_configs[model_type])
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration")
        return model_configs[model_type]

def create_model(model_type: str, config: Dict[str, Any]) -> torch.nn.Module:
    """Create model instance based on type and configuration"""
    if model_type == 'decoder':
        return Decoder(
            num_layers=config['num_layers'],
            D=config['D'],
            H=config['H'],
            hidden_mlp_dim=config['hidden_mlp_dim'],
            inp_features=config['inp_features'],
            out_features=config['out_features'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'encoder':
        return Encoder(
            num_layers=config['num_layers'],
            d_model=config['D'],
            num_heads=config['H'],
            d_ff=config['hidden_mlp_dim'],
            inp_features=config['inp_features'],
            out_features=config['out_features'],
            dropout_rate=config['dropout_rate']
        )
    elif model_type == 'transformer':
        return Transformer(
            input_size=config['input_size'],
            dec_seq_len=config['dec_seq_len'],
            dim_val=config.get('dim_val', 32),
            n_encoder_layers=config.get('n_encoder_layers', 1),
            n_decoder_layers=config.get('n_decoder_layers', 1),
            n_heads=config.get('n_heads', 4),
            dropout_encoder=config.get('dropout_encoder', 0.2),
            dropout_decoder=config.get('dropout_decoder', 0.2)
        )
    elif model_type == 'cnn':
        return CNNForecastNet(
            hidden_size=config['hidden_size'],
            kernel_size=config['kernel_size'],
            padding=config['padding'],
            drop_rate=config['drop_rate']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    # Parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_model_config(args.model, args.config)
    if args.epochs is not None:
        config['epochs'] = args.epochs

    # Set up directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    # Print training information
    print(f"\nStarting training for {args.model.upper()} model")
    print(f"Using device: {DEVICE}")
    print(f"Configuration: {config}")

    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            input_sequence_length=INPUT_SEQUENCE_LENGTH,
            target_sequence_length=TARGET_SEQUENCE_LENGTH
        )
        print("Data loaders created successfully")

        # Create model
        model = create_model(args.model, config)
        print(f"Model created successfully: {model.__class__.__name__}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        # Initialize trainer
        trainer = ModelTrainer(
            model=model,
            model_name=args.model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=config['learning_rate']
        )

        # Train model
        trainer.train(
            epochs=config['epochs'],
            early_stopping_patience=20
        )

        # Evaluate model
        metrics = trainer.evaluate()
        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")

        # Plot training history
        trainer.plot_training_history()

        # Get and save training summary
        summary = trainer.get_training_summary()
        summary_path = Path(f'logs/{args.model}_training_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f)
        print(f"\nTraining summary saved to {summary_path}")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return

if __name__ == "__main__":
    main()