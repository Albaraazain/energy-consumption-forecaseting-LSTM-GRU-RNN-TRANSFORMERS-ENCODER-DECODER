import json

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from typing import Optional, Tuple, Dict

from models.attention.transformer import Transformer
from utils.data_utils import process_data, denormalize_data
from config.config import DEVICE, TransformerConfig, DataConfig

def load_model(checkpoint_path: str) -> Transformer:
    """Load the trained transformer model"""
    model = Transformer(
        input_size=TransformerConfig.input_size,
        dec_seq_len=TransformerConfig.dec_seq_len,
        num_predicted_features=TransformerConfig.num_predicted_features,
        d_model=TransformerConfig.dim_val,
        nhead=TransformerConfig.n_heads,
        num_encoder_layers=TransformerConfig.n_encoder_layers,
        num_decoder_layers=TransformerConfig.n_decoder_layers,
        dim_feedforward=TransformerConfig.dim_feedforward_encoder,
        dropout=0.0  # Set to 0 for inference
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

    # Handle state dict loading
    state_dict = checkpoint['model_state_dict']
    if 'positional_encoding' in state_dict:
        # Copy the positional encoding to both encoder and decoder positions
        pos_encoding = state_dict.pop('positional_encoding')
        state_dict['encoder_pos'] = pos_encoding
        state_dict['decoder_pos'] = pos_encoding[:, :TransformerConfig.dec_seq_len, :]

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def predict(model: Transformer, input_sequence: np.ndarray) -> np.ndarray:
    """Make predictions using the transformer model"""
    # Ensure input_sequence is 2D [batch_size, sequence_length]
    if input_sequence.ndim == 1:
        src = input_sequence.reshape(1, -1)  # Add batch dimension
    else:
        src = input_sequence

    # Add feature dimension if needed
    if src.shape[-1] != 1:
        src = src[..., np.newaxis]

    # Convert to tensor
    src = torch.FloatTensor(src).to(DEVICE)  # [batch_size, seq_len, 1]
    tgt = torch.zeros((1, TransformerConfig.dec_seq_len, 1), device=DEVICE)

    # Generate mask
    tgt_mask = model.generate_square_subsequent_mask(tgt.size(1))

    with torch.no_grad():
        output = model(src, tgt, None, tgt_mask)

    # Return predictions, removing batch dimension
    return output.cpu().numpy()[0, :, 0]




def plot_prediction(actual: np.ndarray, predicted: np.ndarray, save_path: Optional[str] = None):
    """Plot the prediction results with error analysis"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot 1: Actual vs Predicted
    time_steps = np.arange(len(actual))
    ax1.plot(time_steps, actual, 'b-', label='Actual', marker='o')
    ax1.plot(time_steps, predicted, 'r--', label='Predicted', marker='o')
    ax1.set_title('48-Hour Energy Consumption Forecast')
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Energy Consumption')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Error Analysis
    errors = predicted - actual
    sns.histplot(errors, ax=ax2, bins=20, kde=True)
    ax2.set_title('Prediction Error Distribution')
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_predictions(actual: np.ndarray, predicted: np.ndarray,
                        predicted_std: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None) -> dict:
    """Detailed analysis of predictions"""
    # Calculate hourly statistics
    hourly_errors = predicted - actual
    hourly_percentage_errors = (hourly_errors / actual) * 100

    print("\nHourly Analysis:")
    print(f"Average Hourly Error: {np.mean(hourly_errors):.2f}")
    print(f"Median Hourly Error: {np.median(hourly_errors):.2f}")
    print(f"Average Percentage Error: {np.mean(hourly_percentage_errors):.2f}%")

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot 1: Actual vs Predicted with uncertainty
    plt.subplot(2, 1, 1)
    hours = np.arange(len(actual))
    plt.plot(hours, actual, label='Actual', marker='o')
    plt.plot(hours, predicted, label='Predicted', marker='o')

    if predicted_std is not None:
        plt.fill_between(hours,
                         predicted - 2*predicted_std,
                         predicted + 2*predicted_std,
                         alpha=0.2, label='95% Confidence')

    plt.title('48-Hour Prediction with Uncertainty')
    plt.xlabel('Hours')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.grid(True)

    # Plot 2: Error Analysis
    plt.subplot(2, 1, 2)
    plt.plot(hours, hourly_errors, label='Prediction Error', color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Prediction Error Over Time')
    plt.xlabel('Hours')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

    return {
        'hourly_errors': hourly_errors.tolist(),
        'percentage_errors': hourly_percentage_errors.tolist(),
        'first_day_rmse': float(np.sqrt(mean_squared_error(actual[:24], predicted[:24]))),
        'second_day_rmse': float(np.sqrt(mean_squared_error(actual[24:], predicted[24:])))
    }

def predict_with_uncertainty(model: Transformer, input_sequence: np.ndarray,
                             n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with uncertainty estimation"""
    predictions = []

    for _ in range(n_samples):
        # Enable dropout during inference for uncertainty estimation
        model.train()  # Enable dropout
        with torch.no_grad():
            pred = predict(model, input_sequence)
            predictions.append(pred)

    model.eval()  # Set back to eval mode

    # Calculate mean and std of predictions
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)

    return mean_pred, std_pred

def print_statistics(actual: np.ndarray, predicted: np.ndarray):
    """Print detailed statistics about the predictions"""
    print("\nDetailed Statistics:")
    print(f"{'Metric':<20} {'Actual':<15} {'Predicted':<15}")
    print("-" * 50)
    print(f"{'Mean':<20} {np.mean(actual):<15.2f} {np.mean(predicted):<15.2f}")
    print(f"{'Std Dev':<20} {np.std(actual):<15.2f} {np.std(predicted):<15.2f}")
    print(f"{'Min':<20} {np.min(actual):<15.2f} {np.min(predicted):<15.2f}")
    print(f"{'Max':<20} {np.max(actual):<15.2f} {np.max(predicted):<15.2f}")
    print(f"{'Median':<20} {np.median(actual):<15.2f} {np.median(predicted):<15.2f}")

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate all evaluation metrics"""
    metrics = {
        'mse': float(mean_squared_error(targets, predictions)),
        'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
        'mae': float(mean_absolute_error(targets, predictions)),
        'r2': float(r2_score(targets, predictions))
    }
    return metrics

def main():
    # Path to your trained model
    checkpoint_path = "outputs/transformer/checkpoints/checkpoint_best.pt"

    try:
        # Load the model
        model = load_model(checkpoint_path)
        print("Model loaded successfully")

        # Get test data
        _, _, test_data = process_data(DataConfig.DATASET_PATH)
        print("Test data loaded successfully")

        # Make multiple predictions and analyze
        n_samples = 5
        all_predictions = []
        all_actuals = []
        all_analyses = []

        # Create output directory for plots
        output_dir = Path("outputs/transformer/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_samples):
            print(f"\n{'='*50}")
            print(f"Sample {i+1} Analysis")
            print(f"{'='*50}")

            # Select random starting point
            start_idx = np.random.randint(0, len(test_data) - DataConfig.INPUT_SEQUENCE_LENGTH - DataConfig.TARGET_SEQUENCE_LENGTH)

            # Get input sequence and actual future values
            input_sequence = test_data[start_idx:start_idx + DataConfig.INPUT_SEQUENCE_LENGTH]
            actual_future = test_data[start_idx + DataConfig.INPUT_SEQUENCE_LENGTH:
                                      start_idx + DataConfig.INPUT_SEQUENCE_LENGTH + DataConfig.TARGET_SEQUENCE_LENGTH]

            # Make prediction with uncertainty
            predicted_mean, predicted_std = predict_with_uncertainty(model, input_sequence, n_samples=10)

            # Denormalize the values
            actual_future_denorm = denormalize_data(actual_future, DataConfig.DATASET_PATH)
            predicted_mean_denorm = denormalize_data(predicted_mean, DataConfig.DATASET_PATH)
            predicted_std_denorm = predicted_std * np.std(actual_future_denorm)  # Scale uncertainty

            # Store results
            all_predictions.append(predicted_mean_denorm)
            all_actuals.append(actual_future_denorm)

            # Calculate metrics
            metrics = calculate_metrics(actual_future_denorm, predicted_mean_denorm)
            print("\nPrediction Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            # Print detailed statistics
            print_statistics(actual_future_denorm, predicted_mean_denorm)

            # Analyze predictions and save plot
            analysis = analyze_predictions(
                actual_future_denorm,
                predicted_mean_denorm,
                predicted_std_denorm,
                save_path=str(output_dir / f"sample_{i+1}_analysis.png")  # Convert Path to string
            )
            all_analyses.append(analysis)

            # Print pattern analysis
            print("\nPattern Analysis:")
            print(f"First 24h RMSE: {analysis['first_day_rmse']:.2f}")
            print(f"Last 24h RMSE: {analysis['second_day_rmse']:.2f}")
            print(f"Mean Percentage Error: {np.mean(analysis['percentage_errors']):.2f}%")
            print(f"Median Percentage Error: {np.median(analysis['percentage_errors']):.2f}%")

        # Calculate and print aggregate statistics
        print(f"\n{'='*50}")
        print("Aggregate Statistics Across All Samples")
        print(f"{'='*50}")

        all_predictions = np.concatenate(all_predictions)
        all_actuals = np.concatenate(all_actuals)

        # Calculate error patterns
        error_patterns = all_predictions - all_actuals

        aggregate_metrics = calculate_metrics(all_actuals, all_predictions)
        print("\nAggregate Metrics:")
        for metric, value in aggregate_metrics.items():
            print(f"{metric}: {value:.4f}")

        # Calculate average pattern metrics
        avg_first_day_rmse = np.mean([a['first_day_rmse'] for a in all_analyses])
        avg_second_day_rmse = np.mean([a['second_day_rmse'] for a in all_analyses])
        print("\nAverage Pattern Metrics:")
        print(f"Average First Day RMSE: {avg_first_day_rmse:.2f}")
        print(f"Average Second Day RMSE: {avg_second_day_rmse:.2f}")

        # Save aggregate results
        results = {
            'aggregate_metrics': aggregate_metrics,
            'avg_first_day_rmse': float(avg_first_day_rmse),
            'avg_second_day_rmse': float(avg_second_day_rmse),
            'sample_metrics': [calculate_metrics(actual, pred)
                               for actual, pred in zip(all_actuals.reshape(-1, 48),
                                                       all_predictions.reshape(-1, 48))]
        }

        # Save results to JSON
        with open(output_dir / "analysis_results.json", 'w') as f:
            json.dump(results, f, indent=4)

        # Create final summary plot
        plt.figure(figsize=(15, 10))

        # Plot 1: Average errors by sample
        plt.subplot(2, 1, 1)
        sample_means = [np.mean(np.abs(all_predictions[i:i+48] - all_actuals[i:i+48]))
                        for i in range(0, len(all_predictions), 48)]
        plt.bar(range(len(sample_means)), sample_means)
        plt.title('Average MAE by Sample')
        plt.xlabel('Sample')
        plt.ylabel('Mean Absolute Error')

        # Plot 2: Error distribution
        plt.subplot(2, 1, 2)
        sns.histplot(error_patterns, bins=30, kde=True)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(str(output_dir / "aggregate_analysis.png"))
        plt.close()

        print(f"\nAnalysis complete. Results saved to {output_dir}")

    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Create output directories if they don't exist
    Path("outputs/transformer/analysis").mkdir(parents=True, exist_ok=True)

    main()