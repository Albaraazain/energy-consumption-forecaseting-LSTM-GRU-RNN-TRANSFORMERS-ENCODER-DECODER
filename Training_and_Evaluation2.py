import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm

from AttentionBasedModels.Decoder_only import Decoder
from AttentionBasedModels.Enc_and_Dec_Transformer import Transformer
from AttentionBasedModels.Encoder_only import Encoder
from CNNBasedModels.OneD_CNN import CNN_ForecastNet
from RNNBasedModels.GRU import GRU
from RNNBasedModels.LSTM import LSTM
from RNNBasedModels.RNN import Simple_RNN
from Preprocessing import CustomDataset, process_missing_and_duplicate_timestamps
from torch.utils.data import DataLoader

# Device configuration
device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")


class ModelTrainer:
    def __init__(self, model_name, model, train_loader, val_loader, test_loader, device,
                 input_sequence_length, target_sequence_length):
        self.model_name = model_name
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length

        # Results storage
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.actual = []
        self.metrics = {}

    def train_epoch(self, criterion, optimizer):
        self.model.train()
        batch_losses = []

        for batch in self.train_loader:
            optimizer.zero_grad()

            if isinstance(self.model, Transformer):  # Full Transformer
                src, trg, trg_y = batch
                src = src.to(self.device)
                trg = trg.to(self.device)
                trg_y = trg_y.to(self.device)

                # Generate masks
                tgt_mask = self.generate_square_subsequent_mask(
                    dim1=self.target_sequence_length,
                    dim2=self.target_sequence_length
                )
                src_mask = self.generate_square_subsequent_mask(
                    dim1=self.target_sequence_length,
                    dim2=self.input_sequence_length
                )

                outputs = self.model(src, trg, src_mask, tgt_mask)
                loss = criterion(outputs, trg_y)

            elif isinstance(self.model, Encoder):  # Encoder-only Transformer
                x_batch, _, y_batch = batch
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs, _ = self.model(x_batch)
                loss = criterion(outputs, y_batch)

            elif isinstance(self.model, Decoder):  # Decoder-only Transformer
                x_batch, _, y_batch = batch
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                S = x_batch.shape[1]
                mask = self.create_mask(S)

                outputs, _ = self.model(x_batch, mask)
                loss = criterion(outputs, y_batch)

            else:
                x_batch, _, y_batch = batch
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        return np.mean(batch_losses)

    def validate(self, criterion):
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                x_batch, _, y_batch = batch
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if isinstance(self.model, (Decoder, Transformer)):
                    S = x_batch.shape[1]
                    mask = self.create_mask(S)
                    outputs, _ = self.model(x_batch, mask)
                else:
                    outputs = self.model(x_batch)

                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def create_mask(self, size):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1).to(self.device)
        return mask

    def train(self, epochs, criterion, optimizer, scheduler):
        print(f"\nTraining {self.model_name}...")
        min_val_loss = float('inf')
        early_stop_count = 0

        for epoch in tqdm(range(epochs), desc=f"{self.model_name} Training"):
            train_loss = self.train_epoch(criterion, optimizer)
            val_loss = self.validate(criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= 20:
                print("Early stopping!")
                break

            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"train loss: {train_loss:.6f}, "
                  f"validation loss: {val_loss:.6f}")

    def generate_square_subsequent_mask(self, dim1: int, dim2: int) -> Tensor:
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1).to(self.device)

    def evaluate(self, criterion):
        self.model.eval()
        predictions = []
        actual = []
        test_losses = []

        with torch.no_grad():
            for batch in self.test_loader:
                if isinstance(self.model, Transformer):  # Full Transformer
                    src, trg, trg_y = batch
                    src = src.to(self.device)
                    trg = trg.to(self.device)
                    trg_y = trg_y.to(self.device)

                    # Generate masks
                    tgt_mask = self.generate_square_subsequent_mask(
                        dim1=self.target_sequence_length,
                        dim2=self.target_sequence_length
                    )
                    src_mask = self.generate_square_subsequent_mask(
                        dim1=self.target_sequence_length,
                        dim2=self.input_sequence_length
                    )

                    outputs = self.model(src, trg, src_mask, tgt_mask)
                    loss = criterion(outputs, trg_y)

                    predictions.append(outputs.detach().cpu().numpy())
                    actual.append(trg_y.detach().cpu().numpy())

                elif isinstance(self.model, Encoder):  # Encoder-only Transformer
                    x_batch, _, y_batch = batch
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs, _ = self.model(x_batch)
                    loss = criterion(outputs, y_batch)

                    predictions.append(outputs.detach().cpu().numpy())
                    actual.append(y_batch.detach().cpu().numpy())

                elif isinstance(self.model, Decoder):  # Decoder-only Transformer
                    x_batch, _, y_batch = batch
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    S = x_batch.shape[1]
                    mask = self.create_mask(S)

                    outputs, _ = self.model(x_batch, mask)
                    loss = criterion(outputs, y_batch)

                    predictions.append(outputs.detach().cpu().numpy())
                    actual.append(y_batch.detach().cpu().numpy())

                else:
                    x_batch, _, y_batch = batch
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    outputs = self.model(x_batch)
                    loss = criterion(outputs, y_batch)

                    predictions.append(outputs.detach().cpu().numpy())
                    actual.append(y_batch.detach().cpu().numpy())

                test_losses.append(loss.item())

        self.predictions = np.vstack(predictions)
        self.actual = np.vstack(actual)

        # Calculate metrics
        self.metrics = {
            'MSE': self.mean_squared_error(),
            'RMSE': self.root_mean_squared_error(),
            'MAE': self.mean_absolute_error()
        }

        return self.metrics

    def mean_squared_error(self):
        diff = self.predictions - self.actual
        return np.mean(diff ** 2)

    def root_mean_squared_error(self):
        return np.sqrt(self.mean_squared_error())

    def mean_absolute_error(self):
        return np.mean(np.abs(self.predictions - self.actual))

    def plot_predictions(self, points=96):
        k = pd.DataFrame(self.predictions.squeeze(-1))
        m = pd.DataFrame(self.actual.squeeze(-1))

        plt.figure(figsize=(12, 6))
        plt.plot([x for x in range(points)], k[0][:points], label="Predicted")
        plt.plot([x for x in range(points)], m[0][:points], label="Actual")
        plt.title(f"{self.model_name} Predicted vs. Actual Values")
        plt.legend(loc="upper right")
        plt.show()

    def plot_losses(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label="train loss")
        plt.plot(self.val_losses, label="validation loss")
        plt.title(f"{self.model_name} Training vs. Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


# Data setup
input_sequence_length = 168
target_sequence_length = 48

print("Loading and preprocessing data...")
path = "Datasets/DUQ_hourly.csv"
train_set, val_set, test_set = process_missing_and_duplicate_timestamps(filepath=path)

# Initialize datasets and dataloaders
train_dataset = CustomDataset(train_set, input_sequence_length, target_sequence_length,
                              multivariate=False, target_feature=0)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
test_dataset = CustomDataset(test_set, input_sequence_length, target_sequence_length,
                             multivariate=False, target_feature=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
val_dataset = CustomDataset(val_set, input_sequence_length, target_sequence_length,
                            multivariate=False, target_feature=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)

# Initialize models
models = {
    'RNN': {
        'model': Simple_RNN(in_dim=1, hid_dim=1740, out_dim=1, num_layers=1,
                            drop_rate=0.0009001480178615212),
        'epochs': 200,
        'lr': 0.0012874179807017348
    },
    'LSTM': {
        'model': LSTM(input_size=1, hidden_size=100, num_stacked_layers=3,
                      drop_rate=0.1),
        'epochs': 200,
        'lr': 0.00005
    },
    'GRU': {
        'model': GRU(in_dim=1, hid_dim=100, out_dim=1, num_layer=3,
                     drop_rate=0.1),
        'epochs': 200,
        'lr': 0.00005
    },
    '1D-CNN': {
        'model': CNN_ForecastNet(hidden_size=100, kernel_size=5, padding=2,
                                 drop_rate=0.1),
        'epochs': 200,
        'lr': 0.00005
    },
    'Encoder-only': {
        'model': Encoder(num_layers=1, D=32, H=1, hidden_mlp_dim=100,
                         inp_features=1, out_features=1, dropout_rate=0.1),
        'epochs': 50,
        'lr': 0.00005
    },
    'Decoder-only': {
        'model': Decoder(num_layers=1, D=32, H=4, hidden_mlp_dim=32,
                         inp_features=1, out_features=1, dropout_rate=0.1),
        'epochs': 200,
        'lr': 0.00005
    },
    'Full-Transformer': {
        'model': Transformer(input_size=1, dec_seq_len=48, num_predicted_features=1),
        'epochs': 200,
        'lr': 0.00005
    }
}

# Training and evaluation
results = {}
trainers = {}

print("\nStarting model training and evaluation...")
for model_name, model_config in models.items():
    print(f"\n{'-' * 50}")
    print(f"Processing {model_name}")

    # Initialize trainer
    trainer = ModelTrainer(
        model_name=model_name,
        model=model_config['model'],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        input_sequence_length=input_sequence_length,
        target_sequence_length=target_sequence_length
    )

    # Initialize criterion, optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=model_config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    # Train model
    trainer.train(
        epochs=model_config['epochs'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )

    # Evaluate model
    metrics = trainer.evaluate(criterion)
    results[model_name] = metrics
    trainers[model_name] = trainer

    # Plot individual model results
    trainer.plot_predictions()
    trainer.plot_losses()


# Compare all models
def plot_metric_comparison(results, metric):
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    values = [results[model][metric] for model in models]

    plt.bar(models, values)
    plt.title(f'{metric} Comparison Across Models')
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()


# Plot comparisons
print("\nPlotting model comparisons...")
metrics = ['MSE', 'RMSE', 'MAE']
for metric in metrics:
    plot_metric_comparison(results, metric)

# Create final comparison plot
plt.figure(figsize=(15, 8))
for model_name, trainer in trainers.items():
    k = pd.DataFrame(trainer.predictions.squeeze(-1))
    plt.plot(k[0][:96], label=model_name, alpha=0.7)

# Plot actual values
m = pd.DataFrame(trainers['RNN'].actual.squeeze(-1))
plt.plot(m[0][:96], label='Actual', color='black', linewidth=2)
plt.title('Prediction Comparison Across All Models')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Save results to CSV
results_df = pd.DataFrame(results).round(6)
results_df.to_csv('model_comparison_results.csv')

print("\nExecution Complete!")
print(f"Results saved to: model_comparison_results.csv")
print(f"Device used: {device}")
print("\nModel Configurations:")
for model_name, config in models.items():
    print(f"{model_name}: {config['epochs']} epochs, LR: {config['lr']}")


# Additional Analysis Utilities
class ModelAnalyzer:
    def __init__(self, trainers, results):
        self.trainers = trainers
        self.results = results

    def plot_all_learning_curves(self):
        """Plot learning curves for all models in one figure"""
        plt.figure(figsize=(15, 10))
        for model_name, trainer in self.trainers.items():
            plt.plot(trainer.train_losses, label=f"{model_name} (train)", alpha=0.7)
            plt.plot(trainer.val_losses, label=f"{model_name} (val)", linestyle='--', alpha=0.7)

        plt.title('Learning Curves Comparison Across All Models')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def calculate_convergence_metrics(self):
        """Calculate convergence metrics for each model"""
        convergence_metrics = {}
        for model_name, trainer in self.trainers.items():
            # Calculate epochs to best validation loss
            best_epoch = np.argmin(trainer.val_losses)
            best_val_loss = np.min(trainer.val_losses)

            # Calculate convergence speed (average loss improvement per epoch)
            initial_loss = trainer.val_losses[0]
            convergence_speed = (initial_loss - best_val_loss) / (best_epoch + 1)

            convergence_metrics[model_name] = {
                'epochs_to_best': best_epoch,
                'best_val_loss': best_val_loss,
                'convergence_speed': convergence_speed
            }

        return pd.DataFrame(convergence_metrics).round(6)

    def plot_prediction_distribution(self):
        """Plot distribution of predictions vs actual values"""
        plt.figure(figsize=(15, 10))
        for i, (model_name, trainer) in enumerate(self.trainers.items(), 1):
            plt.subplot(3, 3, i)
            plt.hist(trainer.predictions.flatten(), bins=50, alpha=0.5, label='Predictions')
            plt.hist(trainer.actual.flatten(), bins=50, alpha=0.5, label='Actual')
            plt.title(f'{model_name}')
            plt.legend()
        plt.tight_layout()
        plt.show()

    def calculate_error_statistics(self):
        """Calculate detailed error statistics for each model"""
        error_stats = {}
        for model_name, trainer in self.trainers.items():
            errors = trainer.predictions - trainer.actual
            error_stats[model_name] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(np.abs(errors)),
                'min_error': np.min(np.abs(errors)),
                'median_error': np.median(np.abs(errors))
            }

        return pd.DataFrame(error_stats).round(6)

    def generate_summary_report(self, output_file='model_analysis_report.txt'):
        """Generate a comprehensive analysis report"""
        with open(output_file, 'w') as f:
            f.write("=== Model Analysis Report ===\n\n")

            # Performance Metrics
            f.write("1. Performance Metrics:\n")
            f.write(pd.DataFrame(self.results).round(6).to_string())
            f.write("\n\n")

            # Convergence Metrics
            f.write("2. Convergence Metrics:\n")
            f.write(self.calculate_convergence_metrics().to_string())
            f.write("\n\n")

            # Error Statistics
            f.write("3. Error Statistics:\n")
            f.write(self.calculate_error_statistics().to_string())
            f.write("\n\n")

            # Model Rankings
            f.write("4. Model Rankings:\n")
            for metric in ['MSE', 'RMSE', 'MAE']:
                f.write(f"\n{metric} Ranking:\n")
                sorted_models = sorted(self.results.items(), key=lambda x: x[1][metric])
                for rank, (model, metrics) in enumerate(sorted_models, 1):
                    f.write(f"{rank}. {model}: {metrics[metric]:.6f}\n")

        print(f"Analysis report generated: {output_file}")


# Run additional analysis
print("\nRunning additional analysis...")
analyzer = ModelAnalyzer(trainers, results)

# Plot learning curves comparison
analyzer.plot_all_learning_curves()

# Plot prediction distributions
analyzer.plot_prediction_distribution()

# Generate and print convergence metrics
print("\nConvergence Metrics:")
print(analyzer.calculate_convergence_metrics())

# Generate and print error statistics
print("\nError Statistics:")
print(analyzer.calculate_error_statistics())

# Generate comprehensive report
analyzer.generate_summary_report()

# Save all plots
print("\nSaving visualizations...")
for model_name, trainer in trainers.items():
    plt.figure(figsize=(12, 6))
    trainer.plot_predictions()
    plt.savefig(f'predictions_{model_name}.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    trainer.plot_losses()
    plt.savefig(f'losses_{model_name}.png')
    plt.close()

print("\nAnalysis Complete!")
print("Generated files:")
print("1. model_comparison_results.csv - Raw metrics")
print("2. model_analysis_report.txt - Comprehensive analysis")
print("3. predictions_*.png - Individual model prediction plots")
print("4. losses_*.png - Individual model loss plots")

# Optional: Calculate and display memory usage
import psutil

process = psutil.Process()
print(f"\nMemory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
