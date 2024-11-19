import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional

from config.config import DEVICE, TrainingConfig

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 20, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            learning_rate: float = TrainingConfig.learning_rate,
            model_name: str = "transformer"
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model_name = model_name

        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=TrainingConfig.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=TrainingConfig.scheduler_factor,
            patience=TrainingConfig.scheduler_patience,
            min_lr=TrainingConfig.min_learning_rate,
            verbose=True
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None
        self.best_val_loss = float('inf')

        # Create directories
        self.setup_directories()

    def setup_directories(self) -> None:
        """Create necessary directories for saving results"""
        dirs = ['checkpoints', 'results', 'plots']
        for dir_name in dirs:
            Path(f'outputs/{self.model_name}/{dir_name}').mkdir(parents=True, exist_ok=True)

    def _step(self, batch: tuple) -> tuple:
        """Process one batch"""
        src, tgt, tgt_y = [x.to(DEVICE) for x in batch]

        # Create masks for transformer
        tgt_mask = self.model.generate_square_subsequent_mask(tgt.size(1))
        src_mask = None  # We're not using source mask for now

        # Forward pass
        output = self.model(src, tgt, src_mask, tgt_mask)

        # Ensure output and target shapes match
        if output.shape != tgt_y.shape:
            output = output.view(tgt_y.shape)

        return output, tgt_y

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc="Training") as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()

                output, target = self._step(batch)
                loss = self.criterion(output, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    TrainingConfig.gradient_clip_val
                )

                self.optimizer.step()
                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

                # Save checkpoint periodically
                if (batch_idx + 1) % 100 == 0:
                    self.save_checkpoint(f"batch_{batch_idx + 1}")

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                output, target = self._step(batch)
                loss = self.criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs: int) -> None:
        """Train the model"""
        print(f"\nTraining {self.model_name} model for {epochs} epochs")
        print(f"Using device: {DEVICE}")

        early_stopping = EarlyStopping(
            patience=TrainingConfig.early_stopping_patience
        )

        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                self.save_checkpoint("best")

            # Print progress
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")

            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            # Save training history
            self.save_training_history()

            # Plot learning curves
            if (epoch + 1) % 5 == 0:
                self.plot_learning_curves()

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on test set"""
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                output, target = self._step(batch)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())

        # Calculate metrics
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        metrics = {
            'test_loss': total_loss / len(self.test_loader),
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

        self.save_metrics(metrics)
        return metrics

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

        path = Path(f'outputs/{self.model_name}/checkpoints/checkpoint_{name}.pt')
        torch.save(checkpoint, path)

    def save_training_history(self):
        """Save training history"""
        history = {
            'train_losses': [float(loss) for loss in self.train_losses],
            'val_losses': [float(loss) for loss in self.val_losses]
        }

        path = Path(f'outputs/{self.model_name}/results/training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)

    def save_metrics(self, metrics: Dict[str, float]) -> None:
        """Save evaluation metrics"""
        path = Path(f'outputs/{self.model_name}/results/metrics.json')
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=4)

    def plot_learning_curves(self) -> None:
        """Plot and save learning curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title(f'{self.model_name} Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        path = Path(f'outputs/{self.model_name}/plots/learning_curves.png')
        plt.savefig(path)
        plt.close()