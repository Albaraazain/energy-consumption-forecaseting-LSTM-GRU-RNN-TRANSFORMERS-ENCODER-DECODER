import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple
from config.config import DataConfig, DEVICE

class CustomDataset(Dataset):
    """Custom dataset for time series data"""
    def __init__(self, sequence: np.ndarray, input_sequence_length: int, target_sequence_length: int):
        """
        Args:
            sequence: Input sequence of shape [sequence_length, features]
            input_sequence_length: Length of input sequence
            target_sequence_length: Length of target sequence
        """
        # Convert to float32 and create tensor
        self.sequence = torch.tensor(sequence.astype(np.float32), dtype=torch.float32)
        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.window_size = input_sequence_length + target_sequence_length
        self.valid_length = len(self.sequence) - self.window_size + 1

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            src: Source sequence for encoder [input_sequence_length, features]
            trg: Target sequence for decoder input [target_sequence_length, features]
            trg_y: Target sequence for loss calculation [target_sequence_length, features]
        """
        if idx >= self.__len__():
            raise IndexError("Index out of bounds")

        # Get source sequence (input to encoder)
        src = self.sequence[idx:idx + self.input_sequence_length]

        # Get target sequence for decoder input (includes last input value)
        trg = self.sequence[idx + self.input_sequence_length - 1:
                            idx + self.window_size - 1]

        # Get target values for loss calculation
        trg_y = self.sequence[idx + self.input_sequence_length:
                              idx + self.input_sequence_length + self.target_sequence_length]

        # Add feature dimension if needed
        if src.dim() == 1:
            src = src.unsqueeze(-1)
            trg = trg.unsqueeze(-1)
            trg_y = trg_y.unsqueeze(-1)

        return src, trg, trg_y

def process_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process the time series data."""
    print(f"Processing data from {filepath}")

    # Read data
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Ensure values are numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Normalize the data
    data_mean = df.mean(axis=0)
    data_std = df.std(axis=0)
    df_normalized = (df - data_mean) / data_std

    # Convert to numpy array
    data = df_normalized.values

    # Calculate split indices
    train_size = int(len(data) * DataConfig.TRAIN_SPLIT)
    val_size = int(len(data) * DataConfig.VAL_SPLIT)

    # Split data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data

def create_dataloaders(
        data_path: str,
        batch_size: int = DataConfig.BATCH_SIZE,
        input_sequence_length: int = DataConfig.INPUT_SEQUENCE_LENGTH,
        target_sequence_length: int = DataConfig.TARGET_SEQUENCE_LENGTH,
        num_workers: int = DataConfig.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test dataloaders."""

    # Process data
    train_data, val_data, test_data = process_data(data_path)

    # Create datasets
    train_dataset = CustomDataset(train_data, input_sequence_length, target_sequence_length)
    val_dataset = CustomDataset(val_data, input_sequence_length, target_sequence_length)
    test_dataset = CustomDataset(test_data, input_sequence_length, target_sequence_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=DataConfig.PIN_MEMORY,
        drop_last=DataConfig.DROP_LAST
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DataConfig.PIN_MEMORY,
        drop_last=DataConfig.DROP_LAST
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DataConfig.PIN_MEMORY,
        drop_last=DataConfig.DROP_LAST
    )

    return train_loader, val_loader, test_loader

def denormalize_data(normalized_data: np.ndarray, data_path: str) -> np.ndarray:
    """Denormalize the data using statistics from original dataset."""
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    data_mean = df.mean(axis=0).values
    data_std = df.std(axis=0).values

    return normalized_data * data_std + data_mean

def get_data_stats(data_path: str) -> Tuple[float, float]:
    """Get mean and standard deviation of the dataset."""
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    return df.mean(axis=0).values[0], df.std(axis=0).values[0]