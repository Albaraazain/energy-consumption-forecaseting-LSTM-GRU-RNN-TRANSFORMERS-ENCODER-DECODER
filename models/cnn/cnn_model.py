import torch
from torch import nn
import math
from config.config import DEVICE as device

class CNNForecastNet(nn.Module):
    def __init__(
            self,
            hidden_size: int = 64,
            kernel_size: int = 3,
            padding: int = 1,
            drop_rate: float = 0.1,
            input_channels: int = 1,
            target_sequence_length: int = 48,
            num_cnn_layers: int = 3
    ):
        super(CNNForecastNet, self).__init__()

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.drop_rate = drop_rate
        self.input_channels = input_channels
        self.target_sequence_length = target_sequence_length
        self.num_cnn_layers = num_cnn_layers

        # CNN layers
        self.cnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channels if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=padding
                ).to(device),
                nn.BatchNorm1d(hidden_size).to(device),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_rate)
            ).to(device)
            for i in range(num_cnn_layers)
        ])

        # Max pooling layer
        self.max_pool = nn.MaxPool1d(2)

        # Calculate the size after CNN layers and pooling
        self._calculate_flatten_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, hidden_size * 2).to(device)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size).to(device)
        self.fc3 = nn.Linear(hidden_size, target_sequence_length).to(device)

        # Dropout for FC layers
        self.dropout = nn.Dropout(drop_rate)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size).to(device)

        # Initialize weights
        self._initialize_weights()

    def _calculate_flatten_size(self):
        """Calculate the size of the flattened layer after CNNs and pooling"""
        # Create dummy input
        x = torch.randn(1, self.input_channels, 168).to(device)  # 168 is input sequence length

        # Pass through CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
            x = self.max_pool(x)

        # Calculate flatten size
        self.flatten_size = x.shape[1] * x.shape[2]

    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to device and reshape for CNN
        x = x.to(device)
        x = x.permute(0, 2, 1)  # [batch, channels, sequence_length]

        # Pass through CNN layers with residual connections
        cnn_output = x
        for cnn_layer in self.cnn_layers:
            layer_output = cnn_layer(cnn_output)
            cnn_output = self.max_pool(layer_output)

        # Flatten
        flattened = cnn_output.reshape(cnn_output.size(0), -1)

        # Fully connected layers with skip connections
        fc1_output = self.dropout(torch.relu(self.fc1(flattened)))
        fc2_output = self.dropout(torch.relu(self.fc2(fc1_output)))
        fc2_output = self.layer_norm(fc2_output)
        output = self.fc3(fc2_output)

        # Reshape output to [batch, target_sequence_length, 1]
        output = output.unsqueeze(-1)

        return output

    def save(self, path: str):
        """Save model state and configuration"""
        config = {
            'hidden_size': self.hidden_size,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'drop_rate': self.drop_rate,
            'input_channels': self.input_channels,
            'target_sequence_length': self.target_sequence_length,
            'num_cnn_layers': self.num_cnn_layers
        }

        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load model from path"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return model

    def count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage(self) -> float:
        """Get model memory usage in MB"""
        mem_params = sum([param.nelement()*param.element_size() for param in self.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in self.buffers()])
        return (mem_params + mem_bufs) / 1024**2  # Convert to MB