import torch
import torch.nn.functional as F
import math
from torch import nn
from config.config import DEVICE as device

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Initialize layers and move to GPU
        self.qkv_layer = nn.Linear(d_model, 3 * d_model).to(device)
        self.linear_out = nn.Linear(d_model, d_model).to(device)

    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.shape

        # Move input to GPU and process
        x = x.to(device)
        qkv = self.qkv_layer(x)

        # Reshape and split into Q, K, V
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim*3]
        q, k, v = qkv.chunk(3, dim=-1)  # 3 x [batch, heads, seq_len, head_dim]

        # Scaled dot-product attention
        d_k = torch.tensor(q.shape[-1], device=device)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.to(device)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Reshape and project output
        attention_output = attention_output.permute(0, 2, 1, 3)  # [batch, seq_len, heads, head_dim]
        attention_output = attention_output.reshape(batch_size, sequence_length, self.d_model)
        output = self.linear_out(attention_output)

        return output, attention_weights

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        x = x.to(device)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff).to(device)
        self.linear2 = nn.Linear(d_ff, d_model).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.to(device)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiheadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self attention
        attention_output, attention_weights = self.self_attention(x, mask)
        attention_output = self.dropout(attention_output)
        x = self.norm1(x + attention_output)

        # Feed forward
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)

        return x, attention_weights

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, inp_features,
                 out_features, dropout_rate, n_future=48):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_future = n_future

        # Input and output projections
        self.input_projection = nn.Linear(inp_features, d_model).to(device)
        self.output_projection = nn.Linear(d_model, out_features).to(device)

        # Position encoding
        self.pos_encoding = self._create_positional_encoding()

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate).to(device)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)

    def _create_positional_encoding(self, max_seq_len=168):
        position = torch.arange(max_seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) *
                             (-math.log(10000.0) / self.d_model))

        pos_encoding = torch.zeros(1, max_seq_len, self.d_model, device=device)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def forward(self, x, mask=None):
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        B, S, _ = x.shape
        attention_weights = {}

        # Input projection and positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :S, :]
        x = self.dropout(x)

        # Encoder layers
        for i, encoder_layer in enumerate(self.encoder_layers):
            x, attention = encoder_layer(x, mask)
            attention_weights[f'encoder_layer_{i+1}'] = attention

        # Output projection and selection of future timesteps
        output = self.output_projection(x[:, -self.n_future:, :])

        return output, attention_weights

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'num_layers': self.num_layers,
                'd_model': self.d_model,
                'num_heads': self.encoder_layers[0].self_attention.num_heads,
                'd_ff': self.encoder_layers[0].feed_forward.linear1.out_features,
                'inp_features': self.input_projection.in_features,
                'out_features': self.output_projection.out_features,
                'dropout_rate': self.dropout.p,
                'n_future': self.n_future
            }
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return model

    def create_mask(self, size):
        """Create look-ahead mask for self-attention"""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask == 0