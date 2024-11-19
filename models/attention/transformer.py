import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Optional
from config.config import DEVICE as device, DEVICE


def scaled_dot_product_attention(q, k, v, mask=None):
    """Scaled dot product attention mechanism."""
    d_k = torch.tensor(q.shape[-1])
    scaled = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(d_k)

    if mask is not None:
        scaled = scaled + mask

    attention = torch.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_length, d_model = x.size()

        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product_attention(q, k, v, mask)
        values = values.reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        out = self.linear(values)

        return out, attention

class EncoderLayer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.mha = MultiheadAttention(d_model=D, num_heads=H)
        self.ffn = nn.Sequential(
            nn.Linear(D, hidden_mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_mlp_dim, D)
        )

        self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output, attention_weights = self.mha(x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attention_weights

class Encoder(nn.Module):
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.D = D

        self.input_projection = nn.Linear(inp_features, D)
        self.pos_encoding = nn.Parameter(torch.randn(1, 168, D))  # Fixed length for position encoding
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(D, H, hidden_mlp_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(D, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = x * torch.sqrt(torch.tensor(self.D, dtype=torch.float32))
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)

        attention_weights = {}
        for i, layer in enumerate(self.encoder_layers):
            x, attention = layer(x, mask)
            attention_weights[f'encoder_layer{i+1}'] = attention

        x = self.output_projection(x[:, -48:, :])
        return x, attention_weights

class DecoderLayer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.mha = MultiheadAttention(D, H)
        self.ffn = nn.Sequential(
            nn.Linear(D, hidden_mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_mlp_dim, D)
        )

        self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        attn_output, attention_weights = self.mha(x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attention_weights

class Decoder(nn.Module):
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.D = D

        self.input_projection = nn.Linear(inp_features, D)
        self.pos_encoding = nn.Parameter(torch.randn(1, 168, D))
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(D, H, hidden_mlp_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(D, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = x * torch.sqrt(torch.tensor(self.D, dtype=torch.float32))
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.dropout(x)

        attention_weights = {}
        for i, layer in enumerate(self.decoder_layers):
            x, attention = layer(x, mask)
            attention_weights[f'decoder_layer{i+1}'] = attention

        x = self.output_projection(x[:, -48:, :])
        return x, attention_weights

class Transformer(nn.Module):
    def __init__(self, input_size, dec_seq_len, num_predicted_features,
                 d_model=32, nhead=4, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.dec_seq_len = dec_seq_len

        # Input and output projections
        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, num_predicted_features)

        # Create separate positional encodings for encoder and decoder
        self.encoder_pos = self._create_positional_encoding(168, d_model)  # For input sequence
        self.decoder_pos = self._create_positional_encoding(dec_seq_len, d_model)  # For target sequence

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=None
        )

        # Move model to device
        self.to(DEVICE)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding matrix"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pos_encoding = torch.zeros(1, max_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)

        return pos_encoding.to(DEVICE)

    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate a square mask for the sequence."""
        mask = torch.full((size, size), float('-inf'), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        mask = torch.where(mask == float('-inf'),
                           mask,
                           torch.zeros_like(mask))
        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project inputs
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)

        # Add positional encoding
        src = src + self.encoder_pos  # For input sequence
        tgt = tgt + self.decoder_pos  # For target sequence

        # Transformer operations
        memory = self.encoder(src=src, mask=src_mask)
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

        # Project output
        output = self.output_projection(output)

        return output