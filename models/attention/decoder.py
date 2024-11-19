import torch
import torch.nn.functional as F
import math
from torch import nn
from config.config import DEVICE as device

class MultiheadAttention(nn.Module):
    def __init__(self, D, H):
        super(MultiheadAttention, self).__init__()
        self.H = H
        self.D = D

        # Initialize layers and move to GPU
        self.wq = nn.Linear(D, D*H).to(device)
        self.wk = nn.Linear(D, D*H).to(device)
        self.wv = nn.Linear(D, D*H).to(device)
        self.dense = nn.Linear(D*H, D).to(device)

    def concat_heads(self, x):
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()
        x = x.reshape((B, S, H*D))
        return x

    def split_heads(self, x):
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)
        x = x.permute((0, 2, 1, 3))
        return x

    def forward(self, x, mask=None):
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.D)

        if mask is not None:
            attention_scores += (mask * -1e9)

        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        scaled_attention = torch.matmul(attention_weights, v)
        concat_attention = self.concat_heads(scaled_attention)
        output = self.dense(concat_attention)

        return output, attention_weights

class DecoderLayer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.dropout_rate = dropout_rate

        # Initialize all layers and move to GPU
        self.mlp_hidden = nn.Linear(D, hidden_mlp_dim).to(device)
        self.mlp_out = nn.Linear(hidden_mlp_dim, D).to(device)
        self.layernorm1 = nn.LayerNorm(D, eps=1e-9).to(device)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9).to(device)
        self.layernorm3 = nn.LayerNorm(D, eps=1e-9).to(device)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.masked_mha = MultiheadAttention(D, H)

    def forward(self, x, look_ahead_mask):
        x = x.to(device)
        if look_ahead_mask is not None:
            look_ahead_mask = look_ahead_mask.to(device)

        masked_attn, masked_attn_weights = self.masked_mha(x, mask=look_ahead_mask)
        masked_attn = self.dropout1(masked_attn)
        masked_attn = self.layernorm1(masked_attn + x)

        mlp_act = torch.relu(self.mlp_hidden(masked_attn))
        mlp_act = self.mlp_out(mlp_act)
        mlp_act = self.dropout3(mlp_act)

        output = self.layernorm3(mlp_act + masked_attn)

        return output, masked_attn_weights

class Decoder(nn.Module):
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate, n_future=48):
        super(Decoder, self).__init__()
        self.sqrt_D = torch.tensor(math.sqrt(D), device=device)
        self.num_layers = num_layers
        self.n_future = n_future

        # Initialize projections and move to GPU
        self.input_projection = nn.Linear(inp_features, D).to(device)
        self.output_projection = nn.Linear(D, out_features).to(device)

        # Initialize decoder layers
        self.dec_layers = nn.ModuleList([
            DecoderLayer(D, H, hidden_mlp_dim, dropout_rate).to(device)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)

        # Register positional encoding buffer
        pos_encoding = self._get_positional_encoding(D)
        self.register_buffer('pos_encoding', pos_encoding)

    def _get_positional_encoding(self, D, max_position=168):
        position = torch.arange(max_position, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device) * (-math.log(10000.0) / D))

        pos_encoding = torch.zeros(1, max_position, D, device=device)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)

        return pos_encoding

    def create_mask(self, size):
        mask = torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x, mask=None):
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        B, S, _ = x.shape
        attention_weights = {}

        x = self.input_projection(x)
        x *= self.sqrt_D
        x += self.pos_encoding[:, :S, :]
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block = self.dec_layers[i](x=x, look_ahead_mask=mask)
            attention_weights[f'decoder_layer{i + 1}'] = block

        x = self.output_projection(x)
        return x[:, -self.n_future:, :], attention_weights

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'num_layers': self.num_layers,
                'D': self.sqrt_D.item() ** 2,
                'H': self.dec_layers[0].masked_mha.H,
                'hidden_mlp_dim': self.dec_layers[0].mlp_hidden.out_features,
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