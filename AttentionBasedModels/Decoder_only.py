import torch
import torch.nn.functional as F
import math
import numpy as np
from torch import nn
from pathlib import Path
import time

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

device = get_device()

def get_angles(pos, i, D):
    """Calculate angles for positional encoding"""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
    return pos * angle_rates

def setup_checkpoint_dir():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = Path(f"checkpoints/decoder/{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory created: {checkpoint_dir}")
    return checkpoint_dir

def save_model(model, checkpoint_dir, name, epoch=None):
    if epoch is not None:
        path = checkpoint_dir / f"{name}_epoch_{epoch}.pt"
    else:
        path = checkpoint_dir / f"{name}_final.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'device': str(next(model.parameters()).device)
    }, path)
    print(f"Model saved to {path}")

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {path}")
    return model, checkpoint['epoch']

class Multihead_Attention(nn.Module):
    def __init__(self, D, H):
        super(Multihead_Attention, self).__init__()
        self.H = H
        self.D = D

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

def positional_encoding(D, position=168, dim=3):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(D)[np.newaxis, :],
                            D)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    if dim == 3:
        pos_encoding = angle_rads[np.newaxis, ...]
    elif dim == 4:
        pos_encoding = angle_rads[np.newaxis, np.newaxis, ...]

    return torch.tensor(pos_encoding, device=device)

def create_look_ahead_mask(size, device=device):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

class Decoder_Layer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
        super(Decoder_Layer, self).__init__()
        self.dropout_rate = dropout_rate

        self.mlp_hidden = nn.Linear(D, hidden_mlp_dim).to(device)
        self.mlp_out = nn.Linear(hidden_mlp_dim, D).to(device)
        self.layernorm1 = nn.LayerNorm(D, eps=1e-9).to(device)
        self.layernorm2 = nn.LayerNorm(D, eps=1e-9).to(device)
        self.layernorm3 = nn.LayerNorm(D, eps=1e-9).to(device)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.masked_mha = Multihead_Attention(D, H)

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

        self.input_projection = nn.Linear(inp_features, D).to(device)
        self.output_projection = nn.Linear(D, out_features).to(device)
        self.pos_encoding = positional_encoding(D)

        self.dec_layers = nn.ModuleList([
            Decoder_Layer(D, H, hidden_mlp_dim, dropout_rate).to(device)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.checkpoint_dir = setup_checkpoint_dir()

    def save(self, epoch=None):
        save_model(self, self.checkpoint_dir, "decoder", epoch)

    def load(self, path):
        return load_model(self, path)

    def forward(self, x, mask):
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        B, S, D = x.shape
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

def move_to_gpu(model):
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"Model moved to: {next(model.parameters()).device}")
    return model