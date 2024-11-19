import torch.nn.functional as F
import math
import numpy as np
from torch import nn
import torch
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

def setup_checkpoint_dir():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = Path(f"checkpoints/encoder/{timestamp}")
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

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = torch.tensor(q.shape[-1], device=device)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        scaled = scaled + mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)

    return values, attention

class Multihead_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_layer = nn.Linear(d_model, 3 * d_model).to(device)
        self.qkv_linear = nn.Linear(d_model, d_model).to(device)

    def forward(self, x, mask=None):
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        batch_size, sequence_length, input_size = x.size()

        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.qkv_linear(values)
        return out, attention

def get_angles(pos, i, D):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
    return pos * angle_rates

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

def create_look_ahead_mask(size1, size2):
    mask = torch.ones((size1, size2), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        inputs = inputs.to(device)
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden).to(device)
        self.linear2 = nn.Linear(hidden, d_model).to(device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = x.to(device)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Encoder_Layer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
        super(Encoder_Layer, self).__init__()
        self.dropout_rate = dropout_rate

        self.ffn = PositionwiseFeedForward(d_model=D, hidden=hidden_mlp_dim)
        self.layernorm1 = LayerNormalization(parameters_shape=[D])
        self.layernorm2 = LayerNormalization(parameters_shape=[D])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.mha = Multihead_Attention(d_model=D, num_heads=H)

    def forward(self, x, look_ahead_mask=None):
        x = x.to(device)
        if look_ahead_mask is not None:
            look_ahead_mask = look_ahead_mask.to(device)

        attn, attn_weights = self.mha(x, mask=look_ahead_mask)
        attn = self.dropout1(attn)
        attn = self.layernorm1(attn + x)

        mlp_act = self.ffn(attn)
        output = self.layernorm2(mlp_act + attn)

        return output, attn_weights

class Encoder(nn.Module):
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate, n_future=48):
        super(Encoder, self).__init__()
        self.sqrt_D = torch.tensor(math.sqrt(D), device=device)
        self.num_layers = num_layers
        self.n_future = n_future

        self.input_projection = nn.Linear(inp_features, D).to(device)
        self.output_projection = nn.Linear(D, out_features).to(device)

        self.enc_layers = nn.ModuleList([
            Encoder_Layer(D, H, hidden_mlp_dim, dropout_rate).to(device)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.checkpoint_dir = setup_checkpoint_dir()

    def save(self, epoch=None):
        save_model(self, self.checkpoint_dir, "encoder", epoch)

    def load(self, path):
        return load_model(self, path)

    def forward(self, x, mask=None):
        x = x.to(device)
        if mask is not None:
            mask = mask.to(device)

        B, S, D = x.shape
        attention_weights = {}

        x = self.input_projection(x)
        x *= self.sqrt_D
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block = self.enc_layers[i](x=x, look_ahead_mask=mask)
            attention_weights[f'encoder_layer{i + 1}'] = block

        x = self.output_projection(x[:, -self.n_future:, :])
        return x, attention_weights

def move_to_gpu(model):
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"Model moved to: {next(model.parameters()).device}")
    return model