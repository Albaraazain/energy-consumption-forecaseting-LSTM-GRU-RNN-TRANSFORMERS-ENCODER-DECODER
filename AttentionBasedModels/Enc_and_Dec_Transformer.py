import torch
import torch.nn.functional as F
import math
import numpy as np
import torch
from torch import nn, Tensor

# Reuse the get_device function or import it
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

device = get_device()

class PositionalEncoder(nn.Module):
    def __init__(self, dropout: float = 0.1,
                 max_seq_len: int = 168, d_model: int = 32):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = True

        # Move position encodings to GPU
        position = torch.arange(max_seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) *
                             (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_seq_len, d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(device)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self,
                 input_size: int,
                 dec_seq_len: int,
                 out_seq_len: int=48,
                 dim_val: int=32,
                 n_encoder_layers: int=1,
                 n_decoder_layers: int=1,
                 n_heads: int=4,
                 dropout_encoder: float=0.2,
                 dropout_decoder: float=0.2,
                 dropout_pos_enc: float=0.1,
                 dim_feedforward_encoder: int=100,
                 dim_feedforward_decoder: int=100,
                 num_predicted_features: int=1
                 ):
        super().__init__()

        self.dec_seq_len = dec_seq_len

        # Move all layers to GPU
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        ).to(device)

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
        ).to(device)

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        ).to(device)

        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
        ).to(device)

        # Create encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=True
        ).to(device)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        ).to(device)

        # Create decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=True
        ).to(device)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        ).to(device)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None,
                tgt_mask: Tensor=None) -> Tensor:

        # Move input tensors to GPU
        src = src.to(device)
        tgt = tgt.to(device)
        if src_mask is not None:
            src_mask = src_mask.to(device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(device)

        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src=src)

        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )
        decoder_output = self.linear_mapping(decoder_output)

        return decoder_output

def move_to_gpu(model):
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"Model moved to: {next(model.parameters()).device}")
    return model

# Optional testing code
"""
if __name__ == "__main__":
    model = Transformer(
        input_size=1,
        dec_seq_len=48
    )
    model = move_to_gpu(model)
"""