"""
Swadesh v4 Model
Standard Transformer Encoder (MLM)
d=64, 4layers4heads, max_len=48
v4 change：max_len 36→48（S2.4max sequence=48，V4degree-base desc from13→26）
"""
import torch
import torch.nn as nn
import math


class SwaModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len, dropout=0.1):
        super().__init__()
        self.tokens_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.drop(
            self.tokens_embed(x) * math.sqrt(self.d_model) + self.pos_embed(pos)
        )
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h)


def make_model(vocab_size, d=64, n_layers=4, n_heads=4, max_len=48, dropout=0.1):
    return SwaModel(vocab_size, d, n_layers, n_heads, max_len, dropout)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
