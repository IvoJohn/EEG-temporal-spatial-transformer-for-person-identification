import torch
from torch import nn
import math


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        L_modules: int,
        T_dim: int,
        C_dim: int = 64,
        n_users: int = 109,
        heads: int = 8,
        dropout: float = 0.5,
    ):
        super(TemporalTransformer, self).__init__()

        """
        L_modules - number of transformer modules in temporal and in spatial encoders
        T_dim - number of timepoints
        C_dim - number of channels
        n_users - numbers of users so number of output neurons
        heads - number of heads for multihead attention
        dropout - dropout in transformer modules and between fc layers
        """

        self.temporal_encoder = nn.ModuleList(
            [TransformerBlock(C_dim, heads, dropout) for _ in range(L_modules)]
        )
        self.pos_embed = PositionalEncoding(T_dim)
        self.spatial_encoder = nn.ModuleList(
            [TransformerBlock(T_dim, heads, dropout) for _ in range(L_modules)]
        )

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(C_dim * T_dim, T_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(T_dim, n_users)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for encoder in self.temporal_encoder:
            x = encoder(x, x, x)

        x = torch.transpose(x, 1, 2)
        x = self.pos_embed(x)

        for encoder in self.spatial_encoder:
            x = encoder(x, x, x)

        x = self.dropout(self.fc1(self.flat(x)))
        x = self.fc2(x)
        out = self.softmax(x)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, T_dim, heads, dropout, forward_expansion=3):
        super(TransformerBlock, self).__init__()

        """
        T_dim - number of timepoints
        heads - number of heads for multihead attention
        dropout - dropout in transformer modules and between fc layers
        """

        self.attention = nn.MultiheadAttention(T_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(T_dim)
        self.norm2 = nn.LayerNorm(T_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(T_dim, forward_expansion * T_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * T_dim, T_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention, _ = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]

        return self.dropout(x)
