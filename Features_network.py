import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .pos_embed import FeedForward


def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class TransformerLayer(nn.Module):
    """
    实现单层Transformer，包括多头注意力和前馈神经网络。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, n_memories=None):
        super().__init__()

        self.mhatt = MultiHeadAttention(d_model, n_heads, dropout, n_memories=n_memories)
        self.pwff = FeedForward(d_model, d_ff, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.apply(initialize_weights)

    def forward(self, q, k, v, mask=None):
        # assert not torch.isnan(q).any(), "NaN detected in q"
        # assert not torch.isnan(k).any(), "NaN detected in k"
        # assert not torch.isnan(v).any(), "NaN detected in v"

        att_output = self.mhatt(q, k, v, mask)
        # assert not torch.isnan(att_output).any(), "NaN detected in att_output"
        out1 = self.layer_norm1(q + self.dropout(att_output))
        ff_output = self.pwff(out1)
        # assert not torch.isnan(ff_output).any(), "NaN detected in ff_output"
        out2 = self.layer_norm2(out1 + self.dropout(ff_output))
        return out2


class MultiLevelEncoder(nn.Module):
    """
    多层Transformer网络，用于处理多视角特征。
    """

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout, n_memories=None):
        super(MultiLevelEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout, n_memories=n_memories)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.apply(initialize_weights)

    def forward(self, views):
        n_views, (N, S, d) = len(views), views[0].shape
        views = torch.stack(views).reshape(N * n_views, S, d)
        # assert not torch.isnan(views).any(), "NaN detected in views"
        mask = (torch.sum(torch.abs(views), -1) == 0)
        mask = mask.unsqueeze(1).unsqueeze(1)  # (N*n_views, 1, 1, S)

        outs = []
        out = views
        # assert not torch.isnan(out).any(), "NaN detected in out"
        for i, layer in enumerate(self.layers):
            out = layer(out, out, out, mask)
            outs.append(out)
        outs = torch.stack(outs, dim=1)  # N*n_views x l x S x d

        outs = outs.reshape(n_views, N, len(self.layers), S, d)
        masks = mask.reshape(n_views, N, 1, 1, S)
        outs = [o for o in outs]
        masks = [m for m in masks]
        return outs, masks


class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, n_layers, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(n_layers, **kwargs)

    def forward(self, input):
        return super(MemoryAugmentedEncoder, self).forward(input)
