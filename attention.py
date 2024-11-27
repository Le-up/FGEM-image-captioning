import numpy as np
import torch
from torch import nn
from .containers import Module  # 使用绝对导入路径
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MemoryAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, n_memories=0):
        super().__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.n_memories = n_memories
        if self.n_memories > 1:
            self.m_k = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
            self.m_v = nn.Parameter(torch.FloatTensor(1, n_memories, d_model))
            nn.init.normal_(self.m_k, mean=0, std=1 / self.m_k.shape[-1])
            nn.init.normal_(self.m_v, mean=0, std=1 / self.m_v.shape[-1])

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.dropout = nn.Dropout(p=dropout)

        self.apply(self.init_params)

    def init_params(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, q, k, v, attention_mask=None):
        b_s, nq = q.shape[:2]
        nk = k.shape[1]
        eps = 1e-6

        if self.n_memories > 1:
            m_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.n_memories, self.d_model)
            m_v = np.sqrt(self.n_memories) * self.m_v.expand(b_s, self.n_memories, self.d_model)

            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)
            k = torch.cat([self.fc_k(k), m_k], 1).view(b_s, nk + self.n_memories, self.n_heads, self.d_k).permute(0, 2, 3, 1)
            v = torch.cat([self.fc_v(v), m_v], 1).view(b_s, nk + self.n_memories, self.n_heads, self.d_v).permute(0, 2, 1, 3)

            # assert not torch.isnan(q).any(), "NaN detected in q"
            # assert not torch.isnan(k).any(), "NaN detected in k"
            # assert not torch.isnan(v).any(), "NaN detected in v"

            scores = torch.matmul(q, k) / (np.sqrt(self.d_k) + eps)
            if attention_mask is not None:
                scores[:, :, :, :nk] = scores[:, :, :, :nk].masked_fill(attention_mask.bool(), -1e9)
        else:
            q = rearrange(self.fc_q(q), 'b nq (head d) -> b head nq d', head=self.n_heads)
            k = rearrange(self.fc_k(k), 'b nk (head d) -> b head d nk', head=self.n_heads)
            v = rearrange(self.fc_v(v), 'b nv (head d) -> b head nv d', head=self.n_heads)

            # assert not torch.isnan(q).any(), "NaN detected in q"
            # assert not torch.isnan(k).any(), "NaN detected in k"
            # assert not torch.isnan(v).any(), "NaN detected in v"

            scores = torch.matmul(q, k) / (np.sqrt(self.d_k) + eps)
            # assert not torch.isnan(scores).any(), "NaN detected in attention scores"

            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask.bool(), -1e9)

        p_attn = self.dropout(torch.softmax(scores, -1))
        # assert not torch.isnan(p_attn).any(), "NaN detected in attention probabilities"

        out = rearrange(torch.matmul(p_attn, v), 'b h nq dv -> b nq (h dv)')
        # assert not torch.isnan(out).any(), "NaN detected in attention output before fc_o"
        out = self.fc_o(out)
        # assert not torch.isnan(out).any(), "NaN detected in attention output after fc_o"
        return out

class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, dropout, can_be_stateful=False, n_memories=None):
        super().__init__()

        if n_memories is not None:
            self.attention = MemoryAttention(d_model, n_heads, dropout, n_memories)
        else:
            self.attention = MemoryAttention(d_model, n_heads, dropout)

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, self.d_model), device=device))
            self.register_state('running_values', torch.zeros((0, self.d_model), device=device))

    def forward(self, queries, keys, values, attention_mask=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        out = self.attention(queries, keys, values, attention_mask)
        # assert not torch.isnan(out).any(), "NaN detected in attention output"

        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        # assert not torch.isnan(out).any(), "NaN detected after layer normalization"
        return out

class AggregationAttentionLayer(nn.Module):
    def __init__(self, d_model, d_inter, h, dropout=.1):
        super(AggregationAttentionLayer, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_inter)
        self.fc_k = nn.Linear(d_model, h * d_inter)
        self.fc_v = nn.Linear(d_model, h * d_inter)
        self.fc_o = nn.Linear(h * d_inter, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_inter = d_inter
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, targets):
        b_s, nq = queries.shape[:2]
        nk = targets.shape[2]
        q = self.fc_q(queries).view(b_s, nq, 1, self.h, self.d_inter).permute(0, 3, 1, 2, 4)  # (b_s, h, nq, 1, d_inter)
        k = self.fc_k(targets).view(b_s, nq, nk, self.h, self.d_inter).permute(0, 3, 1, 2, 4)  # (b_s, h, nq, nk, d_inter)
        v = self.fc_v(targets).view(b_s, nq, nk, self.h, self.d_inter).permute(0, 3, 1, 2 ,4)  # (b_s, h, nq, nk, d_inter)

        eps = 1e-6  # 小常数避免除以零
        att = torch.sum(q * k / (np.sqrt(self.d_inter) + eps), dim=-1, keepdim=True)  # (b_s, h, nq, nk, 1)
        att = self.dropout(torch.softmax(att, dim=3))
        out = torch.sum(att * v, dim=3)  # (b_s, h, nq, d_inter)
        out = out.permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_inter)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class AggregationAttention(Module):
    def __init__(self, d_model, d_inter, h, dropout=.1, identity_map_reordering=True):
        super(AggregationAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = AggregationAttentionLayer(
            d_model=d_model, d_inter=d_inter, h=h, dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, queries, targets):
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            t_norm = self.layer_norm(targets)
            out = self.attention(q_norm, t_norm)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, targets)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out
