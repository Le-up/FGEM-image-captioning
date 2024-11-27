import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .attention import MultiHeadAttention, AggregationAttention
from .pos_embed import sinusoid_encoding_table, FeedForward
from .containers import Module, ModuleList  # 引入状态处理相关的模块

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_indexed_embeddings(embeddings_file_path):
#     indexed_embeddings = torch.load(embeddings_file_path, map_location='cpu')
#     return indexed_embeddings
#
# embeddings_file_path = '/root/autodl-tmp/IC/out/indexed_embeddings.pt'

class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=.1):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=True)
        self.pwff = FeedForward(d_model, d_ff, dropout)
        self.cross_att = MultiHeadAttention(d_model, n_heads, dropout, can_be_stateful=False)

        self.agg_layers = AggregationAttention(d_model, d_model // n_heads, n_heads)
        self.agg_views = AggregationAttention(d_model, d_model // n_heads, n_heads)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad

        n_views, (N, L, S, d) = len(enc_output), enc_output[0].shape
        enc_output = torch.stack(enc_output, dim=1)  # N x n_views x L x S x d
        enc_output = enc_output.reshape(L * n_views * N, S, d)
        mask_enc_att = torch.stack(mask_enc_att, dim=1)  # N x n_views x 1 x 1 x S
        mask_enc_att = mask_enc_att.unsqueeze(2).expand(N, n_views, L, 1, 1, S)
        mask_enc_att = mask_enc_att.reshape(L * n_views * N, 1, 1, S)

        self_att_exp = self_att.unsqueeze(1).expand(N, L * n_views, *self_att.shape[1:])
        self_att_exp = self_att_exp.reshape(L * n_views * N, *self_att.shape[1:])
        mask_pad_exp = mask_pad.unsqueeze(1).expand(N, L * n_views, *mask_pad.shape[1:])
        mask_pad_exp = mask_pad_exp.reshape(L * n_views * N, *mask_pad.shape[1:])
        enc_att = self.cross_att(
            self_att_exp, enc_output, enc_output, mask_enc_att
        ) * mask_pad_exp  # L * n_views * N x Sq x d

        enc_att = enc_att.reshape(n_views * N, L, *self_att.shape[1:])
        enc_att = torch.transpose(enc_att, 1, 2)  # n_views * N x Sq x L x d
        self_att_exp = self_att.unsqueeze(1).expand(N, n_views, *self_att.shape[1:])
        self_att_exp = self_att_exp.reshape(n_views * N, *self_att.shape[1:])
        mask_pad_exp = mask_pad.unsqueeze(1).expand(N, n_views, *mask_pad.shape[1:])
        mask_pad_exp = mask_pad_exp.reshape(n_views * N, *mask_pad.shape[1:])
        enc_att = self.agg_layers(self_att_exp, enc_att) * mask_pad_exp  # n_views * N x Sq x d

        enc_att = enc_att.reshape(N, n_views, *self_att.shape[1:])
        enc_att = torch.transpose(enc_att, 1, 2)  # N x Sq x n_views x d
        enc_att = self.agg_views(self_att, enc_att) * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class CaptionGenerator(Module):
    def __init__(self, vocab_size, max_len, pad_idx, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, padding_idx=pad_idx), freeze=True)
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.register_state('running_mask_x', torch.zeros((1, 1, 0), dtype=torch.bool, device=device))
        self.register_state('running_seq', torch.zeros((1,), dtype=torch.long, device=device))
        # self.w_fc = nn.Linear(300, d_model)
        self.apply(self._init_weights)

        # self.indexed_embeddings = load_indexed_embeddings(embeddings_file_path)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def get_seq_inputs(self, input):
        b_s, seq_len = input.shape[:2]
        # # 预训练词嵌入
        # embedding_tensors = []
        # for sequence in input:
        #     embedding_sequence = []
        #     for word_index in sequence:
        #         # 转换 word_index 为整数
        #         word_index_value = word_index.item() if isinstance(word_index, torch.Tensor) else word_index
        #         embedding = self.indexed_embeddings[word_index_value]
        #         embedding_sequence.append(embedding)
        #
        #     embedding_tensor = torch.stack(embedding_sequence)
        #     embedding_tensors.append(embedding_tensor)
        # batch_embeddings = torch.stack(embedding_tensors).to(device)
        # word_emb = self.w_fc(batch_embeddings)

        mask_pad = (input != self.pad_idx).unsqueeze(-1).float().to(device)
        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)
        mask_x = mask_x.unsqueeze(0).unsqueeze(0)
        mask_x = mask_x + (input == self.pad_idx).unsqueeze(1).unsqueeze(1).bool()

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)

        if self._is_stateful:
            self.running_mask_x = torch.cat([self.running_mask_x, mask_x], -1)
            mask_x = self.running_mask_x
            self.running_seq.add_(1)
            seq = self.running_seq

        # out = word_emb + self.pos_emb(seq)
        out = self.word_emb(input) + self.pos_emb(seq)
        return out, mask_x, mask_pad

    def forward(self, input, encoder_output, encoder_mask):
        out, mask_x, mask_pad = self.get_seq_inputs(input)

        for layer in self.layers:
            out = layer(out, encoder_output, mask_pad, mask_x, encoder_mask)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)