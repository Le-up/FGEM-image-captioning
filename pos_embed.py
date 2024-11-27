import math
import torch
from torch import nn


def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))

    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out


def sinusoid_encoding_table(max_len, d_model, padding_idx=None):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    if padding_idx is not None:
        out[padding_idx] = 0

    # use the magnitude of kaiming init with fan_in
    fan = nn.init._calculate_correct_fan(out, "fan_in")
    gain = nn.init.calculate_gain('leaky_relu', 0)
    std = gain / math.sqrt(fan)
    a = math.sqrt(3.0) * std
    out *= a

    return out

class FeedForward(nn.Module):
    """
    前馈神经网络。
    """
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()  # 使用 GELU 激活函数

        self._init_weights()

    def _init_weights(self):
        """
        初始化权重。
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, input):
        """
        前向传播。
        :param input: 输入张量
        :return: 输出张量
        """
        out = self.layer_norm(input)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = input + out
        return out

