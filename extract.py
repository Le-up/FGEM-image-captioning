import torch
import torch.nn as nn
from transformers import Swinv2Model


swin_path = "/root/autodl-tmp/IC/swin-base-transformer"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class GlobalFeatures(nn.Module):
    """
    提取全局特征。
    """
    def __init__(self):
        super(GlobalFeatures, self).__init__()
        self.swin_model = Swinv2Model.from_pretrained(swin_path).to(device)

        # 冻结所有层
        for param in self.swin_model.parameters():
            param.requires_grad = False

        # 解冻最后两层
        for layer in [self.swin_model.encoder.layers[-1], self.swin_model.layernorm]:
            for param in layer.parameters():
                param.requires_grad = True

        self.apply(initialize_weights)

    def forward(self, inputs):
        outputs = self.swin_model(inputs)
        out = outputs.last_hidden_state  # [batch_size, seq_len, 1024]
        return out


class FeatureExtractor(nn.Module):
    """
    特征提取器
    """
    def __init__(self, d_model, dropout):
        super(FeatureExtractor, self).__init__()
        self.global_features = GlobalFeatures()
        self.fc_global = nn.Sequential(
            nn.Linear(1024, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )
        self.fc_local = nn.Sequential(
            nn.Conv1d(262, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        self.apply(initialize_weights)

    def forward(self, images, local_features, local_masks):
        # 全局特征
        global_features = self.global_features(images)  # [batch_size, seq_len, 512]
        batch_size, seq_len, _ = global_features.shape
        global_features = global_features.contiguous().view(batch_size * seq_len, -1)
        global_features = self.fc_global(global_features)
        global_features = global_features.view(batch_size, seq_len, -1)

        # 局部特征
        local_features = local_features.permute(0, 2, 1)  # [batch_size, 262, seq_len]
        local_features = self.fc_local(local_features)  # [batch_size, 512, seq_len]
        local_features = local_features.permute(0, 2, 1)  # [batch_size, seq_len, 512]

        local_masks = local_masks.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

        return global_features, None, local_features, local_masks

