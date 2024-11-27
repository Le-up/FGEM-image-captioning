import torch
from torch import nn
from .pos_embed import sinusoid_encoding_table

class ViewGenerator(nn.Module):
    def __init__(self, d_obj, d_vis, d_txt, d_out, drop_rate=0.1, topk=8):
        super().__init__()
        self.d_out = d_out

        # for objects O
        self.obj_mlp = nn.Sequential(
            nn.LayerNorm(d_obj),
            nn.Linear(d_obj, d_out),
            nn.Dropout(drop_rate),
            nn.LayerNorm(d_out)
        )

        # for vis_ctx
        self.vis_mlp = nn.Sequential(
            nn.LayerNorm(d_vis),
            nn.Linear(d_vis, d_out),
            nn.Dropout(drop_rate),
            nn.LayerNorm(d_out)
        )

        # for txt_ctx (whole)
        self.txt_mlp_whole = nn.Sequential(
            nn.LayerNorm(d_txt),
            nn.Linear(d_txt, d_out),
            nn.Dropout(drop_rate),
            nn.LayerNorm(d_out)
        )

        self.txt_pos_whole = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(1, d_out), freeze=True
        )

        self.txt_rank_whole = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(topk * 8, d_out), freeze=True
        )

        self.n_views = 3

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, obj, vis_ctx, txt_ctx):
        views = []
        eps = 1e-6

        # object
        obj_embed = self.obj_mlp(obj)
        obj_mask = (torch.sum(torch.abs(obj), dim=-1) == 0)
        obj_embed[obj_mask] = 0.
        obj_cls = torch.sum(obj_embed, dim=1, keepdim=True)  # N x 1 x d

        obj_norm = torch.sum(~obj_mask, dim=-1, keepdim=True).unsqueeze(-1).float()
        obj_norm = obj_norm + eps  # 避免除以零
        obj_cls = obj_cls / obj_norm.detach()
        obj_embed = torch.cat([obj_cls, obj_embed], dim=1)
        views.append(obj_embed)

        # vis_ctx
        vis = vis_ctx["grid"]
        vis_embed = self.vis_mlp(vis)
        vis_cls = torch.mean(vis_embed, dim=1, keepdim=True)
        vis_embed = torch.cat([vis_cls, vis_embed], dim=1)
        views.append(vis_embed)

        # txt_ctx (whole)
        txt_k = self.txt_mlp_whole(txt_ctx["whole"]["embed"])
        pos_k = self.txt_pos_whole(txt_ctx["whole"]["pos"])
        rank_k = self.txt_rank_whole(txt_ctx["whole"]["rank"])

        embed_k = txt_k + pos_k + rank_k
        txt_cls = torch.mean(embed_k, dim=1, keepdim=True)
        embed_k = torch.cat([txt_cls, embed_k], dim=1)
        views.append(embed_k)

        # pad sequence
        max_len = max([v.shape[1] for v in views])
        N, _, d = views[0].shape
        for i, v in enumerate(views):
            diff_len = max_len - v.shape[1]
            if diff_len > 0:
                p = torch.zeros((N, diff_len, d), device=v.device)
                v = torch.cat([v, p], dim=1)
                views[i] = v

        return views


