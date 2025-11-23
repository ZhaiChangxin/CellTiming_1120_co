# cmd_mlp_model.py
import torch
import torch.nn as nn


class CMDMLPRegressor(nn.Module):
    """
    用于 CMD 迁移的 MLP 回归器：
      - backbone: 多层全连接 + ReLU
      - head: 最后一层回归到 1 维
      - forward(x, return_feat):
          return_feat=False -> 返回 y_hat
          return_feat=True  -> 返回 (y_hat, feature)
    """
    def __init__(self, in_dim, hid_dim=128, depth=3, dropout=0.0):
        super().__init__()

        layers = []
        dim = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hid_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            dim = hid_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hid_dim, 1)

    def forward(self, x, return_feat: bool = False):
        """
        x: (B, in_dim)
        return_feat:
          - False: 返回 y_hat (B,)
          - True:  返回 (y_hat (B,), feature (B, hid_dim))
        """
        feat = self.backbone(x)
        y = self.head(feat).squeeze(-1)
        if return_feat:
            return y, feat
        return y
