# mlp_model.py
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    简单的全连接 MLP 回归模型
    输入: x (batch, in_dim)
    输出: y_hat (batch,)
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

    def forward(self, x):
        """
        x: (B, in_dim)
        返回: (B,) 的归一化后延时
        """
        h = self.backbone(x)
        y = self.head(h).squeeze(-1)
        return y
