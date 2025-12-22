import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, num_numeric_features, num_cell_types, embed_dim=16, hid_dim=128):
        super().__init__()

        # 1. Cell Type 嵌入层: 将离散的 cell_type 映射为 dense vector
        self.cell_embedding = nn.Embedding(num_cell_types, embed_dim)

        # 2. 特征融合后的输入维度
        in_dim = num_numeric_features + embed_dim

        # 3. 主干网络 (3层 MLP)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, 1)  # 输出归一化后的 delay
        )

    def forward(self, x_numeric, cell_ids):
        """
        x_numeric: [batch, num_numeric_features]
        cell_ids:  [batch] (int indices)
        """
        # 获取 Cell 向量
        c_emb = self.cell_embedding(cell_ids)  # [batch, embed_dim]

        # 拼接数值特征和 Cell 特征
        x = torch.cat([x_numeric, c_emb], dim=1)

        # 前向传播
        out = self.net(x)
        return out.squeeze(-1)  # [batch]
