# === Python代码文件: model.py (最终修正版) ===

import torch
import torch.nn as nn


class DisentangledRegressor(nn.Module):
    def __init__(self, in_dim, hid=128, design_dim_override=0):
        """
        in_dim: 表格数据 x 的维度
        hid:    内部隐藏层维度
        design_dim_override: GNN 输出的 z_d 的维度
        """
        super().__init__()

        # 用于处理表格数据 x 的编码器
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )

        design_dim = design_dim_override if design_dim_override > 0 else hid

        # 用于生成“无关”向量 zn 的模块
        self.split_node = nn.Sequential(
            nn.Linear(hid + design_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )

        # ==================== 【关键修正】 ====================
        # self.head 的输入是 h 和 zn 的拼接，所以输入维度应该是 hid + hid
        self.head = nn.Sequential(
            nn.Linear(hid + hid, hid),  # 原错误为: nn.Linear(in_dim + hid, hid)
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU()
        )
        # =======================================================

        # 最终输出头
        self.mu = nn.Linear(hid, 1)
        self.log_var = nn.Linear(hid, 1)

    def forward(self, x, z_d):
        """
        x:   表格特征, shape: [B, in_dim]
        z_d: GNN 设计向量, shape: [B, design_dim]
        """
        # 1. 编码表格特征
        h = self.enc(x)  # h.shape: [B, hid]

        # 2. 拼接 h 和 z_d，生成“无关”向量 zn
        zx = torch.cat([h, z_d], dim=1)  # zx.shape: [B, hid + design_dim]
        zn = self.split_node(zx)  # zn.shape: [B, hid]

        # 3. 拼接 h 和 zn，送入主回归头
        z = torch.cat([h, zn], dim=1)  # z.shape: [B, hid + hid]
        h_main = self.head(z)  # h_main.shape: [B, hid]

        # 4. 得到最终预测
        mu = self.mu(h_main)
        log_var = self.log_var(h_main)

        return mu.squeeze(-1), log_var.squeeze(-1), zn, h
