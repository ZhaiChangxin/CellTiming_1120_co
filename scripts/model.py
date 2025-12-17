# === Python代码文件: model.py ===

import torch
import torch.nn as nn


class DisentangledRegressor(nn.Module):
    def __init__(self, in_dim, hid=128, design_dim_override=0):
        """
        in_dim: 输入特征 x 的维度
        hid:    隐藏层维度
        design_dim_override: GNN 输出的 z_d 维度
        """
        super().__init__()

        # 1. 编码器部分 (Encoder for x)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )

        design_dim = design_dim_override if design_dim_override > 0 else hid

        # 2. 隐变量生成层 (Variational Layers for zn)
        # 先提取特征，不再直接输出 zn
        self.split_pre = nn.Sequential(
            nn.Linear(hid + design_dim, hid),
            nn.ReLU()
        )

        # 预测 zn 的均值和对数方差
        self.zn_mu = nn.Linear(hid, hid)
        self.zn_logvar = nn.Linear(hid, hid)

        # ==================== 预测头 ====================
        # self.head 输入维度: h (hid) + zn (hid)
        self.head = nn.Sequential(
            nn.Linear(hid + hid, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU()
        )
        # ===============================================

        # 输出层 (预测目标值的均值和方差)
        self.mu = nn.Linear(hid, 1)
        self.log_var = nn.Linear(hid, 1)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧 (Reparameterization Trick)
        训练时: z = mu + std * eps
        推理时: z = mu (确定性)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, z_d):
        """
        x:   输入特征, shape: [B, in_dim]
        z_d: GNN 嵌入, shape: [B, design_dim]

        Returns:
            mu_y: 预测值的均值
            log_var_y: 预测值的对数方差
            (zn_mu, zn_logvar): 隐变量 zn 的分布参数 (用于计算 KL Loss)
            zn: 采样后的隐变量
        """
        # 1. 编码 x -> h
        h = self.enc(x)  # h.shape: [B, hid]

        # 2. 拼接 h 和 z_d -> zx
        zx = torch.cat([h, z_d], dim=1)  # zx.shape: [B, hid + design_dim]

        # 3. 计算 zn 的分布
        pre_zn = self.split_pre(zx)
        zn_mu = self.zn_mu(pre_zn)
        zn_logvar = self.zn_logvar(pre_zn)

        # 4. 采样 zn
        zn = self.reparameterize(zn_mu, zn_logvar)

        # 5. 拼接 h 和 zn 进行预测
        z = torch.cat([h, zn], dim=1)  # z.shape: [B, hid + hid]
        h_main = self.head(z)  # h_main.shape: [B, hid]

        # 6. 输出预测结果
        mu_y = self.mu(h_main)
        log_var_y = self.log_var(h_main)

        # 返回元组结构，其中 (zn_mu, zn_logvar) 将被 losses.py 识别并计算 KL
        return mu_y.squeeze(-1), log_var_y.squeeze(-1), (zn_mu, zn_logvar), zn
