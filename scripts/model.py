import torch
import torch.nn as nn


class DisentangledRegressor(nn.Module):
    def __init__(self, in_dim, hid=128, z_dim=64, design_dim_override=None):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.design_dim_override = design_dim_override
        self.split_node = nn.Linear(hid, z_dim)
        self.split_design = None if design_dim_override is not None else nn.Linear(hid, z_dim)
        design_dim = design_dim_override if design_dim_override is not None else z_dim

        self.head = nn.Sequential(
            nn.Linear(z_dim + design_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid // 2), nn.ReLU(),
        )
        self.mu = nn.Linear(hid // 2, 1)
        self.log_var = nn.Linear(hid // 2, 1)

        # === 新增：初始化逻辑 ===
        # 将 log_var 层的权重和偏置初始化为 0
        # 这样初始输出 log_var ≈ 0 -> var ≈ 1，符合标准化后的数据分布
        nn.init.zeros_(self.log_var.weight)
        nn.init.zeros_(self.log_var.bias)
        # 2. [新增] 初始化 mu 为 0 -> 初始预测值为 0 (即均值)
        # 这能保证初始 Loss = 0.5 * log(2pi) + 0 ≈ 1.4，绝对不会爆炸
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        # =======================

    def forward(self, x, z_design_external=None):
        h = self.enc(x)
        z_node = self.split_node(h)
        if self.design_dim_override is not None:
            assert z_design_external is not None, "design_dim_override set but z_design_external is None"
            z_design = z_design_external.expand(x.shape[0], -1)
        else:
            z_design = self.split_design(h)
        z = torch.cat([z_node, z_design], dim=1)
        h2 = self.head(z)
        mu = self.mu(h2).squeeze(-1)
        log_var = self.log_var(h2).squeeze(-1)
        return mu, log_var, z_node, z_design