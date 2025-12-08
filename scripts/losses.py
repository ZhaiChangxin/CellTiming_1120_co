# === Python代码文件: losses.py (已修复) ===

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNLL(nn.Module):
    def __init__(self,
                 min_log_var: float = -4.0,
                 max_log_var: float = 4.0,
                 max_abs: float = 10.0):
        """
        min_log_var / max_log_var: 控制预测方差的范围。
        max_abs: 控制预测均值的最大幅度。
        """
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.max_abs = max_abs

    def forward(self, mu, log_var, y):
        """
        这里的 mu 和 log_var 是模型直接输出的、未经限制的原始值。
        我们在这里用 tanh 将它们平滑地映射到目标范围。
        """

        # ------------------  【关键修改】 ------------------
        # 1) 使用 tanh 将模型的原始 log_var 输出平滑地映射到 [min_log_var, max_log_var] 区间。
        #    这保证了梯度总是可以回传，不会因为 clamp 而变为 0。
        log_var_range = self.max_log_var - self.min_log_var
        log_var_mid = (self.max_log_var + self.min_log_var) / 2.0
        # `tanh` 输出在 (-1, 1), 经过缩放和平移后，结果被限制在期望的区间内
        log_var = log_var_mid + (log_var_range / 2.0) * torch.tanh(log_var)

        # 2) 对 mu 做同样的处理，将其映射到 [-max_abs, max_abs] 区间。
        #    这步虽然不如 log_var 关键，但也是一个好习惯，可以增强稳定性。
        mu = self.max_abs * torch.tanh(mu / self.max_abs)

        # 3) 对真实标签 y 的 clamp 依然保留，以防止数据中的极端异常值影响训练。
        if self.max_abs is not None:
            y = torch.clamp(y, -self.max_abs, self.max_abs)
        # ----------------------------------------------------

        # 4) 计算方差 var
        var = torch.exp(log_var)

        # 5) 标准高斯 NLL 计算
        return 0.5 * (torch.log(2 * torch.pi * var)
                      + (y - mu) ** 2 / var).mean()


def cmd_loss(x, y, K=3, eps=1e-6):
    # 中心化和标准化
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)
    y = (y - y.mean(0, keepdim=True)) / (y.std(0, keepdim=True) + eps)

    # 计算各阶矩的差异
    loss = (x.mean(0) - y.mean(0)).pow(2).sum()
    for k in range(2, K + 1):
        loss = loss + ((x ** k).mean(0) - (y ** k).mean(0)).pow(2).sum()

    return loss / x.shape[1]


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        sim = torch.matmul(feats, feats.t()) / self.t

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()

        # for numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        eye = torch.eye(sim.size(0), device=sim.device)

        # 计算分母
        exp_sim = torch.exp(sim) * (1 - eye)
        all_sum = exp_sim.sum(dim=1)

        # 计算分子
        pos_mask = mask - eye
        pos_sum = (exp_sim * pos_mask).sum(dim=1)

        # 计算 loss
        loss = -torch.log((pos_sum + 1e-9) / (all_sum + 1e-9))

        return loss.mean()
