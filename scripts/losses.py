
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNLL(nn.Module):
    def __init__(self,
                 min_log_var: float = -4.0,
                 max_log_var: float = 4.0,
                 max_abs: float = 10.0):
        """
        min_log_var / max_log_var:
            控制预测方差的范围：
            var ∈ [exp(-4)≈0.018, exp(4)≈54.6]

        max_abs:
            控制标准化后的 y / mu 的最大幅度。
            因为 y 已经做了 z-score，|y|>10 属于极端离群点，
            截断对主分布几乎没有影响。
        """
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.max_abs = max_abs

    def forward(self, mu, log_var, y):
        # 1) 限制目标和预测幅度（在标准化空间内）
        if self.max_abs is not None:
            mu = torch.clamp(mu, -self.max_abs, self.max_abs)
            y  = torch.clamp(y,  -self.max_abs, self.max_abs)

        # 2) 限制 log_var -> var 的范围，防止 var→0 或 var→过大
        log_var = torch.clamp(log_var,
                              min=self.min_log_var,
                              max=self.max_log_var)
        var = torch.exp(log_var)

        # 3) 标准高斯 NLL
        return 0.5 * (torch.log(2 * torch.pi * var)
                      + (y - mu) ** 2 / var).mean()
def cmd_loss(x, y, K=3, eps=1e-6):
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)
    y = (y - y.mean(0, keepdim=True)) / (y.std(0, keepdim=True) + eps)
    loss = (x.mean(0) - y.mean(0)).pow(2).sum()
    for k in range(2, K+1):
        loss = loss + ((x**k).mean(0) - (y**k).mean(0)).pow(2).sum()
    return loss / x.shape[1]

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature
    def forward(self, feats, labels):
        feats = F.normalize(feats, dim=1)
        sim = torch.matmul(feats, feats.t()) / self.t
        labels = labels.view(-1,1)
        mask = torch.eq(labels, labels.t()).float()
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - logits_max.detach()
        eye = torch.eye(sim.size(0), device=sim.device)
        exp_sim = torch.exp(sim) * (1 - eye)
        pos_mask = mask - eye
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        loss = -torch.log((pos_sum + 1e-9) / (all_sum + 1e-9))
        return loss.mean()
