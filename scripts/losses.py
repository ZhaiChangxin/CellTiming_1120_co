# === Python代码文件: losses.py (已修复) ===
import torch
import torch.nn as nn
import numpy as np


def gaussian_nll(y, mu, log_var, reduction='mean'):
    """
    计算高斯负对数似然 (Gaussian Negative Log-Likelihood)。
    用于回归任务，假设模型输出均值 mu 和对数方差 log_var。

    公式: NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2) + C
    """
    # 限制 log_var 防止数值爆炸 (可选，视稳定性而定)
    log_var = torch.clamp(log_var, min=-10, max=10)

    sigma_2 = torch.exp(log_var)

    # loss term
    loss = 0.5 * (log_var + (y - mu) ** 2 / sigma_2)

    # 加上常数项 0.5 * log(2π) 让数值具有统计意义 (优化时可忽略，但为了严谨加上)
    loss = loss + 0.5 * np.log(2 * np.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def kl_divergence(z_q, z_p):
    """
    计算两个高斯分布之间的 KL 散度: KL(Q || P)

    Args:
        z_q: 变分后验分布 Q(z|x)，格式为 tuple (mu_q, logvar_q)
        z_p: 先验分布 P(z)，格式为 tuple (mu_p, logvar_p)。
             如果 z_p 为 None，则默认 P(z) 为标准正态分布 N(0, I)。
    """
    mu_q, logvar_q = z_q

    if z_p is None:
        # KL(N(mu, var) || N(0, 1))
        # 公式: -0.5 * sum(1 + log(var) - mu^2 - var)
        # dim=1 表示在 latent 维度求和
        kld = -0.5 * torch.sum(1 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=1)
    else:
        # KL(N(mu1, var1) || N(mu2, var2))
        mu_p, logvar_p = z_p

        # 公式: 0.5 * sum(log(var2/var1) + (var1 + (mu1-mu2)^2)/var2 - 1)
        kld = 0.5 * torch.sum(
            logvar_p - logvar_q - 1 +
            (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp(),
            dim=1
        )

    return kld.mean()  # 对 batch 取平均


def total_loss(y_true, mu_pred, logvar_pred, z_q, z_p, kl_weight=0.01):
    """
    总损失函数计算。

    Args:
        y_true: 真实标签 (Batch, 1)
        mu_pred: 预测均值 (Batch, 1)
        logvar_pred: 预测对数方差 (Batch, 1)
        z_q: Encoder 输出的潜在分布参数 (mu, logvar)
        z_p: 先验分布参数 (通常为 None 或设计向量分布)
        kl_weight: KL 散度的权重系数 (beta-VAE 中的 beta)

    Returns:
        loss: 总损失 (反向传播用)
        recon_loss_item: 回归损失数值 (日志用)
        kl_loss_item: KL 散度数值 (日志用)
    """

    # 1. 回归损失 (Reconstruction / Prediction Loss)
    # 使用 NLL 而不是 MSE，是为了让模型学习预测的不确定性
    recon_loss = gaussian_nll(y_true, mu_pred, logvar_pred)

    # 2. KL 散度 (Regularization)
    # 检查 z_q 是否是 tuple (mu, logvar)，如果模型是确定性的 (Deterministic)，可能只传回 Tensor
    kl_loss = torch.tensor(0.0, device=y_true.device)

    if isinstance(z_q, (tuple, list)) and len(z_q) == 2:
        kl_loss = kl_divergence(z_q, z_p)

    # 3. 总损失
    loss = recon_loss + kl_weight * kl_loss

    return loss, recon_loss.item(), kl_loss.item()
