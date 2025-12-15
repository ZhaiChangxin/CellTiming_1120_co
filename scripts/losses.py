import torch
import torch.nn.functional as F


def gaussian_nll(y, mu, log_var, epsilon=1e-6):
    """
    计算高斯分布的负对数似然 (Negative Log Likelihood).
    用于回归任务中的不确定性估计。

    Loss = 0.5 * (log_var + (y - mu)^2 / exp(log_var)) + C
    """
    # 确保 log_var 不会造成数值不稳定
    # 限制 log_var 范围，防止 exp(log_var) 变为 0 或无穷大
    log_var = torch.clamp(log_var, min=-10, max=10)
    var = torch.exp(log_var)

    # NLL 公式
    nll = 0.5 * (log_var + (y - mu) ** 2 / (var + epsilon))
    return nll.mean()  # 返回 batch 的平均值


def latent_kl_loss(z_q, z_p):
    """
    计算潜变量的 KL 散度 (用于 Phase 1 & 2 的解耦正则化).
    假设 z_q, z_p 是 (mu, logvar) 的元组，或者如果是 Tensor 则计算 L2 距离。
    """
    if isinstance(z_q, tuple) and isinstance(z_p, tuple):
        # 两个高斯分布之间的 KL
        mu_q, logvar_q = z_q
        mu_p, logvar_p = z_p
        kl = 0.5 * (logvar_p - logvar_q - 1 +
                    (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p))
        return kl.mean()
    else:
        # 如果不是分布参数，退化为简单的特征距离 (MSE)
        # 对应原有代码中可能的 latent matching
        return F.mse_loss(z_q, z_p)


def total_loss(y_true, y_mu, y_log_var, z_q=None, z_p=None, kl_weight=0.01):
    """
    综合损失函数

    参数:
    y_true: 真实标签
    y_mu: 预测均值
    y_log_var: 预测方差的对数
    z_q: 变分后验潜变量 (Phase 3 可为 None)
    z_p: 先验潜变量 (Phase 3 可为 None)
    kl_weight: 潜变量 KL 损失的权重

    返回:
    (total_loss, nll_loss, kl_loss)
    """

    # 1. 任务回归损失 (NLL) - 核心部分
    # 这部分驱动模型去拟合 y，并学习数据本身的噪声 (Aleatoric Uncertainty)
    nll = gaussian_nll(y_true, y_mu, y_log_var)

    # 2. 潜变量正则化 (Latent Regularization)
    # 仅在 Phase 1 & 2 启用，用于特征解耦
    kl = torch.tensor(0.0, device=y_true.device)
    if z_q is not None and z_p is not None:
        kl = latent_kl_loss(z_q, z_p)

    # 3. 总损失
    loss = nll + kl_weight * kl

    return loss, nll, kl
