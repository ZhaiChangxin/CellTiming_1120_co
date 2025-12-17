# === Python代码文件: layers.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. Gradient Reversal Layer (Stage 2)
# ==========================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha


# ==========================================
# 2. Bayesian Linear Layer (Stage 3)
# ==========================================
class BayesianLinear(nn.Module):
    """
    使用 Reparameterization Trick 实现的贝叶斯线性层。
    权重 W ~ N(mu, sigma), sigma = log(1 + exp(rho))
    """

    def __init__(self, in_features, out_features, prior_mu=0.0, prior_sigma=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重参数 (Variational Posterior parameters)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

        # 偏置参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        # 初始化逻辑将在 convert 函数中处理，这里给默认随机
        self.reset_parameters()

        # Prior (通常设为 Stage 2 的训练结果或标准正态)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -3.0)  # rho=-3 -> sigma ~ 0.05
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x):
        if self.training:
            # 采样 W 和 b
            w_sigma = torch.log1p(torch.exp(self.weight_rho))
            b_sigma = torch.log1p(torch.exp(self.bias_rho))

            w_epsilon = torch.randn_like(self.weight_mu)
            b_epsilon = torch.randn_like(self.bias_mu)

            weight = self.weight_mu + w_sigma * w_epsilon
            bias = self.bias_mu + b_sigma * b_epsilon
        else:
            # 推理时使用均值 (Maximum A Posteriori estimate 简化)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_loss(self):
        """计算 KL(q(w) || p(w))"""
        w_sigma = torch.log1p(torch.exp(self.weight_rho))
        b_sigma = torch.log1p(torch.exp(self.bias_rho))

        # 简化版高斯 KL 公式 (假设先验是对角高斯)
        # KL = log(sig_p/sig_q) + (sig_q^2 + (mu_q - mu_p)^2)/(2*sig_p^2) - 0.5

        kl_w = torch.log(self.prior_sigma / w_sigma) + \
               (w_sigma ** 2 + (self.weight_mu - self.prior_mu) ** 2) / (2 * self.prior_sigma ** 2) - 0.5

        kl_b = torch.log(self.prior_sigma / b_sigma) + \
               (b_sigma ** 2 + (self.bias_mu - 0.0) ** 2) / (2 * self.prior_sigma ** 2) - 0.5

        return kl_w.sum() + kl_b.sum()
