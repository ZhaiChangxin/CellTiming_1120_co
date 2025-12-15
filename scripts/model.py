import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ==========================================
#   工具组件: 梯度反转 & 贝叶斯层
# ==========================================

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class BayesianLinear(nn.Module):
    """
    贝叶斯线性层，权重服从高斯分布 w ~ N(mu, sigma)。
    """

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 变分参数: 均值和 log(sigma)
        # [mu] 初始化将由外部传入（权重继承），这里先随机
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        # 初始化
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -5.0)  # [优化] 初始方差设得更小一点，让它先像个确定性网络
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -5.0)

        # 先验分布的标准差
        self.prior_log_sigma = torch.log(torch.tensor(prior_sigma))

    def forward(self, input):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        if self.training:
            w_epsilon = torch.randn_like(weight_sigma)
            b_epsilon = torch.randn_like(bias_sigma)
            weight = self.weight_mu + weight_sigma * w_epsilon
            bias = self.bias_mu + bias_sigma * b_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

    def kl_loss(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        # 简化版 KL
        kl = 0.5 * (self.prior_log_sigma - torch.log(weight_sigma) +
                    (weight_sigma ** 2 + self.weight_mu ** 2) / (2 * torch.exp(self.prior_log_sigma) ** 2) - 0.5).sum()
        kl += 0.5 * (self.prior_log_sigma - torch.log(bias_sigma) +
                     (bias_sigma ** 2 + self.bias_mu ** 2) / (2 * torch.exp(self.prior_log_sigma) ** 2) - 0.5).sum()
        return kl


class DomainClassifier(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.BatchNorm1d(hid),
            nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.layer(x)


# ==========================================
#   主模型: DisentangledRegressor
# ==========================================

def bnn_from_linear(linear_layer, dropout_p=0.0):
    """
    [核心修复] 辅助函数：将普通 Linear 层转换为 BayesianLinear，并继承权重！
    """
    if isinstance(linear_layer, nn.Linear):
        bnn = BayesianLinear(linear_layer.in_features, linear_layer.out_features)
        # [权重继承] 关键步骤！
        bnn.weight_mu.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            bnn.bias_mu.data = linear_layer.bias.data.clone()
        else:
            bnn.bias_mu.data.zero_()
        return bnn
    return linear_layer


class DisentangledRegressor(nn.Module):
    def __init__(self, in_dim, hid=128, design_dim_override=0):
        super().__init__()
        self.hid = hid

        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )

        design_dim = design_dim_override if design_dim_override > 0 else hid

        self.split_node = nn.Sequential(
            nn.Linear(hid + design_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid)
        )

        self.head_in_dim = hid + hid
        # 注意：这里我们明确定义每一层，方便后续遍历替换
        self.head = nn.Sequential(
            nn.Linear(self.head_in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU()
        )

        self.mu = nn.Linear(hid, 1)
        self.log_var = nn.Linear(hid, 1)

        self.is_bnn = False

    def forward(self, x, z_d, return_feat=False):
        h = self.enc(x)
        zx = torch.cat([h, z_d], dim=1)
        zn = self.split_node(zx)
        z_combined = torch.cat([h, zn], dim=1)

        if return_feat:
            return z_combined

        h_main = self.head(z_combined)
        mu = self.mu(h_main)
        log_var = self.log_var(h_main)

        return mu.squeeze(-1), log_var.squeeze(-1), zn, h

    # =========================================================
    #  [核心修复] 修改后的 convert_to_bnn
    # =========================================================
    def convert_to_bnn(self, dropout_p=0.0):
        """
        [Phase 3] 将回归头转换为 BNN，并继承 Phase 2 的权重 (Warm Start)
        """
        print(f"[Model] Converting Prediction Head to Bayesian Layers (Weights Inherited, Dropout={dropout_p})...")
        device = next(self.parameters()).device

        # 1. 转换 Head (Hidden Layers)
        # 我们需要手动提取原有的 Linear 层并复制权重
        old_head = self.head
        new_layers = []

        # 假设 head 结构是 Linear -> ReLU -> Linear -> ReLU
        # 我们按顺序重建

        # Layer 1
        layer1_bnn = bnn_from_linear(old_head[0])
        new_layers.append(layer1_bnn)
        if dropout_p > 0: new_layers.append(nn.Dropout(dropout_p))
        new_layers.append(nn.ReLU())

        # Layer 2
        layer2_bnn = bnn_from_linear(old_head[2])
        new_layers.append(layer2_bnn)
        if dropout_p > 0: new_layers.append(nn.Dropout(dropout_p))
        new_layers.append(nn.ReLU())

        self.head = nn.Sequential(*new_layers).to(device)

        # 2. 转换输出层 (mu / log_var)
        self.mu = bnn_from_linear(self.mu).to(device)
        self.log_var = bnn_from_linear(self.log_var).to(device)

        self.is_bnn = True

    def get_bnn_kl_loss(self):
        kl = 0.0
        if not self.is_bnn:
            return kl
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                kl += m.kl_loss()
        return kl
