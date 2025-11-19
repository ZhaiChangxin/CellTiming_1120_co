
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianNLL(nn.Module):
    def forward(self, mu, log_var, y):
        var = torch.exp(log_var).clamp_min(1e-9)
        return 0.5 * (torch.log(2*torch.pi*var) + (y - mu)**2 / var).mean()

#def cmd_loss(x, y, K=5):
    #def moment(vec, k):
        mean = vec.mean(dim=0, keepdim=True)
        c = vec - mean
        return (c ** k).mean(dim=0)
    #loss = (x.mean(dim=0) - y.mean(dim=0)).pow(2).sum()
    #for k in range(2, K+1):
        loss = loss + (moment(x, k) - moment(y, k)).pow(2).sum()
    #return loss
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
