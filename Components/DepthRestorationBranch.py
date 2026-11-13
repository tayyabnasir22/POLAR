from Components.ConvOps import ConvOps
from Components.ResidualGroup import ResidualGroup
import torch.nn as nn
import torch

class DepthRestorationBranch(nn.Module):
    def __init__(self, num_feats: int, kernel_size: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, num_feats, 3, padding=1),
            nn.GroupNorm(8, num_feats),
            nn.SiLU(),
        )
        self.body = ResidualGroup(ConvOps.DefaultConv, num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.to_residual = nn.Conv2d(num_feats, 1, 3, padding=1)   # corruption estimate
 
        # log-space temperature: tau = exp(log_tau) > 0 always.
        log_tau = torch.log(torch.tensor(float(0.1)))
        self.log_tau = nn.Parameter(log_tau)

    def forward(self, depth_lr):
        feat = self.body(self.stem(depth_lr))
        residual = self.to_residual(feat)
        clean_lr = depth_lr - residual  # DnCNN (Zhang et al., 2017)
 
        # Confidence from correction magnitude, gradient-isolated from
        # the restoration path.
        tau = self.log_tau.exp().clamp_min(1e-4)
        conf = torch.exp(-residual.detach().abs() / tau)
 
        return clean_lr, conf