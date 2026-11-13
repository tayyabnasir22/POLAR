import torch.nn as nn
import torch
from Components.GatedConv2d import GatedConv2d
import torch.nn.functional as F

class ConfidenceGatedMasking(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 3, n_layers: int = 2):
        super().__init__()
        layers = [GatedConv2d(hidden_dim + 1, hidden_dim, kernel_size)]
        for _ in range(n_layers - 1):
            layers.append(GatedConv2d(hidden_dim, hidden_dim, kernel_size))
        self.gated = nn.Sequential(*layers)
 
    def forward(self, depth_feat, conf):
        # Resize confidence from the LR restoration branch to this scale.
        conf_r = F.interpolate(
            conf, size=depth_feat.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        masked = self.gated(torch.cat([depth_feat, conf_r], dim=1))
        return depth_feat + masked  # residual: identity where depth is clean