import torch.nn as nn
import torch
from Components.ConvOps import ConvOps
from Components.ResidualGroup import ResidualGroup

class DetailsInjectionFusion(nn.Module):
    def __init__(self, num_feats: int, kernel_size: int, n_resblocks: int = 6):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.GroupNorm(8, num_feats),
        )
        self.rho = nn.Parameter(torch.tensor(0.0))
        self.refiner = ResidualGroup(
            ConvOps.DefaultConv, num_feats, kernel_size,
            reduction=16, n_resblocks=n_resblocks,
        )

    def forward(self, depth_features, refinement=None):
        if refinement is None:
            return self.refiner(depth_features)

        refinement = self.projector(refinement)
        fused = depth_features + self.rho * refinement
        return self.refiner(fused)