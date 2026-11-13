import torch.nn as nn
import torch
import torch.nn.functional as F

class TrigBalancedFusion(nn.Module):
    def __init__(self, hidden_dim: int, per_channel: bool = True, eps: float = 1e-6, real: bool = True):
        super().__init__()
        self.eps = eps
        gate_ch = hidden_dim if per_channel else 1

        def score_head(in_ch: int):
            head = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim // 2, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_dim // 2, gate_ch, 1),
            )
            # bias init so softplus(s) ~= 1 for both heads at start
            # -> a_d = a_r -> phi = pi/4 -> equal weighting.
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, 0.5413)  # softplus(0.5413) ~ 1.0
            return head

        # Metric confidence channel (+1 is for the confidence)
        self.depth_score = score_head(hidden_dim + 1 if real else hidden_dim)
        self.rgb_score = score_head(hidden_dim)

    def forward(self, depth_feat, rgb_feat, conf=None):
        if conf is None:
            a_d = F.softplus(self.depth_score(depth_feat))
        else:
            conf_r = F.interpolate(
                conf, size=depth_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

            a_d = F.softplus(self.depth_score(torch.cat([depth_feat, conf_r], dim=1)))
        a_r = F.softplus(self.rgb_score(rgb_feat))
 
        norm = torch.sqrt(a_d ** 2 + a_r ** 2 + self.eps)
        w_d = a_d / norm  # cos(phi)
        w_r = a_r / norm  # sin(phi)
 
        return w_d * depth_feat + w_r * rgb_feat