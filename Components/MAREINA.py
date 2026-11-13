from Components.TrigBalancedFusion import TrigBalancedFusion
import torch.nn as nn
from Components.DepthEncoder import DepthEncoder
from Components.RGBEncoder import RGBEncoder

class MAREINA(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        per_channel_gate: bool = True,
        n_depth_blocks: int = 6,
        real: bool = True
    ):
        super().__init__()

        self.depth_enc = DepthEncoder(hidden_dim, n_depth_blocks, kernel_size)
        self.rgb_encoder = RGBEncoder(hidden_dim, kernel_size)

        self.init_rgb_projection = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )

        self.fusion = TrigBalancedFusion(hidden_dim, per_channel=per_channel_gate, real=real)

    def forward(self, ll, depth_feat, conf=None):
        rgb0 = self.init_rgb_projection(ll)
        encoded_rgb = self.rgb_encoder(rgb0)
        encoded_depth = self.depth_enc(depth_feat)

        return self.fusion(
            encoded_depth, encoded_rgb, conf
        )