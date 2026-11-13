import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryStudent(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # /2
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),

            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),

            # /4
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.SiLU(),

            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GroupNorm(8, width * 2),
            nn.SiLU(),

            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),

            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, image):
        b = F.softplus(self.net(image))
        b = b / (b.mean(dim=(2, 3), keepdim=True) + 1e-3)
        return b