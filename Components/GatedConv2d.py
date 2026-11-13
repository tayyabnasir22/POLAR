import torch.nn as nn
import torch

class GatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
 
    def forward(self, x):
        feat, gate = self.conv(x).chunk(2, dim=1)
        return self.norm(self.act(feat) * torch.sigmoid(gate))