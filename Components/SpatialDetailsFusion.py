import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialDetailsFusion(nn.Module):
    def __init__(self, hidden_dim: int):
        super(SpatialDetailsFusion, self).__init__()
        
        self.structure_extractor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self.texture_extractor = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.SiLU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, dilation=4, padding=4),
            nn.SiLU(),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        self.band_projector_lh_hl = nn.Sequential(
            nn.Conv2d(2 * 3, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
                
        self.band_projector_hh = nn.Sequential(
            nn.Conv2d(3*3, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )
    
    def forward(self, lh, hl, hh):        
        struct_v = self.band_projector_lh_hl(torch.cat([lh, hl], dim=1))
        structure = self.structure_extractor(struct_v)

        gated_structure = structure
        texture_v = self.band_projector_hh(torch.cat([lh, hl, hh], dim=1))
        texture = self.texture_extractor(texture_v)
        gated_texture = texture

        return self.refine(gated_structure + gated_texture)