from Components.BoundaryAwareRefinementBlock import BoundaryAwareRefinementBlock
from Components.DetailsInjectionFusion import DetailsInjectionFusion
from Components.DownsampleX16 import DownsampleX16
from Components.MAREINA import MAREINA
from Components.DepthRestorationBranch import DepthRestorationBranch
from pytorch_wavelets import DWTForward
import torch.nn as nn
import torch.nn.functional as F
import torch
from Components.DinoV3.Dino3VisionTransformer import Dino3VisionTransformer
from Components.SpatialDetailsFusion import SpatialDetailsFusion
from Components.UpscaleDenseProjection import UpscaleDenseProjection
from Components.BoundaryStudent import BoundaryStudent

class POLAR(nn.Module):
    def __init__(
            self,
            use_pretrained: bool,
            num_feats: int = 64,
            img_size: int = 256,
            patch_size: int = 16,
            scale: int = 4,
            kernel_size=3,
            freeze_dino: bool = True,
            train_mode: bool = True
        ):
        super(POLAR, self).__init__()
        self.training_mode = train_mode
        self.patch_size = patch_size

        self.init_depth_projection = DownsampleX16(1, num_feats)

        self.upsampler1 = UpscaleDenseProjection(num_feats, num_feats*2, 2)
        self.upsampler2 = UpscaleDenseProjection(num_feats*2, num_feats*3, 2)
        self.upsampler3 = UpscaleDenseProjection(num_feats*3, num_feats*4, 2)

        self.final_upscale = UpscaleDenseProjection(num_feats*4, num_feats*4, 2)

        self.semantics_encoder = Dino3VisionTransformer(
            img_size=img_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
            ffn_ratio=4,
            patch_size=self.patch_size,
            mask_k_bias=True,
            layerscale_init=1e-5,
            n_storage_tokens=4,
            ffn_layer="mlp",
        )

        if use_pretrained == True:
            ckpt = torch.hub.load_state_dict_from_url(
                "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNHpiOTJ3enZpNjFwYWF2cHp2cHFnODcyIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NzY0OTY1MjB9fX1dfQ__&Signature=SjHrDsrDPKklzUzy5JUdtle8aLDtw9wkskyfT4M0702RunMey3Y5Jna61ewUYo-ynCdCpJ%7EIn2EHa1Sv7nuZj6-X-7lfTk2SqYq4Jxgk%7E85bwL2P2i3a-PLG393tGtFPWEdN3ovcW7ScIGrW8dM0XRj9iWqYA1SS2wUN3d71YKV2hjwIgi36qqnYGL9TAIcLeQzNvoT1spMLk-PYeUXuppOLa%7EzzXBT88DUR-oMZPthDR3FU8mBbaARJBa2IfF10F4CuNFtfx5m1YWseoCS%7EGRqBzpOGixS01UUgR6u%7Ed2Nx3gEHXeLP1Pi2kmVf%7E4538qS5MnrkT-qFYF7eOhZ6wA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=989805356906888",
                map_location="cpu"
            )

            filtered_state = {}
            for k, v in ckpt.items():
                if k.startswith("head") or k == "pos_embed":
                    continue
                filtered_state[k] = v

            missing, unexpected = self.semantics_encoder.load_state_dict(filtered_state, strict=False)

            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        if freeze_dino == True:
            for param in self.semantics_encoder.parameters():
                param.requires_grad = False
            self.semantics_encoder.eval()

        self.boundary_student = BoundaryStudent(in_ch=3, width=32)

        self.wavelet_encoder = DWTForward(J=1, wave='haar')

        self.depth_builder1 = MAREINA(num_feats, kernel_size)
        self.depth_builder2 = MAREINA(num_feats*2, kernel_size)
        self.depth_builder3 = MAREINA(num_feats*3, kernel_size)
        self.depth_builder4 = MAREINA(num_feats*4, kernel_size)

        self.details_builder1 = SpatialDetailsFusion(num_feats)
        self.details_builder2 = SpatialDetailsFusion(num_feats*2)
        self.details_builder3 = SpatialDetailsFusion(num_feats*3)
        self.details_builder4 = SpatialDetailsFusion(num_feats*4)

        self.degradation_fix = DepthRestorationBranch(num_feats)

        self.fusion1 = DetailsInjectionFusion(num_feats, kernel_size)
        self.fusion2 = DetailsInjectionFusion(num_feats*2, kernel_size)
        self.fusion3 = DetailsInjectionFusion(num_feats*3, kernel_size)
        self.fusion4 = DetailsInjectionFusion(num_feats*4, kernel_size)

        self.residual1 = nn.Sequential(
            nn.Conv2d(in_channels=num_feats*2, out_channels=num_feats*2, kernel_size=1),
            nn.GroupNorm(8, num_feats*2),
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(in_channels=num_feats*3, out_channels=num_feats*3, kernel_size=1),
            nn.GroupNorm(8, num_feats*3),
        )

        self.residual3 = nn.Sequential(
            nn.Conv2d(in_channels=num_feats*4, out_channels=num_feats*4, kernel_size=1),
            nn.GroupNorm(8, num_feats*4),
        )

        self.last_conv = nn.Sequential(
            nn.Conv2d(num_feats * 4, num_feats * 3, kernel_size=3, bias=True, padding=1),
            nn.SiLU(),

            nn.Conv2d(num_feats * 3, num_feats * 2, kernel_size=3, bias=True, padding=1),
            nn.SiLU(),

            nn.Conv2d(num_feats * 2, num_feats, kernel_size=3, bias=True, padding=1),
            nn.SiLU(),

            nn.Conv2d(num_feats, num_feats // 2, kernel_size=3, bias=True, padding=1),
            nn.SiLU(),

            nn.Conv2d(num_feats // 2, 1, kernel_size=3, bias=True, padding=1),
        )

        self.boundary_map = BoundaryAwareRefinementBlock(3, num_feats)
        self.residual4 = nn.Sequential(
            nn.Conv2d(in_channels=num_feats*4 + num_feats, out_channels=num_feats*4, kernel_size=1),
            nn.GroupNorm(8, num_feats*4),
        )

    @torch.no_grad()
    def _teacher_boundary(self, image, patch_h, patch_w):
        last_token = self.semantics_encoder.forward_features(image)['x_norm_patchtokens']
        B, N, C = last_token.shape
        last_token = last_token.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
        return self.boundary_map.semantic_boundary_map(last_token)

    def dwt_full_pyramid(self, x, level=4, wave='haar'):
        LLs = []
        LHs = []
        HLs = []
        HHs = []
        current = x
        for i in range(level):
            yl, yh = self.wavelet_encoder(current)
            LLs.append(yl)

            y = yh[0]  # (B, C, 3, H, W)

            LHs.append(y[:, :, 0, :, :])
            HLs.append(y[:, :, 1, :, :])
            HHs.append(y[:, :, 2, :, :])

            current = yl

        return LLs, LHs, HLs, HHs

    def forward(self, image, depth):
        clean_lr, conf = self.degradation_fix(depth)

        H = image.shape[-2]
        W = image.shape[-1]

        patch_h, patch_w = H // self.patch_size, W // self.patch_size


        depth_bicubic = F.interpolate(
            depth,
            size=image.shape[-2:],
            mode='bicubic'
        )

        depth0 = self.init_depth_projection(depth_bicubic)

        LLs, LHs, HLs, HHs = self.dwt_full_pyramid(image)

        depth1 = self.depth_builder1(LLs[3], depth0, conf)
        details1 = self.details_builder1(LHs[3], HLs[3], HHs[3])
        depth1 = self.fusion1(depth1, details1)

        res1 = self.residual1(self.upsampler1(depth1))

        depth2 = self.depth_builder2(LLs[2], res1, conf)
        details2 = self.details_builder2(LHs[2], HLs[2], HHs[2])
        depth2 = self.fusion2(depth2, details2)

        res2 = self.residual2(self.upsampler2(depth2))

        depth3 = self.depth_builder3(LLs[1], res2, conf)
        details3 = self.details_builder3(LHs[1], HLs[1], HHs[1])
        depth3 = self.fusion3(depth3, details3)

        res3 = self.residual3(self.upsampler3(depth3))

        depth4 = self.depth_builder4(LLs[0], res3, conf)
        details4 = self.details_builder4(LHs[0], HLs[0], HHs[0])
        depth4 = self.fusion4(depth4, details4)

        res4 = self.final_upscale(depth4)

        student_boundary = None
        teacher_boundary = None

        student_boundary = self.boundary_student(image)
        boundary = student_boundary

        if self.training_mode and self.semantics_encoder is not None:
            teacher_boundary = self._teacher_boundary(image, patch_h, patch_w)
    
        refined, _ = self.boundary_map(image, boundary, depth)

        res4 = self.residual4(torch.cat([res4, refined], dim=1))

        out = self.last_conv(res4)

        aux = {
            'student_boundary': student_boundary,
            'teacher_boundary': teacher_boundary,
        }
        return out + depth_bicubic, clean_lr, aux