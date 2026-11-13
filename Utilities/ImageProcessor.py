import torch
import torch.nn.functional as F

class ImageProcessor:            
    @staticmethod
    def PadToSize(x, patch_size_h, patch_size_w):
        if x.dim() == 4:
            _, _, h, w = x.shape
        elif x.dim() == 3:
            _, h, w = x.shape
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        pad_h = patch_size_h - h
        pad_w = patch_size_w - w

        # F.pad format: (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0), h, w
    
    @staticmethod
    def CropFromTop(x, orig_h, orig_w):
        if x.dim() == 4:
            return x[:, :, :orig_h, :orig_w]
        elif x.dim() == 3:
            return x[:, :orig_h, :orig_w]
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")