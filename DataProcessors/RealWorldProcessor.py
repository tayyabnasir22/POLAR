from DataProcessors.DataProcessorBase import DataProcessorBase
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

class RealWorldProcessor(DataProcessorBase):
    def __init__(self, rgb_path: str, depth_path: str, depth_norm_path: str, mask_path: str,
                 min_max_path: str, depth_lr_norm_path: str, repeat: int = 1, augment: bool = False,
                 train: bool = False, use_mask: bool = False):
        # Add mmap_mode='r' in case you have limited RAM
        self.depths = np.load(depth_path)#[:100]
        self.depths_norm = np.load(depth_norm_path)#[:100]
        self.images = np.load(rgb_path)#[:100]
        self.masks = np.load(mask_path)#[:100]
        self.minmax = np.load(min_max_path)#[:100]

        # Real LR depth from the sensor, already normalized with its own min/max
        self.depths_lr_norm = np.load(depth_lr_norm_path)#[:100]

        self.augment = augment
        self.repeat = repeat
        self.train = train
        # use_mask=True (TOFDSR): GT has invalid pixels, loss and RMSE must skip them.
        # use_mask=False (RGB-D-D): masks are returned as all ones, i.e. behaviour is unchanged.
        self.use_mask = use_mask

        self.base_len = self.depths.shape[0]

    def augment_data(self, img, gt, gt_norm, mask, lr):
        if random.random() < 0.5:
            # horizontal flip
            img = TF.hflip(img)
            gt = TF.hflip(gt)
            gt_norm = TF.hflip(gt_norm)
            mask = TF.hflip(mask)
            lr = TF.hflip(lr)
        if random.random() < 0.5:
            # vertical flip
            img = TF.vflip(img)
            gt = TF.vflip(gt)
            gt_norm = TF.vflip(gt_norm)
            mask = TF.vflip(mask)
            lr = TF.vflip(lr)
        return img, gt, gt_norm, mask, lr

    @staticmethod
    def LRValidMask(mask, size, erode: int = 5):
        # Validity of the bicubic-downsampled GT (clean_lr). An LR pixel is valid only if
        # every HR pixel under the (anti-aliased) bicubic kernel is valid, otherwise holes
        # in the GT bleed into the degradation target.
        m = mask.unsqueeze(0)                                                  # 1, 1, H, W
        m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=2 * erode + 1, stride=1, padding=erode)
        m = F.interpolate(m, size=tuple(size), mode='area')
        return (m > 0.999).float().squeeze(0)

    def Resize(self, tensor, size):
        return TF.resize(tensor, size, interpolation=TF.InterpolationMode.BICUBIC)

    def __len__(self):
        return self.base_len * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.base_len

        depth = torch.from_numpy(self.depths[idx]).float().unsqueeze(0)
        depth_norm = torch.from_numpy(self.depths_norm[idx]).float().unsqueeze(0)
        img = torch.from_numpy(self.images[idx]).float()
        mask = torch.from_numpy(self.masks[idx]).float().unsqueeze(0)
        lr_real = torch.from_numpy(self.depths_lr_norm[idx]).float().unsqueeze(0)

        _, h, w = img.shape

        if self.train and self.augment:
            img, depth, depth_norm, mask, lr_real = self.augment_data(
                img, depth, depth_norm, mask, lr_real)
      
        # minmax stores [max, min] of the LR depth (what is available at inference)
        mn = torch.tensor(self.minmax[idx][1], dtype=torch.float32).view(1, 1, 1)
        mx = torch.tensor(self.minmax[idx][0], dtype=torch.float32).view(1, 1, 1)

        clean_lr = self.Resize(depth, lr_real.shape[-2:])       # raw depth units
        clean_lr_norm = (clean_lr - mn) / (mx - mn + 1e-8)       # same min/max as lr_real and gt_norm

        if self.use_mask:
            mask_lr = self.LRValidMask(mask, lr_real.shape[-2:])
        else:
            mask = torch.ones_like(mask)
            mask_lr = torch.ones_like(lr_real)

        return {
            'rgb': img,
            'gt': depth,
            'gt_norm': depth_norm,
            'lr': lr_real,
            'mask': mask,
            'mask_lr': mask_lr,
            'min': mn,
            'max': mx,
            'orig_h': h,
            'orig_w': w,
            'clean_lr': clean_lr,
            'clean_lr_norm': clean_lr_norm
        }