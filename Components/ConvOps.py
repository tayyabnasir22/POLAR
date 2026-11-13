import torch.nn as nn
import torch.nn.init as init

class ConvOps:
    @staticmethod
    def DefaultConv(in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

    @staticmethod
    def ProjectionConv(in_channels, out_channels, scale, up=True):
        kernel_size, stride, padding = {
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2),
            16: (20, 16, 2)
        }[scale]
        if up:
            conv_f = nn.ConvTranspose2d
        else:
            conv_f = nn.Conv2d

        return conv_f(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
    