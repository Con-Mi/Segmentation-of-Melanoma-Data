# https://github.com/ternaus/TernausNet/blob/master/unet_models.py

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torchvision import models

class _UpsampleBlockTransposed(nn.Sequential):
    def __init__(self, in_channels, out_channels, bot_neck):
        super(_UpsampleBlock, self).__init__()
        self.in_channels = in_channels
        self.add_module("group_norm1", nn.GroupNorm(num_groups=in_channels//16, num_channels=in_channels))
        self.add_module("elu1", nn.ELU(inplace = True))
        self.add_module("conv", nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=1, stride = 1, padding = 0, bias = False)),
        self.add_module("group_norm2", nn.GroupNorm(num_groups=(bot_neck*growth_rate)//2 , num_channels= bot_neck*growth_rate)),
        self.add_module("elu2", nn.ELU(inplace=True)),
        self.add_module("transp_conv", nn.Convtranspose2d(in_channels = out_channels, out_channels = out_channels*bot_neck, kernel_size=3, stride = 1, padding = 1, bias = False))

    def forward(self, x):
        new_features = super(_UpsampleBlock, self).forward(x)
        return new_features 

class _UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_UpsampleBlock, self).__init__()
        self.in_channels = in_channels
        self.add_module("group_norm", nn.GroupNorm(num_groups=in_channels//16, num_channels=in_channels))
        self.add_module("elu", nn.ELU(inplace = True))
        self.add_module("conv", nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, stride = 1, padding = 1, bias = False))
    def forward(self, x):
        new_features = super(_UpsampleBlock, self).forward(x)
        return nn.functional.interpolate(new_features, scale_factor = 2, mode = "nearest") 


class _SmoothBlock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, bot_neck, growth_rate):
        super(_SmoothBlock).__init_()
        self.add_module("group_norm", nn.GroupNorm(num_groups=num_input_features//36, num_channels=num_input_features))
        self.add_module("elu", nn.ELU(inplace = True))
        self.add_module("conv1", nn.Conv2d(in_channels=num_input_features, out_channels = bot_neck*growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module("elu", nn.ELU(inplace = True))
    def forward(self, x):
        out = super(_SmoothBlock).forward(x)
        return out

class PyramidalDenseNet(nn.Module):
    def __init__(self, pretrained = False, growth_rate, bot_neck):
        super(PyramidalDenseNet, self).__init__()
        self.encoder = models.densenet121(pretrained = pretrained).features
        self.low_conv = nn.Sequential(OrderedDict[
            self.add_module("conv1", self.encoder[0]), 
            self.add_module("batch_norm", self.encoder[1]),
            self.add_module("relu", self.encoder[2]),
            self.add_module("max_pool", self.encoder[3])
        ])
        self.denseblock_1 = self.encoder[4]
        self.denseblock_2 = self.encoder[6]
        self.denseblock_3 = self.encoder[8]
        self.denseblock_4 = self.encoder[10]
       
        self.transition1 = self.encoder[5]
        self.transition2 = self.encoder[7]
        self.transition3 = self.encoder[9]
        
        self.smooth_block1 = self._SmoothBlock()
        self.smooth_block2 = self._SmoothBlock()
        self.smooth_block3 = self._SmoothBlock()
        self.smooth_block4 = self._SmoothBlock()

        self.upsample1 = _UpsampleBlock()
        self.upsample2 = _UpsampleBlock()
        self.upsample3 = _UpsampleBlock()
        self.upsample4 = _UpsampleBlock()

    def forward(self, x):
        out = self.low_conv(x)

        blck1 = self.denseblock_1(out)
        blck1 = self.transition_1(blck1)

        blck2 = self.denseblock_2(blck1)
        blck2 = self.transition_2(blck2)

        blck3 = self.denseblock_3(blck2)
        blck3 = self.transition_3(blck3)

        out = self.denseblock_4(blck3)

