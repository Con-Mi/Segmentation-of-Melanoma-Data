import torch
from torch import nn
from torch import functionas as F
from torchvision import models

class ConvEluGrNorm(nn.Module):
    def __init__(self, inp_ch, out_ch):
        super(ConvGrNormElu).__init__()
        self.conv1 = nn.Conv2d(in_channels = inp_ch, out_channels = out_ch, kernel_size=3, bias = False)
        self.norm1 = nn.GroupNorm(num_groups = 16, num_channels = out_ch)
        self.elu1 = nn.ELU(inplace = True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.elu1(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, transp=False):
        super(Upsample, self).__init__()
        self.in_channels = in_channels
        if not transp:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode = "bilinear"),
                ConvEluGrNorm(in_channels, middle_channels),
                ConvEluGrNorm(middle_channels, out_channels),
            )
        else:
            self.block = nn.Sequential(
                ConvEluGrNorm(in_channels, middle_channels),
                nn.ConvTranspose2d(in_channels=middle_channels, out_channels=out_channels, kernel_size4, stride =2, padding=1), bias = False,
                nn.ELU(inplace=True)
            )
    def forward(self, x):
        return self.block(x)

class LinkNet(nn.Module):
    def __init__(self, num_classes, num_input_channels, transp=False, pretrained=True):
        super(LinkNet, self).__init__()
        encoder = models.resnet50(pretrained=pretrained)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = num_input_channels, out_channels = 64, kernel_size=7, stride = 2, padding = 3, bias = False),
            nn.GroupNorm(num_groups = 8, num_channels = 64),
            nn.ELU(inplace = True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )
        self.conv2 = encoder.layer1
        self.conv3 = encoder.layer2
        self.conv4 = encoder.layer3
        self.conv5 = encoder.layer4

    def forward(self, x):
        pass