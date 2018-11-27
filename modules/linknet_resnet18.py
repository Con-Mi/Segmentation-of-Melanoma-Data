import torch
from torch import nn
from torch import functional as F
from torchvision import models

class ConvEluGrNorm(nn.Module):
    def __init__(self, inp_ch, out_ch):
        super(ConvEluGrNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = inp_ch, out_channels = out_ch, kernel_size=3, padding = 1, bias = False)
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
                nn.Upsample(scale_factor=2, mode = "nearest"),
                ConvEluGrNorm(in_channels, middle_channels),
                ConvEluGrNorm(middle_channels, out_channels),
            )
        else:
            self.block = nn.Sequential(
                ConvEluGrNorm(in_channels, middle_channels),
                nn.ConvTranspose2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias = False),
                nn.ELU(inplace=True)
            )
    def forward(self, x):
        return self.block(x)

class LinkNet18(nn.Module):
    def __init__(self, num_classes, num_input_channels, num_filters = 32, transp=False, pretrained=True):
        super(LinkNet18, self).__init__()
        encoder = models.resnet18(pretrained=pretrained)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = num_input_channels, out_channels = 64, kernel_size=7, stride = 2, padding = 3, bias = False),
            nn.GroupNorm(num_groups = 8, num_channels = 64),
            nn.ELU(inplace = True),
        )                           #256
        self.conv2 = encoder.layer1 #128
        self.conv3 = encoder.layer2 #64
        self.conv4 = encoder.layer3 #32
        self.conv5 = encoder.layer4 #16

        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) #8

        self.center = Upsample(512, num_filters*8, num_filters*8) #16

        self.dec5 = Upsample(512 + num_filters*8, num_filters*8, num_filters*8) # 32
        self.dec4 = Upsample(256 + num_filters*8, num_filters*8, num_filters*8) # 64
        self.dec3 = Upsample(128 + num_filters*8, num_filters*2, num_filters*2)  # 128
        self.dec2 = Upsample(64 + num_filters*2, num_filters*2, num_filters*2)  # 256
        
        self.dec1 = Upsample(64+num_filters*2, num_filters, num_filters)        # 512

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.pool(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        out = self.pool(conv5)
        
        center = self.center(out)

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


def linknet18_model(num_classes=1, num_input_channels=3):
    return LinkNet18(num_classes, num_input_channels)
