import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torchvision import models

class FCNDense(nn.Module):
    def __init__(self, pretrained = False, num_init_features = 64, growth_rate = 32, bot_neck = 4, n_class = 1):
        super(FCNDense, self).__init__()
        self.encoder = models.densenet121(pretrained = pretrained).features
        # OBSOLETE and unnecessary
        """
        self.low_conv = nn.Sequential(OrderedDict([
                ("conv0", nn.Conv2d(in_channels = 3, out_channels = num_init_features, kernel_size = 7, stride = 2, padding = 3, bias=False)),
                ("group_norm0", nn.GroupNorm(num_groups = num_init_features//4, num_channels = num_init_features)),
                ("elu", nn.ELU(inplace = True)),
                ("max_pool", nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        ]))
        self.denseblock_1 = self.encoder[4]
        self.denseblock_2 = self.encoder[6]
        self.denseblock_3 = self.encoder[8]
        self.denseblock_4 = self.encoder[10]
       
        self.transition_1 = self.encoder[5]
        self.transition_2 = self.encoder[7]
        self.transition_3 = self.encoder[9]
        """
        # fc1
        self.fc1 = nn.Conv2d(1024, 4096, kernel_size = 1, padding = 0, bias = False)
        self.elu1 = nn.ELU(inplace=True)
        self.gn1 = nn.GroupNorm(num_groups=16, num_channels=4096)
        
        # fc2
        self.fc2 = nn.Conv2d(4096, 4096, kernel_size = 1, padding = 0, bias = False)
        self.elu2 = nn.ELU(inplace=True)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=4096)

        self.score_fr = nn.Conv2d(4096, n_class, kernel_size = 1, padding = 0, bias = False)
        self.elu3 = nn.ELU(inplace=True)

        self.upscore = nn.ConvTranspose2d(n_class, n_class, kernel_size = 32, stride=32, padding = 0, bias=False)
        # TRY THIS: F.interpolete(out, size = x.size()[2], mode = "bilinear')

    def forward(self, x):
        out = self.encoder(x)
        """
        out = self.denseblock_1(out)
        out = self.transition_1(out)

        out = self.denseblock_2(out)
        out = self.transition_2(out)

        out = self.denseblock_3(out)
        out = self.transition_3(out)

        out = self.denseblock_4(out)
        """
        out = self.fc1(out)
        out = self.elu1(out)
        out = self.gn1(out)

        out = self.fc2(out)
        out = self.elu2(out)
        out = self.gn2(out)

        out = self.score_fr(out)
        out = self.elu3(out)
        out = self.upscore(out)

        return out


def fcn_model(is_pretrained=False):
    return FCNDense(is_pretrained)
