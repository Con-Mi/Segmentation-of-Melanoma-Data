from torch import nn
from torch.nn import functional as F

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bot_neck, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("group_norm1", nn.GroupNorm(num_groups=num_input_features // ( (bot_neck * growth_rate) // 2 ), num_channels=num_input_features)),
        self.add_module("elu1", nn.ELU(inplace=True)),
        self.add_module("conv1", nn.Conv2d(in_channels=num_input_features, out_channels = bot_neck*growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module("group_norm2", nn.GroupNorm(num_groups=(bot_neck*growth_rate)//2 , num_channels= bot_neck*growth_rate)),
        self.add_module("elu2", nn.ELU(inplace=True)),
        self.add_module("conv2", nn.Conv2d(in_channels=bot_neck*growth_rate, out_channels=growth_rate, kernel_size = 3, stride = 1, padding=1, bias = False)),
        self.drop_rate = drop_rate
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p = self.drop_rate, training = self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, bot_neck, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i*growth_rate, growth_rate, bot_neck, drop_rate)
            self.add_module("denselayer%d" % (i+1), layer)
    
class _TransitionLayer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionLayer, self).__init__()
        self.add_module("group_norm1", nn.GroupNorm(num_groups = num_input_features//8, num_channels = num_input_features))
        self.add_module("elu1", nn.ELU(inplace = True))
        self.add_module("conv", nn.Conv2d(in_channels = num_input_features, out_channels = num_output_features, kernel_size = 1, stide = 1, bias = False))
        self.add_module("avg_pool", nn.AvgPool2d(kernel_size = 2, stride = 2))

class PyramidalDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=[6, 12, 24, 16], num_init_features=64, bot_neck = 4, drop_rate=0):
        super(PyramidalDenseNet, self).__init__()
        