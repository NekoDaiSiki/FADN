import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil

def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'Resnet':
            return ResBlock(channel_in, channel_out)

        else:
            return None

    return constructor



class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out

