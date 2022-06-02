import torch
import torch.nn as nn
import torch.nn.functional as F

from .ResidualAttentionNetwork.attention_module import AttentionModule_stage2_cifar

__all__ = ['NormalAttMod', 'RanAttMod', 'ResidualAttMod']


class NormalAttMod(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(NormalAttMod, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = out_channels
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                      padding=1)  # extracting feature map from backbone
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x) + 1
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]),
                             mode='nearest')
        # mask's shape is adjusted according to the processed input
        # in this way, the input is not determined
        return x * mask


class RanAttMod(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RanAttMod, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = out_channels
        self.conv_extract = AttentionModule_stage2_cifar(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x) + 1
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]),
                             mode='nearest')
        return x * mask

class ResidualAttMod(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualAttMod, self).__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = out_channels
        self.conv_extract = BasicBlock(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x) + 1
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]),
                             mode='nearest')
        # mask's shape is adjusted according to the processed input
        # in this way, the input is not determined
        return x * mask

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
