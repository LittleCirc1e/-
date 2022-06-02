import torch.nn as nn

from .basic_layers import ResidualBlock
from .attention_module import AttentionModule_stage1, AttentionModule_stage2, AttentionModule_stage3, AttentionModule_stage0
from .attention_module import AttentionModule_stage1_cifar, AttentionModule_stage2_cifar, AttentionModule_stage3_cifar

cfgs = {
    "56":[1, 1, 1],
    "92":[1, 2, 3]
}


def make_back_bone(cfg:list):
    in_channel = 3

    layers = []
    layers += nn.Sequential(
        nn.Conv2d(3, 64, 7, 2, 3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )
    layers += nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    layers += ResidualBlock(64, 256)
    for v in range(cfg[0]):
        layers += AttentionModule_stage1(256, 256)
    layers += ResidualBlock(256, 512, 2)
    for v in range(cfg[1]):
        layers += AttentionModule_stage2(512, 512)
    layers += ResidualBlock(512, 1024, 2)
    for v in range(cfg[2]):
        layers += AttentionModule_stage3(1024, 1024)
    layers += ResidualBlock(1024, 2048, 2)
    layers += ResidualBlock(2048, 2048)
    layers += ResidualBlock(2048, 2048)
    layers += nn.Sequential(
        nn.BatchNorm2d(2048),
        nn.ReLU(inplace=True),
        nn.AdaptiveMaxPool2d((1, 1))
    )

    return nn.Sequential(*layers)



