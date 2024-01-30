import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    ):
        super(ConvBlock, self).__init__()

        self.CONV = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.BNORM = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=False)

        self.MAXPOOL = nn.MaxPool2d(3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x = self.CONV(x)
        x = self.BNORM(x)
        x = self.MAXPOOL(x)

        return x