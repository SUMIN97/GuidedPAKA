import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math

from detectron2.layers import Conv2d, get_norm


class PAKA2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, r=4, bias=False,
                 norm=None):
        super(PAKA2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size ** 2, 1, 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.conv_c = nn.Sequential(
            Conv2d(in_channels, in_channels // r, kernel_size=1),  # conv1x1(in_channels, in_channels//r, stride),
            nn.ReLU(True),
            Conv2d(in_channels // r, in_channels, kernel_size=1),  # conv1x1(in_channels//r, in_channels),
            # get_norm(norm, in_channels),#nn.BatchNorm2d(in_channels),
        )

        self.conv_d = nn.Sequential(
            Conv2d(in_channels, in_channels // r, kernel_size=3, padding=1, stride=1, dilation=1),
            # conv3x3(in_channels, in_channels//r, stride, dilation=dilation),
            nn.ReLU(True),
            Conv2d(in_channels // r, kernel_size ** 2, kernel_size=1),  # conv1x1(in_channels//r, kernel_size ** 2),
            # get_norm(norm, kernel_size**2),#nn.BatchNorm2d(kernel_size ** 2),
        )

        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride, dilation=dilation)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x):
        b, n, h, w = x.shape
        return F.conv3d(
            self.unfold(x).view(b, n, self.kernel_size ** 2, h // self.stride, w // self.stride) * (
                        1 + torch.tanh(self.conv_d(x).unsqueeze(1) + self.conv_c(x).unsqueeze(2))),
            self.weight, self.bias).squeeze(2)









