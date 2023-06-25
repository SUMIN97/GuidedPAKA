import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import math

from detectron2.layers import Conv2d, get_norm

class GuidedPAKA2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, r=4, bias=False,
                 norm=None, activation=None):
        super(GuidedPAKA2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.norm = norm
        self.activation = activation

        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size ** 2, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.unfold = nn.Unfold(kernel_size, padding=padding, stride=stride, dilation=dilation)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)


    
    def forward(self, x, guide_ch, guide_sp):
        filter = guide_ch.unsqueeze(2) + guide_sp.unsqueeze(1)  # [0,2]
        filter = 1 + torch.tanh(filter)

        b, c, h, w = x.shape
        x = F.conv3d(self.unfold(x).view(b, c, self.kernel_size ** 2, h // self.stride, w // self.stride) * filter,
                     self.weight, self.bias).squeeze(2)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x