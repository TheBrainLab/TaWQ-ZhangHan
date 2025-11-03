import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate, base
from typing import Callable
import math
from abc import abstractmethod


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha
        return grad_x, None


class Sigmoid(nn.Module):
    def __init__(self, alpha=4.0):
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def sign_function(x, alpha):
        return sigmoid.apply(x, alpha)

    def forward(self, x: torch.Tensor):
        return self.sign_function(x, self.alpha)


class BaseNode(base.MemoryModule):
    def __init__(self, ):
        super().__init__()
        self.register_memory('v', 0.)
        self.sign_sig1 = Sigmoid(alpha=4.0)
        self.sign_sig2 = Sigmoid(alpha=4.0)
        self.vth = 0.25
        self.tau = 2.0

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x - self.v) / self.tau

    def neuronal_fire(self):
        return (self.sign_sig1(self.v - self.vth) + self.sign_sig2(self.v + self.vth)) / 2.0

    def neuronal_reset(self, spike):
        spike_d = spike.detach()
        self.v = (1. - torch.abs(spike_d)) * self.v

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

class TaWQ(BaseNode):
    def __init__(self, ):
        super().__init__()
        self.register_memory('v_seq', None)
        self.register_memory('spike_seq', None)

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq


class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) and len(kernel_size) == 2:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1]))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.tawq = TaWQ()

    def forward(self, x: torch.Tensor): # x.shape = T, B, Ci, H, W  weight.shape = T, Co, Ci, kH, kW
        weight_sign = (self.weight - self.weight.mean()) / self.weight.std()
        weight_sign = weight_sign.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
        weight_sign = self.tawq(weight_sign)
        print(torch.abs(weight_sign).mean())
        weight_sign = weight_sign / (torch.abs(weight_sign).mean(dim=(2, 3, 4), keepdim=True) + 1e-6)
        temp = []
        for t in range(x.shape[0]):
            temp.append(F.conv2d(x[t], weight_sign[t], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=None))
        x = torch.stack(temp, 0)
        return x # x.shape = T, B, Co, H', W'


class QConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.tawq = TaWQ()

    def forward(self, x: torch.Tensor):
        weight_sign = (self.weight - self.weight.mean()) / self.weight.std()
        weight_sign = weight_sign.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        weight_sign = self.tawq(weight_sign)
        print(torch.abs(weight_sign).mean())
        weight_sign = weight_sign / (torch.abs(weight_sign).mean(dim=(2, 3), keepdim=True) + 1e-6)
        temp = []
        for t in range(x.shape[0]):
            temp.append(F.conv1d(x[t], weight_sign[t], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, bias=None))
        x = torch.stack(temp, 0)
        return x # x.shape = T, B, Co, N

