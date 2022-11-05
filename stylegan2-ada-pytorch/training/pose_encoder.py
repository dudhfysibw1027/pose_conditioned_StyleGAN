import math
import random
import functools
import operator
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
#from torch.autograd import Function
#import torch.utils.checkpoint as cp
#from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
#                      build_norm_layer, build_upsample_layer, constant_init,
#                      kaiming_init)
#from mmcv.runner import load_checkpoint
#from mmcv.utils.parrots_wrapper import _BatchNorm
#from mmseg.utils import get_root_logger

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


# Source: text2human
class ShapeAttrEmbedding(nn.Module):

    def __init__(self, dim, out_dim, cls_num_list):
        super(ShapeAttrEmbedding, self).__init__()

        for idx, cls_num in enumerate(cls_num_list):
            setattr(
                self, f'attr_{idx}',
                nn.Sequential(
                    nn.Linear(cls_num, dim), nn.LeakyReLU(),
                    nn.Linear(dim, dim)))
        self.cls_num_list = cls_num_list
        self.attr_num = len(cls_num_list)
        self.fusion = nn.Sequential(
            nn.Linear(dim * self.attr_num, out_dim), nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, attr):
        attr_embedding_list = []
        for idx in range(self.attr_num):
            attr_embed_fc = getattr(self, f'attr_{idx}')
            attr_embedding_list.append(
                attr_embed_fc(
                    F.one_hot(
                        attr[:, idx],
                        num_classes=self.cls_num_list[idx]).to(torch.float32)))
        attr_embedding = torch.cat(attr_embedding_list, dim=1)
        attr_embedding = self.fusion(attr_embedding)

        return attr_embedding


# Source: pose_with_style
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, attr_channel=128, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.attr_channel = attr_channel

        print(in_channel)
        #print(out_channel)
        print(attr_channel)
        
        '''self.fusion = nn.Sequential(
            nn.Linear(in_channel+attr_channel, in_channel)
        )'''

        self.conv1 = ConvLayer(in_channel+128, in_channel+128, 3)
        self.conv2 = ConvLayer(in_channel+128, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel+128, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        print(input.size())
        #input = self.fusion(input)

        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class PoseEncoder(nn.Module):
    def __init__(self, ngf=64, blur_kernel=[1, 3, 3, 1], size=256, attr_dim=128):
        super().__init__()
        self.size = size
        convs = [ConvLayer(1, ngf, 1)]                      # in 1 out 64
        convs.append(ResBlock(ngf, ngf*2, attr_dim, blur_kernel))     # in 64 out 128
        convs.append(ResBlock(ngf*2, ngf*4, attr_dim, blur_kernel))
        convs.append(ResBlock(ngf*4, ngf*8, attr_dim, blur_kernel))
        convs.append(ResBlock(ngf*8, ngf*8, attr_dim, blur_kernel))
        if self.size == 512:
            convs.append(ResBlock(ngf*8, ngf*8, attr_dim, blur_kernel))
        if self.size == 1024:
            convs.append(ResBlock(ngf*8, ngf*8, attr_dim, blur_kernel))
            convs.append(ResBlock(ngf*8, ngf*8, attr_dim, blur_kernel))

        self.convs = nn.Sequential(*convs)

    def forward(self, input, attr_embedding):
        x = input
        b, c = attr_embedding.size()
        for idx, conv in enumerate(self.convs):
            if idx != 0:
                _, _, h, w = x.size()
                #print(type(x))
                #print(type(attr_embedding.view(b, c, 1, 1).expand(b, c, h, w)))
                print(torch.cat(
                        (x,
                        attr_embedding.view(b, c, 1, 1).expand(b, c, h, w)), 
                        dim=1
                    ).size())
                x = conv(
                    torch.cat(
                        (x,
                        attr_embedding.view(b, c, 1, 1).expand(b, c, h, w)), 
                        dim=1
                    )
                )
                
            else:
                x = conv(x)

        #out = self.convs(input)
        return x
