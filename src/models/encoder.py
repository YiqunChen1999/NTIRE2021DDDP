
r"""
Author:
    Yiqun Chen
Docs:
    Encoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
from collections import OrderedDict
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import timm
from attribdict import AttribDict
from utils import utils
from .modules import *

_ENCODER = {}

def add_encoder(encoder):
    _ENCODER[encoder.__name__] = encoder
    return encoder


@add_encoder
class DPDEncoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(DPDEncoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.block_1 = self._build_block(2, 6, 64, dropout=0.0)
        self.block_2 = self._build_block(2, 64, 128, dropout=0.0)
        self.block_3 = self._build_block(2, 128, 256, dropout=0.0)
        self.block_4 = self._build_block(2, 256, 512, dropout=0.4)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _build_block(self, num_conv, in_channels, out_channels, dropout=0.0):
        layer_list = []
        for idx in range(num_conv):
            layer_list.append(
                ("conv_"+str(idx), nn.Conv2d(in_channels if idx == 0 else out_channels, out_channels, 3, stride=1, padding=1))
            )
            layer_list.append(("relu_"+str(idx), nn.ReLU()))
        if dropout:
            layer_list.append(
                ("dropout", nn.Dropout(dropout))
            )
        block = nn.Sequential(OrderedDict(layer_list))
        return block

    def forward(self, inp):
        enc_1 = self.block_1(inp)
        
        enc_2 = self.max_pool(enc_1)
        enc_2 = self.block_2(enc_2)

        enc_3 = self.max_pool(enc_2)
        enc_3 = self.block_3(enc_3)

        enc_4 = self.max_pool(enc_3)
        enc_4 = self.block_4(enc_4)

        bottleneck = self.max_pool(enc_4)

        return enc_1, enc_2, enc_3, enc_4, bottleneck



if __name__ == "__main__":
    print(_ENCODER)
