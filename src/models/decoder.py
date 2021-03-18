
r"""
Author:
    Yiqun Chen
Docs:
    Decoder classes.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
from collections import OrderedDict
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from .modules import *

_DECODER = {}

def add_decoder(decoder):
    _DECODER[decoder.__name__] = decoder
    return decoder


@add_decoder
class DPDDecoder(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(DPDDecoder, self).__init__()
        self.cfg = cfg
        self._build()

    def _build(self):
        self.block_1_1, self.block_1_2 = self._build_block(2, 1024, 512)
        self.block_2_1, self.block_2_2 = self._build_block(2, 512, 256)
        self.block_3_1, self.block_3_2 = self._build_block(2, 256, 128)
        self.block_4_1, self.block_4_2 = self._build_block(2, 128, 64)
        self.out_block = nn.Sequential(
            nn.Conv2d(64, 3, 3, stride=1, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(3, 3, 1, stride=1), 
            nn.Sigmoid(), 
        )

    def _build_block(self, num_conv, in_channels, out_channels):
        block_1 = nn.Sequential(OrderedDict([
            ("upsampling", nn.UpsamplingNearest2d(scale_factor=2)), 
            ("conv", nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
        ]))
        layer_list = []
        for idx in range(num_conv):
            layer_list.append(
                ("conv_"+str(idx), nn.Conv2d(out_channels*2 if idx == 0 else out_channels, out_channels, 3, stride=1, padding=1))
            )
            layer_list.append(
                ("relu_"+str(idx), nn.ReLU())
            )
        block_2 = nn.Sequential(OrderedDict(layer_list))
        return block_1, block_2

    def forward(self, inp, *args, **kwargs):
        enc_1, enc_2, enc_3, enc_4, bottleneck = inp
        
        # decoder block 1
        out = self.block_1_1(bottleneck)
        out = torch.cat([out, enc_4], dim=1)
        out = self.block_1_2(out)

        # decoder block 2
        out = self.block_2_1(out)
        out = torch.cat([out, enc_3], dim=1)
        out = self.block_2_2(out)

        # decoder block 3
        out = self.block_3_1(out)
        out = torch.cat([out, enc_2], dim=1)
        out = self.block_3_2(out)

        # decoder block 4
        out = self.block_4_1(out)
        out = torch.cat([out, enc_1], dim=1)
        out = self.block_4_2(out)
        
        out = self.out_block(out)
        return out



if __name__ == "__main__":
    print(_DECODER)
    model = _DECODER["UNetDecoder"](None)
    print(_DECODER)
