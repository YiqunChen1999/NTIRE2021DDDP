
r"""
Author:
    Yiqun Chen
Docs:
    Build model from configurations.
"""

import os, sys
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
from torch import nn
import torch.nn.functional as F

from utils import utils
from .encoder import _ENCODER
from .decoder import _DECODER
from .modules import _BOTTLENECK


class Model(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super(Model, self).__init__()
        self.cfg = cfg
        self._build_model()

    def _build_model(self):
        self.encoder = _ENCODER[self.cfg.MODEL.ENCODER](self.cfg)
        self.decoder = _DECODER[self.cfg.MODEL.DECODER](self.cfg)
        b_kwargs = dict(in_channels=512, out_channels=1024, drop_rate=0.4)
        self.bottleneck = _BOTTLENECK[self.cfg.MODEL.BOTTLENECK](self.cfg, **b_kwargs)
        
    def forward(self, data, *args, **kwargs):
        enc_1, enc_2, enc_3, enc_4, bottleneck = self.encoder(data)
        bottleneck = self.bottleneck(bottleneck)
        out = self.decoder((enc_1, enc_2, enc_3, enc_4, bottleneck))
        return out


def build_model(cfg, logger=None):
    with utils.log_info(msg="Build model from configurations.", level="INFO", state=True, logger=logger):
        model = Model(cfg)
    return model