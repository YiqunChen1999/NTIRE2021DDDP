
r"""
Author:
    Yiqun Chen
Docs:
    Main functition to run program.
"""

import sys, os, copy
import torch, torchvision

from configs.configs import cfg
from utils import utils
from models import model_builder
from data import data_loader
from generate import generate

def main():
    # Build model.
    model = model_builder.build_model(cfg=cfg)

    # Read checkpoint.
    ckpt = torch.load(cfg.MODEL.PATH2CKPT) if cfg.GENERAL.RESUME else {}

    if cfg.GENERAL.RESUME:
        model.load_state_dict(ckpt["model"])
    # Set device.
    model, device = utils.set_device(model, cfg.GENERAL.GPU)
    
    try:
        test_data_loader = data_loader.build_data_loader(cfg, cfg.DATA.DATASET, "test")
    except:
        raise ValueError("Failed to build data loader for test.")
    
    generate(epoch=epoch, cfg=cfg, model=model, data_loader=test_data_loader, device=device)


if __name__ == "__main__":
    main()


