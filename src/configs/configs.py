
r"""
Author:
    Yiqun Chen
Docs:
    Configurations, should not call other custom modules.
"""

import os, sys, copy, argparse
from attribdict import AttribDict as Dict

configs = Dict()
cfg = configs

parser = argparse.ArgumentParser()
parser.add_argument("id", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("batch_size", type=int)
parser.add_argument("resume", default="false", choices=["true", "false"], type=str)
parser.add_argument("gpu", type=str)
args = parser.parse_args()

# ================================ 
# GENERAL
# ================================ 
cfg.GENERAL.ROOT                                =   os.path.join(os.getcwd(), ".")
cfg.GENERAL.ID                                  =   "{}".format(args.id)
cfg.GENERAL.BATCH_SIZE                          =   args.batch_size
cfg.GENERAL.RESUME                              =   True if args.resume == "true" else False
cfg.GENERAL.GPU                                 =   eval(args.gpu)

# ================================ 
# MODEL
# ================================ 
cfg.MODEL.ENCODER                               =   "DPDEncoder" # ["DPDEncoder", "DPDEncoderV2"]
cfg.MODEL.DECODER                               =   "DPDDecoder" # ["DPDDecoder", "DPDDecoderV2"]
cfg.MODEL.CKPT_DIR                              =   os.path.join(cfg.GENERAL.ROOT, "checkpoints", cfg.GENERAL.ID)
cfg.MODEL.PATH2CKPT                             =   os.path.join(cfg.MODEL.CKPT_DIR, "checkpoints.pth")
cfg.MODEL.BOTTLENECK                            =   "DPDBottleneck"

# ================================ 
# DATA
# ================================ 
cfg.DATA.DIR                                    =   {
    "DualPixelNTIRE2021": "/home/yqchen/Data/DualPixelNTIRE2021", 
}
cfg.DATA.NUMWORKERS                             =   4 
cfg.DATA.DATASET                                =   args.dataset # "DualPixelCanon"
cfg.DATA.BIT_DEPTH                              =   16 # NOTE DO NOT CHANGE THIS
cfg.DATA.MEAN                                   =   [0, 0, 0]
cfg.DATA.NORM                                   =   [2**cfg.DATA.BIT_DEPTH-1, 2**cfg.DATA.BIT_DEPTH-1, 2**cfg.DATA.BIT_DEPTH-1]

# ================================ 
# SAVE
# ================================ 
cfg.SAVE.DIR                                    =   os.path.join(os.path.join(cfg.GENERAL.ROOT, "results", cfg.GENERAL.ID, cfg.DATA.DATASET))
cfg.SAVE.SAVE                                   =   True


_paths = [
    cfg.MODEL.CKPT_DIR, 
    cfg.SAVE.DIR, 
]
_paths.extend(list(cfg.DATA.DIR.as_dict().values()))

for _path in _paths:
    if not os.path.exists(_path):
        os.makedirs(_path)

