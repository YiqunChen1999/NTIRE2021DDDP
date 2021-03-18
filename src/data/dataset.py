
"""
Author:
    Yiqun Chen
Docs:
    Dataset classes.
"""

import os, sys, random
sys.path.append(os.path.join(sys.path[0], ".."))
sys.path.append(os.path.join(os.getcwd(), "src"))
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
# PIL.Image seems cannot read 16-bit .png file.
# from PIL import Image
import cv2, copy
from tqdm import tqdm
import numpy as np

from utils import utils

_DATASET = {}

def add_dataset(dataset):
    _DATASET[dataset.__name__] = dataset
    return dataset


@add_dataset
class DualPixelNTIRE2021(torch.utils.data.Dataset):
    """
    Info:
        Dataset of NTIRE2021, folders and files structure:
        DualPixelNTIRE2021:
            - test:
                - l_view:
                    00000.png
                    00001.png
                    ...
                - r_view:
                    00000.png
                    00001.png
                    ...
    """
    def __init__(self, cfg, split, *args, **kwargs):
        super(DualPixelNTIRE2021, self).__init__()
        self.cfg = cfg
        self.split = split
        self._build()

    def _build(self):
        self.data = []
        self.t_img_list = []
        self.l_img_list = []
        self.r_img_list = []
        self.path2imgs = os.path.join(self.cfg.DATA.DIR[self.cfg.DATA.DATASET], self.split)

        t_img_list = os.listdir(os.path.join(self.path2imgs, "target"))
        r_img_list = os.listdir(os.path.join(self.path2imgs, "r_view"))
        l_img_list = os.listdir(os.path.join(self.path2imgs, "l_view"))

        t_img_dict = {img.split(".")[0]: img for img in t_img_list}
        l_img_dict = {img.split(".")[0]: img for img in l_img_list}
        r_img_dict = {img.split(".")[0]: img for img in r_img_list}

        pbar = tqdm(total=len(t_img_list), dynamic_ncols=True)
        for idx, img_idx in enumerate(l_img_dict.keys()):
            if img_idx == "":
                continue
            if img_idx in l_img_dict.keys() and img_idx in r_img_dict.keys():
                item = {
                    "img_idx": img_idx, 
                    "l_view": l_img_dict[img_idx], 
                    "r_view": r_img_dict[img_idx], 
                }
                self.data.append(item)

            pbar.update()
        pbar.close()

    def __getitem__(self, idx):
        data = {}

        l_img = cv2.imread(os.path.join(self.path2imgs, "l_view", self.data[idx]["l_view"]), -1)
        r_img = cv2.imread(os.path.join(self.path2imgs, "r_view", self.data[idx]["r_view"]), -1)

        data["img_idx"] = self.data[idx]["img_idx"]

        # Transform: [H, W, C] -> [C, H, W]        
        data["l_view"] = np.transpose((l_img-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)
        data["r_view"] = np.transpose((r_img-np.array(self.cfg.DATA.MEAN))/np.array(self.cfg.DATA.NORM), (2, 0, 1)).astype(np.float32)

        return data
        
    def __len__(self):
        return len(self.data)
        


