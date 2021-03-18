
r"""
Author:
    Yiqun Chen
Docs:
    Utilities, should not call other custom modules.
"""

import os, sys, copy, functools, time, contextlib, math
import torch, torchvision
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

@contextlib.contextmanager
def log_info(msg="", level="INFO", state=False, logger=None):
    log = print if logger is None else logger.log_info
    _state = "[{:<8}]".format("RUNNING") if state else ""
    log("[{:<20}] [{:<8}] {} {}".format(time.asctime(), level, _state, msg))
    yield
    if state:
        _state = "[{:<8}]".format("DONE") if state else ""
        log("[{:<20}] [{:<8}] {} {}".format(time.asctime(), level, _state, msg))


def inference(model, data, device):
    """
    Info:
        Inference once, without calculate any loss.
    Args:
        - model (nn.Module):
        - data (dict): necessary keys: "l_view", "r_view"
        - device (torch.device)
    Returns:
        - out (Tensor): predicted.
    """
    def _inference_V1(model, data, device):
        l_view, r_view = data["l_view"], data["r_view"]
        assert len(l_view.shape) == len(r_view.shape) == 4, "Incorrect shape."
        inp = torch.cat([l_view, r_view], dim=1)
        inp = inp.to(device)
        out = model(inp)
        return out, 

    def _inference_V2(model, data, device):
        l_view, r_view = data["l_view"], data["r_view"]
        assert len(l_view.shape) == len(r_view.shape) == 4, "Incorrect shape."
        inp = torch.cat([l_view, r_view], dim=1)
        inp = inp.to(device)
        out, feats = model(inp)
        return out, feats

    def _inference_V3(model, data, device):
        # Serve for multi-resolution model.
        l_view, r_view = data["l_view"], data["r_view"]
        assert len(l_view.shape) == len(r_view.shape) == 4, "Incorrect shape."
        inp = torch.cat([l_view, r_view], dim=1)
        inp = inp.to(device)
        out = model(inp)
        return out[0], out[1: ]

    def _inference_V4(model, data, device):
        l_view, r_view = data["l_view"], data["r_view"]
        assert len(l_view.shape) == len(r_view.shape) == 4, "Incorrect shape."
        inp = torch.cat([l_view, r_view], dim=0)
        inp = inp.to(device)
        out = model(inp)
        return out, 

    return _inference_V1(model, data, device) 


def save_image(output, mean, norm, path2file):
    """
    Info:
        Save output to specific path.
    Args:
        - output (Tensor | ndarray): takes value from range [0, 1].
        - mean (float):
        - norm (float): 
        - path2file (str | os.PathLike):
    Returns:
        - (bool): indicate succeed or not.
    """
    if isinstance(output, torch.Tensor):
        output = output.numpy()
    output = ((output.transpose((1, 2, 0)) * norm) + mean).astype(np.uint16)
    try:
        cv2.imwrite(path2file, output)
        return True
    except:
        return False


def set_device(model: torch.nn.Module, gpu_list: list, logger=None):
    with log_info(msg="Set device for model.", level="INFO", state=True, logger=logger):
        if not torch.cuda.is_available():
            with log_info(msg="CUDA is not available, using CPU instead.", level="WARNING", state=False, logger=logger):
                device = torch.device("cpu")
        if len(gpu_list) == 0:
            with log_info(msg="Use CPU.", level="INFO", state=False, logger=logger):
                device = torch.device("cpu")
        elif len(gpu_list) == 1:
            with log_info(msg="Use GPU {}.".format(gpu_list[0]), level="INFO", state=False, logger=logger):
                device = torch.device("cuda:{}".format(gpu_list[0]))
                model = model.to(device)
        elif len(gpu_list) > 1:
            raise NotImplementedError("Multi-GPU mode is not implemented yet.")
    return model, device


if __name__ == "__main__":
    log_info(msg="DEBUG MESSAGE", level="DEBUG", state=False, logger=None)
    