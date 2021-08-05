from __future__ import division
import torch
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
from time import time


def l1_loss(gt_depth, depth):
    loss = (gt_depth - torch.squeeze(depth , 1)).abs().mean()
    return loss