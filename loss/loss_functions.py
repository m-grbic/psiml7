from __future__ import division
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
from time import time
from hyperparameters import SMOOTH_L1, SMOOTH_THRESH


def l1_loss(gt_depth, depth):
    # Valid depth
    valid = (gt_depth > 0) & (gt_depth < 10)
    # L1 loss
    loss = (gt_depth[valid] - depth[valid].clamp(min=0, max=10)).abs().mean()
    # L2 loss
    if SMOOTH_L1:
        if loss < SMOOTH_THRESH:
            loss = (gt_depth[valid] - depth[valid].clamp(min=0, max=10)).square().mean()
    return loss


def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        #scaled_map=scaled_map.clamp(1e-3,80)
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss


def weighted_loss(l1, l2, w1, w2):
    return w1*l1 + w2*l2