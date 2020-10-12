import torch
from torch import nn
import math

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def safe_log(tensor, eps=1e-16):
    is_zero = torch.le(tensor, eps)
    tensor = torch.where(is_zero, torch.ones_like(tensor), tensor)
    tensor = torch.where(is_zero, torch.zeros_like(tensor), torch.log(tensor))
    return tensor

def geometric_transform(pose_tensor, similarity=False, nonlinear=True, as_matrix=False):
    """Convers paramer tensor into an affine or similarity transform.
    This function is adapted from:
    https://github.com/akosiorek/stacked_capsule_autoencoders/blob/master/capsules/math_ops.py
    Args:
    pose_tensor: [..., 6] tensor.
    similarity: bool.
    nonlinear: bool; applies nonlinearities to pose params if True.
    Returns:
    [..., 2, 3] tensor.
    """

    scale_x, scale_y, theta, shear, trans_x, trans_y = torch.split(pose_tensor, 1, -1)

    if nonlinear:
        scale_x, scale_y = torch.sigmoid(scale_x) + 1e-2, torch.sigmoid(scale_y) + 1e-2
        trans_x, trans_y, shear = torch.tanh(trans_x * 5.), torch.tanh(trans_y * 5.), torch.tanh(shear * 5.)
        theta *= 2. * math.pi
    else:
        scale_x, scale_y = (abs(i) + 1e-2 for i in (scale_x, scale_y))

    c, s = torch.cos(theta), torch.sin(theta)

    if similarity:
        scale = scale_x
        pose = [scale * c, -scale * s, trans_x, scale * s, scale * c, trans_y]

    else:
        pose = [
            scale_x * c + shear * scale_y * s, -scale_x * s + shear * scale_y * c,
            trans_x, scale_y * s, scale_y * c, trans_y
        ]

    pose = torch.cat(pose, -1)

    # convert to a matrix
    shape = list(pose.shape[:-1])
    shape += [2, 3]
    pose = torch.reshape(pose, shape)
    if as_matrix:
        zeros = torch.zeros_like(pose[:, :, :, :1, :1])
        last = torch.cat([zeros, zeros, zeros+1], -1)
        pose = torch.cat([pose, last], dim=-2)
    return pose