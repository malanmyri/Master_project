import math
from typing import Union, Sequence
import torch
import torch.nn.functional as F


def determine_direction(dx, dy, num_sectors):
    """
    Determines the directional label for a given dx, dy pair based on the specified number of polar sectors,
    with the first sector centered at 0 degrees (directly to the right).

    Parameters:
    - dx: Change in x-coordinate.
    - dy: Change in y-coordinate.
    - num_sectors: Number of sectors to divide the polar coordinate system into.

    Returns:
    - A vector of length num_sectors with a 1 at the index corresponding to the sector of the dx, dy pair,
      and 0s elsewhere.
    """
    if num_sectors < 1:
        raise ValueError("Number of sectors must be at least 1.")
    
    angle = math.atan2(dy, dx)
    angle_degrees = (math.degrees(angle) + 360) % 360
    degrees_per_sector = 360 / num_sectors
    adjusted_angle_degrees = (angle_degrees + degrees_per_sector / 2) % 360
    sector = int(adjusted_angle_degrees // degrees_per_sector)
    
    direction_vector = [0] * num_sectors
    direction_vector[sector] = 1

    return direction_vector




__all__ = ['get_pad', 'pad']


def _calc_pad(size: int,
              kernel_size: int = 3,
              stride: int = 1,
              dilation: int = 1):
    pad = (((size + stride - 1) // stride - 1) * stride + kernel_size - size) * dilation
    return pad // 2, pad - pad // 2


def _get_compressed(item: Union[int, Sequence[int]], index: int):
    return item


def get_pad(size: Union[int, Sequence[int]],
            kernel_size: Union[int, Sequence[int]] = 3,
            stride: Union[int, Sequence[int]] = 1,
            dilation: Union[int, Sequence[int]] = 1):
    len_size = 1
    pad = ()
    for i in range(len_size):
        pad = _calc_pad(size=_get_compressed(size, i),
                        kernel_size=_get_compressed(kernel_size, i),
                        stride=_get_compressed(stride, i),
                        dilation=_get_compressed(dilation, i)) + pad
    return pad


def pad(x: torch.Tensor,
        size: Union[int, Sequence[int]],
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        dilation: Union[int, Sequence[int]] = 1):
    return F.pad(x, get_pad(size, kernel_size, stride, dilation))