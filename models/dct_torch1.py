import math
from math import sqrt
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_laplacian_kernel2d(kernel_size: int) -> torch.Tensor:
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
        raise TypeError(f"ksize must be an odd positive integer. Got {kernel_size}")

    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding

def filter2d(
    input: torch.Tensor, kernel: torch.Tensor, border_type: str = 'reflect', normalized: bool = False,
    padding: str = 'same'
) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

    if border_type not in ['constant', 'reflect', 'replicate', 'circular']:
        raise ValueError(f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}")

    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got {type(padding)}")

    if padding not in ['valid', 'same']:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == 'same':
        padding_shape: List[int] = _compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == 'same':
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    if len(input.size()) < 2:
        raise TypeError(f"input should be at least 2D tensor. Got {input.size()}")
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))

def laplacian(
    input: torch.Tensor, kernel_size: int, border_type: str = 'reflect', normalized: bool = True
) -> torch.Tensor:
    kernel: torch.Tensor = torch.unsqueeze(get_laplacian_kernel2d(kernel_size), dim=0)

    if normalized:
        kernel = normalize_kernel2d(kernel)

    return filter2d(input, kernel, border_type)


class Laplacian(nn.Module):
    def __init__(self, kernel_size: int, border_type: str = 'reflect', normalized: bool = True) -> None:
        super().__init__()
        self.kernel_size: int = kernel_size
        self.border_type: str = border_type
        self.normalized: bool = normalized

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(kernel_size='
            + str(self.kernel_size)
            + ', '
            + 'normalized='
            + str(self.normalized)
            + ', '
            + 'border_type='
            + self.border_type
            + ')'
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return laplacian(input, self.kernel_size, self.border_type, self.normalized)