import math
from math import sqrt
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_blur2d(input: torch.Tensor,
                    kernel_size: Tuple[int, int],
                    sigma: Tuple[float, float],
                    border_type: str = 'reflect',
                    separable: bool = True) -> torch.Tensor:
    if separable:
        kernel_x: torch.Tensor = get_gaussian_kernel1d(kernel_size[1], sigma[1])
        kernel_y: torch.Tensor = get_gaussian_kernel1d(kernel_size[0], sigma[0])
        out = filter2d_separable(input, kernel_x[None], kernel_y[None], border_type)
    else:
        kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, sigma)
        out = filter2d(input, kernel[None], border_type)
    return out

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    device, dtype = None, None
    if isinstance(sigma, torch.Tensor):
        device, dtype = sigma.device, sigma.dtype
    x = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def get_gaussian_kernel1d(kernel_size: int, sigma: float, force_even: bool = False) -> torch.Tensor:

    if not isinstance(kernel_size, int) or ((kernel_size % 2 == 0) and not force_even) or (kernel_size <= 0):
        raise TypeError("kernel_size must be an odd positive integer. " "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d

def get_gaussian_kernel2d(
    kernel_size: Tuple[int, int], sigma: Tuple[float, float], force_even: bool = False
) -> torch.Tensor:
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(f"kernel_size must be a tuple of length two. Got {kernel_size}")
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(f"sigma must be a tuple of length two. Got {sigma}")
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
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


def filter2d_separable(input: torch.Tensor,
                       kernel_x: torch.Tensor,
                       kernel_y: torch.Tensor,
                       border_type: str = 'reflect',
                       normalized: bool = False,
                       padding: str = 'same') -> torch.Tensor:
    out_x = filter2d(input, kernel_x.unsqueeze(0), border_type, normalized, padding)
    out = filter2d(out_x, kernel_y.unsqueeze(-1), border_type, normalized, padding)
    return out

def unsharp_mask(
    input: torch.Tensor, kernel_size: Tuple[int, int], sigma: Tuple[float, float], border_type: str = 'reflect'
) -> torch.Tensor:
    data_blur: torch.Tensor = gaussian_blur2d(input, kernel_size, sigma, border_type)
    data_sharpened: torch.Tensor = input + (input - data_blur)
    return data_sharpened


class UnsharpMask(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int], sigma: Tuple[float, float], border_type: str = 'reflect') -> None:
        super().__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.border_type = border_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return unsharp_mask(input, self.kernel_size, self.sigma, self.border_type)

#unsharp_mask(input, (3, 3), (1.5, 1.5))