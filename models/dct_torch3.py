import math
from math import sqrt
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sobel_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_sobel_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])

def get_sobel_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])

def get_diff_kernel2d() -> torch.Tensor:
    kernel_x: torch.Tensor = get_diff_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])

def get_diff_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3."""
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]])

def get_spatial_gradient_kernel2d(mode: str, order: int) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError(
            "mode should be either sobel\
                         or diff. Got {}".format(
                mode
            )
        )
    if order not in [1, 2]:
        raise TypeError(
            "order should be either 1 or 2\
                         Got {}".format(
                order
            )
        )
    if mode == 'sobel' and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError("")
    return kernel

def get_sobel_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = get_sobel_kernel_5x5_2nd_order()
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = _get_sobel_kernel_5x5_2nd_order_xy()
    return torch.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order() -> torch.Tensor:
    gxx: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]])
    gyy: torch.Tensor = gxx.transpose(0, 1)
    gxy: torch.Tensor = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]])
    return torch.stack([gxx, gxy, gyy])

def get_sobel_kernel_5x5_2nd_order() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ]
    )


def _get_sobel_kernel_5x5_2nd_order_xy() -> torch.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ]
    )

def get_spatial_gradient_kernel3d(mode: str, order: int, device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order scale pyramid gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError(
            "mode should be either sobel\
                         or diff. Got {}".format(
                mode
            )
        )
    if order not in [1, 2]:
        raise TypeError(
            "order should be either 1 or 2\
                         Got {}".format(
                order
            )
        )
    if mode == 'sobel':
        raise NotImplementedError("Sobel kernel for 3d gradient is not implemented yet")
    if mode == 'diff' and order == 1:
        kernel = get_diff_kernel3d(device, dtype)
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device, dtype)
    else:
        raise NotImplementedError("")
    return kernel

def get_diff_kernel3d(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel: torch.Tensor = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)


def get_diff_kernel3d_2nd_order(device=torch.device('cpu'), dtype=torch.float) -> torch.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3."""
    kernel: torch.Tensor = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device,
        dtype=dtype,
    )
    return kernel.unsqueeze(1)

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

def spatial_gradient(input: torch.Tensor, mode: str = 'sobel', order: int = 1, normalized: bool = True) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")
    # allocate kernel
    kernel: torch.Tensor = get_spatial_gradient_kernel2d(mode, order)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.to(input).detach()
    tmp_kernel = tmp_kernel.unsqueeze(1).unsqueeze(1)

    # convolve input tensor with sobel kernel
    kernel_flip: torch.Tensor = tmp_kernel.flip(-3)

    # Pad with "replicate for spatial dims, but with zeros for channel
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels: int = 3 if order == 2 else 2
    padded_inp: torch.Tensor = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, 'replicate')[:, :, None]

    return F.conv3d(padded_inp, kernel_flip, padding=0).view(b, c, out_channels, h, w)


def spatial_gradient3d(input: torch.Tensor, mode: str = 'diff', order: int = 1) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 5:
        raise ValueError(f"Invalid input shape, we expect BxCxDxHxW. Got: {input.shape}")
    b, c, d, h, w = input.shape
    dev = input.device
    dtype = input.dtype
    if (mode == 'diff') and (order == 1):
        # we go for the special case implementation due to conv3d bad speed
        x: torch.Tensor = F.pad(input, 6 * [1], 'replicate')
        center = slice(1, -1)
        left = slice(0, -2)
        right = slice(2, None)
        out = torch.empty(b, c, 3, d, h, w, device=dev, dtype=dtype)
        out[..., 0, :, :, :] = (x[..., center, center, right] - x[..., center, center, left])
        out[..., 1, :, :, :] = (x[..., center, right, center] - x[..., center, left, center])
        out[..., 2, :, :, :] = (x[..., right, center, center] - x[..., left, center, center])
        out = 0.5 * out
    else:
        # prepare kernel
        # allocate kernel
        kernel: torch.Tensor = get_spatial_gradient_kernel3d(mode, order)

        tmp_kernel: torch.Tensor = kernel.to(input).detach()
        tmp_kernel = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # convolve input tensor with grad kernel
        kernel_flip: torch.Tensor = tmp_kernel.flip(-3)

        # Pad with "replicate for spatial dims, but with zeros for channel
        spatial_pad = [kernel.size(2) // 2,
                       kernel.size(2) // 2,
                       kernel.size(3) // 2,
                       kernel.size(3) // 2,
                       kernel.size(4) // 2,
                       kernel.size(4) // 2]
        out_ch: int = 6 if order == 2 else 3
        out = F.conv3d(F.pad(input, spatial_pad, 'replicate'),
                       kernel_flip,
                       padding=0,
                       groups=c).view(b, c, out_ch, d, h, w)
    return out


def sobel(input: torch.Tensor, normalized: bool = True, eps: float = 1e-6) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    # comput the x/y gradients
    edges: torch.Tensor = spatial_gradient(input, normalized=normalized)

    # unpack the edges
    gx: torch.Tensor = edges[:, :, 0]
    gy: torch.Tensor = edges[:, :, 1]

    # compute gradient maginitude
    magnitude: torch.Tensor = torch.sqrt(gx * gx + gy * gy + eps)

    return magnitude


class SpatialGradient(nn.Module):
    def __init__(self, mode: str = 'sobel', order: int = 1, normalized: bool = True) -> None:
        super().__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + '('
            'order=' + str(self.order) + ', ' + 'normalized=' + str(self.normalized) + ', ' + 'mode=' + self.mode + ')'
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return spatial_gradient(input, self.mode, self.order, self.normalized)


class SpatialGradient3d(nn.Module):
    def __init__(self, mode: str = 'diff', order: int = 1) -> None:
        super().__init__()
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel3d(mode, order)
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' 'order=' + str(self.order) + ', ' + 'mode=' + self.mode + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        return spatial_gradient3d(input, self.mode, self.order)


class Sobel(nn.Module):
    def __init__(self, normalized: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized: bool = normalized
        self.eps: float = eps

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' 'normalized=' + str(self.normalized) + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return sobel(input, self.normalized, self.eps)

