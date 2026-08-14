# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

import threading

import torch
import torch.nn.functional as F

# The base sampling grid only depends on the spatial size, device and dtype,
# but the eager path rebuilt it (arange + meshgrid + stack) on every call.
# flow_warp is called ~3x per frame-step during propagate (~1300x per clip),
# so caching the grid removes ~5 kernel launches + 4 small allocations per
# call while keeping the arithmetic (and thus the numerics) identical.
_GRID_CACHE: dict[tuple, torch.Tensor] = {}
_GRID_CACHE_LOCK = threading.Lock()


def _get_base_grid(h, w, device, dtype) -> torch.Tensor:
    key = (h, w, device, dtype)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        with _GRID_CACHE_LOCK:
            grid = _GRID_CACHE.get(key)
            if grid is None:
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(0, h, device=device, dtype=dtype),
                    torch.arange(0, w, device=device, dtype=dtype),
                    indexing='ij')
                grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
                _GRID_CACHE[key] = grid
    return grid


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    grid = _get_base_grid(h, w, device, x.dtype)
    grid.requires_grad_(False)

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.to(dtype=x.dtype)

    # MPS: skip grid_sample when numel==0 to avoid empty Placeholder
    # (PR #148133 https://github.com/pytorch/pytorch/pull/148133);
    # applies to all padding modes.
    if x.device.type == 'mps' and (x.numel() == 0 or grid_flow.numel() == 0):
        return torch.empty(x.shape, device=x.device, dtype=x.dtype)

    # MPS: border unsupported.
    # Clamp grid to [-1,1] + zeros
    # (issue #125098 https://github.com/pytorch/pytorch/issues/125098#issuecomment-2270384282)
    # to have the same effect as `border`.
    if x.device.type == 'mps' and padding_mode == 'border':
        grid_flow = grid_flow.clamp(-1.0, 1.0)
        return F.grid_sample(x, grid_flow, mode=interpolation, padding_mode='zeros', align_corners=align_corners)

    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output
