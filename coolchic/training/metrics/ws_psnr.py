import math
import torch
from torch import Tensor
from typing import Union, Dict
from coolchic.io.format.yuv import DictTensorYUV

def compute_ws_mse(img1: Tensor, img2: Tensor) -> torch.Tensor:
    if img1.dim() == 4:
        img1 = img1.squeeze(0)  # (C, H, W)
    if img2.dim() == 4:
        img2 = img2.squeeze(0)
    
    C, height, width = img1.shape
    dtype = img1.dtype
    device = img1.device

    phis = torch.arange(height + 1, dtype=dtype, device=device) * torch.pi / height
    deltaTheta = 2 * torch.pi / width

    column = deltaTheta * (-torch.cos(phis[1:]) + torch.cos(phis[:-1]))
    w = column.view(1, height, 1) # broadcasting covers C and W

    wmse_per_channel = ((img1 - img2) ** 2 * w).sum(dim=(1, 2)) / (4 * torch.pi)
    return wmse_per_channel

def compute_ws_psnr(img1: Union[Tensor, DictTensorYUV], img2: Union[Tensor, DictTensorYUV], max_val: float = 1.0) -> float:
    if isinstance(img1, Tensor):
        wmse_per_channel = compute_ws_mse(img1, img2)
    else:
        wmse_y = compute_ws_mse(img1.get("y"), img2.get("y"))
        wmse_u = compute_ws_mse(img1.get("u"), img2.get("u"))
        wmse_v = compute_ws_mse(img1.get("v"), img2.get("v"))
        wmse_per_channel = torch.cat([wmse_y, wmse_u, wmse_v], dim=0)

    # Evita log(0)
    wmse_per_channel = torch.where(wmse_per_channel == 0, torch.tensor(1e-10, device=wmse_per_channel.device), wmse_per_channel)
    wspsnr_per_channel = 10 * torch.log10(max_val**2 / wmse_per_channel)
    
    return float(wspsnr_per_channel.mean().item())
