import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips
import torch

def compute_lpips(img1: torch.Tensor,
                  img2: torch.Tensor,
                  net: str = 'alex',
                  use_gpu: bool = False) -> float:
    loss_fn = lpips.LPIPS(net=net)
    if use_gpu and torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
        img1 = img1.cuda()
        img2 = img2.cuda()
    
    with torch.no_grad():
        d = loss_fn(img1, img2)
    return d.item()

def compute_ssim(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError(f"Image size mismatch: {img1.shape} vs {img2.shape}")

    img1 = img1.astype(np.float32) / 255.0 if img1.max() > 1.0 else img1.astype(np.float32)
    img2 = img2.astype(np.float32) / 255.0 if img2.max() > 1.0 else img2.astype(np.float32)

    ssim_value = ssim(img1, img2, channel_axis=2, data_range=1.0)
    return ssim_value

def compute_psnr(img1, img2, max_val=255.0):
    if img1.shape != img2.shape:
        raise ValueError(f"Image size mismatch: {img1.shape} vs {img2.shape}")

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr