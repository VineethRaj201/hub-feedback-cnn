"""
corruptions.py

Utilities for applying simple corruptions to MNIST images.
Used to test whether the hub-feedback architecture is more robust
than the baseline under degraded or ambiguous inputs.
"""

import torch
import torchvision.transforms.functional as TF


def add_gaussian_noise(x: torch.Tensor, std: float = 0.3) -> torch.Tensor:
    """
    Add Gaussian noise to an image tensor.
    x: (B, C, H, W) or (C, H, W), values in [0, 1]
    """
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(0.0, 1.0)


def add_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 2.0) -> torch.Tensor:
    """
    Apply Gaussian blur to an image tensor.
    x: (B, C, H, W), values in [0, 1]
    kernel_size must be odd.
    """
    blurred = torch.stack([
        TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=sigma)
        for img in x
    ])
    return blurred


def add_occlusion(x: torch.Tensor, patch_size: int = 10) -> torch.Tensor:
    """
    Zero out a random square patch in each image.
    x: (B, C, H, W)
    """
    x = x.clone()
    B, C, H, W = x.shape
    for i in range(B):
        top = torch.randint(0, H - patch_size, (1,)).item()
        left = torch.randint(0, W - patch_size, (1,)).item()
        x[i, :, top:top + patch_size, left:left + patch_size] = 0.0
    return x


CORRUPTIONS = {
    "clean":          lambda x: x,
    "gaussian_noise": lambda x: add_gaussian_noise(x, std=0.3),
    "blur":           lambda x: add_blur(x, kernel_size=5, sigma=2.0),
    "occlusion":      lambda x: add_occlusion(x, patch_size=10),
}
