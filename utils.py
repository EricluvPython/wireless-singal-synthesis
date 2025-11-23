#!/usr/bin/env python3
"""
Utility functions for data processing and visualization
"""

import numpy as np
import torch


def vecs_to_images(vecs_std: np.ndarray, img_size: int = 16, clip: float = 3.0) -> torch.Tensor:
    """Convert standardized feature vectors to square images in [-1,1]
    
    Args:
        vecs_std: Standardized feature vectors (B, num_features)
        img_size: Size of output square image
        clip: Clipping value for standardized features
        
    Returns:
        Images tensor (B, 1, img_size, img_size)
    """
    v = np.clip(vecs_std, -clip, clip) / clip  # -> [-1,1]
    B, num_features = v.shape
    
    imgs = np.zeros((B, 1, img_size, img_size), dtype=np.float32)
    total_pixels = img_size * img_size
    
    if num_features <= img_size:
        # Fill first row with features, tile vertically
        row = np.zeros((B, img_size), dtype=np.float32)
        row[:, :num_features] = v
        tile = np.repeat(row[:, None, :], img_size, axis=1)  # (B, img_size, img_size)
        imgs[:, 0, :, :] = tile
    else:
        # Reshape features to fill the image
        for i in range(B):
            flat = np.zeros(total_pixels, dtype=np.float32)
            flat[:num_features] = v[i]
            imgs[i, 0, :, :] = flat.reshape(img_size, img_size)
    
    return torch.from_numpy(imgs)


def images_to_vecs_std(imgs: torch.Tensor, num_features: int, clip: float = 3.0) -> np.ndarray:
    """Convert square images back to standardized feature vectors
    
    Args:
        imgs: Images tensor (B, 1, img_size, img_size)
        num_features: Number of features to extract
        clip: Clipping value used during encoding
        
    Returns:
        Standardized feature vectors (B, num_features)
    """
    x = imgs[:, 0, :, :]  # (B, img_size, img_size)
    B = x.shape[0]
    img_size = x.shape[1]
    
    if num_features <= img_size:
        # Extract from first row (averaged across vertical dimension)
        cols = []
        for j in range(num_features):
            cols.append(x[:, :, j].mean(dim=1))
        v_scaled = torch.stack(cols, dim=1)
    else:
        # Extract from flattened image
        flat = x.reshape(B, -1)[:, :num_features]
        v_scaled = flat
    
    vecs_std = (v_scaled * clip).cpu().numpy().astype(np.float32)
    return vecs_std


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the appropriate device (CUDA if available, else CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
