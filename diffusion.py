#!/usr/bin/env python3
"""
DDPM (Denoising Diffusion Probabilistic Model) implementation
"""

import torch


class DDPM:
    """Denoising Diffusion Probabilistic Model"""
    
    def __init__(self, timesteps: int = 400, beta_start: float = 1e-4, 
                 beta_end: float = 0.02, device: str = 'cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x0, t, noise=None):
        """Add noise to clean images according to timestep t
        
        Args:
            x0: Clean images (B, C, H, W)
            t: Timesteps (B,)
            noise: Optional pre-generated noise
            
        Returns:
            Noisy images and the noise used
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise
    
    @torch.no_grad()
    def sample(self, model, n: int, img_size: int, channels: int, y=None):
        """Sample images from the diffusion model
        
        Args:
            model: Trained diffusion model
            n: Number of samples to generate
            img_size: Size of square images
            channels: Number of channels
            y: Optional class labels for conditional generation
            
        Returns:
            Generated images (n, channels, img_size, img_size)
        """
        model.eval()
        x = torch.randn(n, channels, img_size, img_size, device=self.device)
        
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((n,), t_idx, dtype=torch.long, device=self.device)
            pred_noise = model(x, t, y=y)
            
            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bar[t_idx]
            beta = self.betas[t_idx]
            
            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1.0 / torch.sqrt(alpha)) * (
                x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise
        
        return x
