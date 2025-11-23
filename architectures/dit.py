#!/usr/bin/env python3
"""
Diffusion Transformer (DiT) architecture

DiT combines Vision Transformer with diffusion models for conditional generation.
Key features:
- Patch-based image encoding
- Sinusoidal time embeddings
- Class-conditional generation
- Transformer encoder backbone
"""

import torch
import torch.nn as nn
import numpy as np


class DiT(nn.Module):
    """Diffusion Transformer for conditional generation
    
    Args:
        img_size: Size of square input images (default: 16)
        patch_size: Size of square patches (default: 4)
        in_channels: Number of input channels (default: 1)
        d_model: Transformer embedding dimension (default: 256)
        depth: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 4)
        num_classes: Number of classes for conditioning (default: 4)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, 
                 img_size: int = 16, 
                 patch_size: int = 4, 
                 in_channels: int = 1,
                 d_model: int = 256, 
                 depth: int = 4, 
                 num_heads: int = 4, 
                 num_classes: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.d_model = d_model
        
        # Patch embedding: convert image patches to embeddings
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Time embedding MLP: converts timestep to embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Class embedding for conditional generation
        self.class_embed = nn.Embedding(num_classes, d_model)
        
        # Transformer encoder backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model * 4,
            dropout=dropout, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection: convert embeddings back to patches
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_size * patch_size * in_channels)
        )
        
    def get_time_embedding(self, timesteps, dim):
        """Generate sinusoidal time embeddings
        
        Args:
            timesteps: Timestep tensor (B,)
            dim: Embedding dimension
            
        Returns:
            Time embeddings (B, dim)
        """
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
        
    def forward(self, x, t, y=None):
        """Forward pass
        
        Args:
            x: Input images (B, C, H, W)
            t: Timesteps (B,)
            y: Optional class labels (B,)
            
        Returns:
            Predicted noise (B, C, H, W)
        """
        B = x.shape[0]
        
        # Convert image to patch embeddings
        x = self.patch_embed(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Generate and add time embedding
        t_emb = self.get_time_embedding(t, self.d_model).to(x.device)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)  # (B, 1, d_model)
        
        # Add class embedding if provided (conditional generation)
        if y is not None:
            c_emb = self.class_embed(y).unsqueeze(1)  # (B, 1, d_model)
            x = x + t_emb + c_emb
        else:
            x = x + t_emb
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project to patch space
        x = self.final_layer(x)  # (B, num_patches, patch_size^2 * channels)
        
        # Reshape patches back to image
        x = x.view(B, self.num_patches, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3)  # (B, channels, num_patches, patch_size, patch_size)
        
        # Reconstruct full image from patches
        H = W = self.img_size // self.patch_size
        x = x.reshape(B, -1, H, W, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C, H, patch_size, W, patch_size)
        x = x.reshape(B, -1, self.img_size, self.img_size)
        
        return x
