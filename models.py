#!/usr/bin/env python3
"""
Model architectures: DiT, GAN, and MLP Classifier
"""

import torch
import torch.nn as nn
import numpy as np


class DiT(nn.Module):
    """Diffusion Transformer for conditional generation"""
    
    def __init__(self, img_size: int = 16, patch_size: int = 4, in_channels: int = 1,
                 d_model: int = 256, depth: int = 4, num_heads: int = 4, 
                 num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Class embedding (for conditional generation)
        self.class_embed = nn.Embedding(num_classes, d_model)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_size * patch_size * in_channels)
        )
        
    def get_time_embedding(self, timesteps, dim):
        """Sinusoidal time embedding"""
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
        
    def forward(self, x, t, y=None):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Time embedding
        t_emb = self.get_time_embedding(t, self.d_model).to(x.device)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)  # (B, 1, d_model)
        
        # Class embedding (if provided)
        if y is not None:
            c_emb = self.class_embed(y).unsqueeze(1)  # (B, 1, d_model)
            x = x + t_emb + c_emb
        else:
            x = x + t_emb
        
        # Transformer
        x = self.transformer(x)
        
        # Output projection
        x = self.final_layer(x)  # (B, num_patches, patch_size^2 * channels)
        
        # Reshape to image
        x = x.view(B, self.num_patches, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3)  # (B, channels, num_patches, patch_size, patch_size)
        
        # Reconstruct image from patches
        H = W = self.img_size // self.patch_size
        x = x.reshape(B, -1, H, W, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C, H, patch_size, W, patch_size)
        x = x.reshape(B, -1, self.img_size, self.img_size)
        
        return x


class Generator(nn.Module):
    """Conditional Generator for GAN"""
    
    def __init__(self, latent_dim: int = 100, num_features: int = 7, 
                 num_classes: int = 4, hidden: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, latent_dim)
        
        # Generator network
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),  # *2 for noise + class embedding
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, num_features),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, z, labels):
        label_emb = self.class_embed(labels)
        x = torch.cat([z, label_emb], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """Conditional Discriminator for GAN"""
    
    def __init__(self, num_features: int = 7, num_classes: int = 4, hidden: int = 128):
        super().__init__()
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, num_features)
        
        # Discriminator network
        self.net = nn.Sequential(
            nn.Linear(num_features * 2, hidden),  # *2 for data + class embedding
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        label_emb = self.class_embed(labels)
        x_concat = torch.cat([x, label_emb], dim=1)
        return self.net(x_concat)


class MLP(nn.Module):
    """MLP Classifier for downstream evaluation"""
    
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
