#!/usr/bin/env python3
"""
Conditional GAN architecture

Implements Generator and Discriminator for conditional GAN training.
Both models use class embeddings for conditional generation.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional Generator for GAN
    
    Generates synthetic RSS samples conditioned on class labels.
    
    Args:
        latent_dim: Dimension of latent noise vector (default: 100)
        num_features: Number of output features (default: 7)
        num_classes: Number of classes for conditioning (default: 4)
        hidden: Hidden layer size (default: 128)
    """
    
    def __init__(self, 
                 latent_dim: int = 100, 
                 num_features: int = 7, 
                 num_classes: int = 4, 
                 hidden: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        
        # Learnable class embedding
        self.class_embed = nn.Embedding(num_classes, latent_dim)
        
        # Generator network
        # Input: concatenation of noise (latent_dim) and class embedding (latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, num_features),
            nn.Tanh()  # Output in [-1, 1] range (matches standardized data)
        )
    
    def forward(self, z, labels):
        """Generate samples
        
        Args:
            z: Latent noise vectors (B, latent_dim)
            labels: Class labels (B,)
            
        Returns:
            Generated samples (B, num_features)
        """
        # Embed class labels
        label_emb = self.class_embed(labels)
        
        # Concatenate noise and class embedding
        x = torch.cat([z, label_emb], dim=1)
        
        # Generate samples
        return self.net(x)


class Discriminator(nn.Module):
    """Conditional Discriminator for GAN
    
    Distinguishes between real and fake samples, conditioned on class labels.
    
    Args:
        num_features: Number of input features (default: 7)
        num_classes: Number of classes for conditioning (default: 4)
        hidden: Hidden layer size (default: 128)
    """
    
    def __init__(self, 
                 num_features: int = 7, 
                 num_classes: int = 4, 
                 hidden: int = 128):
        super().__init__()
        
        # Learnable class embedding
        self.class_embed = nn.Embedding(num_classes, num_features)
        
        # Discriminator network
        # Input: concatenation of data (num_features) and class embedding (num_features)
        self.net = nn.Sequential(
            nn.Linear(num_features * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, 1),
            nn.Sigmoid()  # Output probability in [0, 1]
        )
    
    def forward(self, x, labels):
        """Classify samples as real or fake
        
        Args:
            x: Input samples (B, num_features)
            labels: Class labels (B,)
            
        Returns:
            Probability of being real (B, 1)
        """
        # Embed class labels
        label_emb = self.class_embed(labels)
        
        # Concatenate data and class embedding
        x_concat = torch.cat([x, label_emb], dim=1)
        
        # Classify
        return self.net(x_concat)
