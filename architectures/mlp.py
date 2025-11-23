#!/usr/bin/env python3
"""
Multi-Layer Perceptron (MLP) Classifier

Simple feedforward neural network for downstream classification tasks.
Used to evaluate the quality of synthetic data augmentation.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP Classifier for downstream evaluation
    
    A deep feedforward network with batch normalization, ReLU activations,
    and dropout for regularization. Used to evaluate how well synthetic
    data improves classification performance.
    
    Args:
        in_dim: Number of input features
        num_classes: Number of output classes
        hidden: Hidden layer size (default: 256)
    """
    
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 4
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input features (B, in_dim)
            
        Returns:
            Logits (B, num_classes)
        """
        return self.net(x)
