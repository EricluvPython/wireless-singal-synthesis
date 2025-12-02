#!/usr/bin/env python3
"""
Configuration management for DiT Data Fidelity Experiment
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    """Main configuration for the experiment"""
    
    # Paths
    WORKDIR: str = "./work_dir"
    CHECKPOINT_DIR: str = "./checkpoints"
    
    # Dataset
    DATASET_NAME: str = "uci"  # uci, powder, mnist
    
    # Model architecture
    IMG_SIZE: int = 16           # rasterized "image" is 16x16 (28 for MNIST)
    CHANNELS: int = 1            # grayscale
    PATCH: int = 4               # for DiT: 4x4 patches -> (16/4)^2=16 tokens (7 for MNIST: 28/7=4 patches per dim)
    WIDTH: int = 256             # transformer d_model
    DEPTH: int = 4               # transformer layers
    HEADS: int = 4               # attention heads
    DROP: float = 0.1
    
    # Training
    BATCH: int = 128
    GEN_EPOCHS: int = 2000       # epochs for DiT/GAN training
    MLP_EPOCHS: int = 100        # epochs for MLP classifier
    LR: float = 2e-4
    MLP_LR: float = 2e-3
    
    # Diffusion
    BETA_START: float = 1e-4
    BETA_END: float = 0.02
    TIMESTEPS: int = 400
    
    # EMA
    EMA: bool = True
    EMA_DECAY: float = 0.999
    
    # GAN
    GAN_LATENT_DIM: int = 100
    GAN_HIDDEN: int = 128
    
    # MLP
    MLP_HIDDEN: int = 256
    
    # Data processing
    VEC_CLIP: float = 3.0        # map standardized features linearly into [-1,1]
    
    # Experiment
    TRAIN_RATIOS: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0])
    
    # Checkpointing
    CHECKPOINT_EVERY: int = 5  # Save checkpoint every N epochs
    KEEP_LAST_N: int = 3         # Keep only last N checkpoints
    
    # Wandb
    USE_WANDB: bool = True
    WANDB_PROJECT: str = "dit-data-fidelity"
    WANDB_ENTITY: str = None     # Set to your wandb username/team
    
    # Physics-guided loss (for MNIST and UniCellular experiments)
    USE_PHYSICS_LOSS: bool = False
    LAMBDA_VERT: float = 0.01
    LAMBDA_OCC: float = 0.01
    LAMBDA_SMOOTH: float = 0.001
    LAMBDA_HANDOVER: float = 0.0001
    MNIST_PRIORS_PATH: str = "./work_dir/mnist_priors.pt"
    
    # Adaptive physics loss weighting
    USE_ADAPTIVE_PHYSICS_LOSS: bool = True  # Automatically scale physics losses
    PHYSICS_LOSS_MAX_RATIO: float = 0.5     # Physics losses won't exceed this fraction of diffusion loss
    
    # Two-stage training: warmup without physics, then enable physics
    PHYSICS_WARMUP_EPOCHS: int = 100  # Train without physics for this many epochs, then enable
    
    # Random seed
    SEED: int = 42
    
    def __post_init__(self):
        os.makedirs(self.WORKDIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)


def get_config():
    """Get the default configuration"""
    return Config()
