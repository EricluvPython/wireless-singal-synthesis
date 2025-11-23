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
    
    # Model architecture
    IMG_SIZE: int = 16           # rasterized "image" is 16x16
    CHANNELS: int = 1            # grayscale
    PATCH: int = 4               # for DiT: 4x4 patches -> (16/4)^2=16 tokens
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
    CHECKPOINT_EVERY: int = 100  # Save checkpoint every N epochs
    KEEP_LAST_N: int = 3         # Keep only last N checkpoints
    
    # Wandb
    USE_WANDB: bool = True
    WANDB_PROJECT: str = "dit-data-fidelity"
    WANDB_ENTITY: str = None     # Set to your wandb username/team
    
    # Random seed
    SEED: int = 42
    
    def __post_init__(self):
        os.makedirs(self.WORKDIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)


def get_config():
    """Get the default configuration"""
    return Config()
