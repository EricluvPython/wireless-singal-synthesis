#!/usr/bin/env python3
"""
Train DiT models on UniCellular dataset with and without physics-guided losses

This script trains generative models on cellular RSS fingerprints for indoor localization.
"""

import argparse
import os
import torch
from config import Config, get_config
from data_loader import load_dataset
from architectures import DiT
from diffusion import DDPM
from trainers import DiTTrainer
from utils import set_seed, get_device


def train_unicellular_baseline(building='deeb', epochs=100, use_wandb=False):
    """
    Experiment A: Baseline UniCellular DiT (no physics losses)
    
    Args:
        building: 'deeb' (9 floors) or 'alexu' (7 floors)
        epochs: Number of training epochs
        use_wandb: Enable wandb logging
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT A: Baseline UniCellular DiT - {building.upper()} (No Physics Losses)")
    print("="*70)
    
    # Configuration
    config = get_config()
    config.DATASET_NAME = f"unicellular_{building}"
    config.USE_PHYSICS_LOSS = False
    config.MODEL_SUFFIX = "_baseline"
    config.GEN_EPOCHS = epochs
    config.USE_WANDB = use_wandb
    
    # Setup
    set_seed(config.SEED)
    device = get_device()
    
    # Load dataset
    dataset_info = load_dataset(f"unicellular_{building}", config.WORKDIR, config.SEED)
    num_features = dataset_info.num_features
    num_classes = dataset_info.num_classes
    
    print(f"\n Dataset: {dataset_info.name}")
    print(f"  Features: {num_features} (cell towers)")
    print(f"  Classes: {num_classes} (floors)")
    
    # For RSS data, we'll use a 1D representation
    # We'll reshape to a square-ish image for DiT
    # E.g., 13 features -> pad to 16 -> 4x4 image
    import math
    img_size = int(math.ceil(math.sqrt(num_features)))
    if img_size % 2 == 1:  # Make it even for easier patching
        img_size += 1
    
    patch_size = img_size // 4  # 4 patches per dimension
    if patch_size == 0:
        patch_size = 1
    
    print(f"  Reshaped to: {img_size}x{img_size} (patch size: {patch_size})")
    
    # Initialize model
    dit_model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=num_classes,
        dropout=config.DROP
    ).to(device)
    
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    # Update config with dataset-specific info
    config.IMG_SIZE = img_size
    config.PATCH = patch_size
    config.NUM_FEATURES = num_features
    
    # Train
    trainer = DiTTrainer(config, dataset_info, device)
    trained_model = trainer.train(dit_model, ddpm, epochs=epochs, use_wandb=use_wandb)
    
    print(f"\n✓ Baseline UniCellular DiT training complete!")
    print(f"✓ Checkpoints saved in: {config.CHECKPOINT_DIR}")
    
    return trained_model, ddpm, config


def train_unicellular_physics_guided(building='deeb', epochs=100, use_wandb=False,
                                     lambda_vert=0.1, lambda_occ=0.1,
                                     lambda_smooth=0.01, lambda_handover=0.01):
    """
    Experiment B: Physics-guided UniCellular DiT
    
    Args:
        building: 'deeb' (9 floors) or 'alexu' (7 floors)
        epochs: Number of training epochs
        use_wandb: Enable wandb logging
        lambda_*: Weights for different physics losses
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT B: Physics-Guided UniCellular DiT - {building.upper()}")
    print("="*70)
    
    # Configuration
    config = get_config()
    config.DATASET_NAME = f"unicellular_{building}"
    config.USE_PHYSICS_LOSS = True
    config.MODEL_SUFFIX = "_physics"
    config.LAMBDA_VERT = lambda_vert
    config.LAMBDA_OCC = lambda_occ
    config.LAMBDA_SMOOTH = lambda_smooth
    config.LAMBDA_HANDOVER = lambda_handover
    config.GEN_EPOCHS = epochs
    config.USE_WANDB = use_wandb
    
    # Setup
    set_seed(config.SEED)
    device = get_device()
    
    # Load dataset
    dataset_info = load_dataset(f"unicellular_{building}", config.WORKDIR, config.SEED)
    num_features = dataset_info.num_features
    num_classes = dataset_info.num_classes
    
    print(f"\nDataset: {dataset_info.name}")
    print(f"  Features: {num_features} (cell towers)")
    print(f"  Classes: {num_classes} (floors)")
    
    # Reshape to square-ish image
    import math
    img_size = int(math.ceil(math.sqrt(num_features)))
    if img_size % 2 == 1:
        img_size += 1
    
    patch_size = img_size // 4
    if patch_size == 0:
        patch_size = 1
    
    print(f"  Reshaped to: {img_size}x{img_size} (patch size: {patch_size})")
    
    # Initialize model
    dit_model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=num_classes,
        dropout=config.DROP
    ).to(device)
    
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    # Update config
    config.IMG_SIZE = img_size
    config.PATCH = patch_size
    config.NUM_FEATURES = num_features
    
    # Train
    trainer = DiTTrainer(config, dataset_info, device)
    trained_model = trainer.train(dit_model, ddpm, epochs=epochs, use_wandb=use_wandb)
    
    print(f"\n✓ Physics-guided UniCellular DiT training complete!")
    print(f"✓ Checkpoints saved in: {config.CHECKPOINT_DIR}")
    
    return trained_model, ddpm, config


def main():
    parser = argparse.ArgumentParser(
        description="Train DiT on UniCellular dataset with optional physics-guided losses"
    )
    parser.add_argument(
        'experiment',
        choices=['baseline', 'physics', 'both'],
        help='Which experiment to run'
    )
    parser.add_argument(
        '--building',
        choices=['deeb', 'alexu'],
        default='deeb',
        help='Building to use (deeb=9 floors, alexu=7 floors, default: deeb)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable wandb logging'
    )
    parser.add_argument(
        '--lambda-vert',
        type=float,
        default=0.1,
        help='Weight for vertical attenuation loss (default: 0.1)'
    )
    parser.add_argument(
        '--lambda-occ',
        type=float,
        default=0.1,
        help='Weight for occupancy loss (default: 0.1)'
    )
    parser.add_argument(
        '--lambda-smooth',
        type=float,
        default=0.01,
        help='Weight for smoothness loss (default: 0.01)'
    )
    parser.add_argument(
        '--lambda-handover',
        type=float,
        default=0.01,
        help='Weight for handover loss (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    # Check if priors exist when running physics experiment
    if args.experiment in ['physics', 'both']:
        priors_path = f"./work_dir/unicellular_{args.building}_priors.pt"
        if not os.path.exists(priors_path):
            print("\n" + "="*70)
            print("ERROR: UniCellular priors not found!")
            print("="*70)
            print(f"\nPlease run the following command first:")
            print(f"  python unicellular_stats.py --building {args.building}")
            print(f"\nThis will precompute the physics priors needed for training.")
            print("="*70 + "\n")
            return
    
    # Run experiments
    if args.experiment == 'baseline':
        train_unicellular_baseline(args.building, args.epochs, args.use_wandb)
    
    elif args.experiment == 'physics':
        train_unicellular_physics_guided(
            args.building, args.epochs, args.use_wandb,
            args.lambda_vert, args.lambda_occ,
            args.lambda_smooth, args.lambda_handover
        )
    
    elif args.experiment == 'both':
        print("\n" + "="*70)
        print(f"Running BOTH experiments sequentially on {args.building.upper()}")
        print("="*70)
        
        # Experiment A: Baseline
        train_unicellular_baseline(args.building, args.epochs, args.use_wandb)
        
        # Experiment B: Physics-guided
        train_unicellular_physics_guided(
            args.building, args.epochs, args.use_wandb,
            args.lambda_vert, args.lambda_occ,
            args.lambda_smooth, args.lambda_handover
        )
        
        print("\n" + "="*70)
        print("Both experiments completed!")
        print("="*70)


if __name__ == "__main__":
    main()
