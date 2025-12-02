#!/usr/bin/env python3
"""
Train DiT models on MNIST with and without physics-guided losses

This script provides a simple interface to run the MNIST toy experiments.
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


def train_mnist_baseline(epochs=10, use_wandb=False):
    """
    Experiment A: Baseline MNIST DiT (no physics losses)
    """
    print("\n" + "="*70)
    print("EXPERIMENT A: Baseline MNIST DiT (No Physics Losses)")
    print("="*70)
    
    # Configuration
    config = get_config()
    config.DATASET_NAME = "mnist"
    config.IMG_SIZE = 28
    config.PATCH = 7  # 28/7 = 4 patches per dimension -> 16 total patches
    config.USE_PHYSICS_LOSS = False
    config.MODEL_SUFFIX = "_baseline"  # Add suffix for checkpoint naming
    config.GEN_EPOCHS = epochs
    config.USE_WANDB = use_wandb
    
    # Setup
    set_seed(config.SEED)
    device = get_device()
    
    # Load dataset
    dataset_info = load_dataset("mnist", config.WORKDIR, config.SEED)
    
    # Initialize model (MNIST has 10 classes)
    dit_model = DiT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=10,  # MNIST has 10 digit classes
        dropout=config.DROP
    ).to(device)
    
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    # Train
    trainer = DiTTrainer(config, dataset_info, device)
    trained_model = trainer.train(dit_model, ddpm, epochs=epochs, use_wandb=use_wandb)
    
    print(f"\n✓ Baseline MNIST DiT training complete!")
    print(f"✓ Checkpoints saved in: {config.CHECKPOINT_DIR}")
    
    return trained_model, ddpm, config


def train_mnist_physics_guided(epochs=10, use_wandb=False,
                                lambda_vert=0.1, lambda_occ=0.1,
                                lambda_smooth=0.01, lambda_handover=0.01):
    """
    Experiment B: Physics-guided MNIST DiT
    """
    print("\n" + "="*70)
    print("EXPERIMENT B: Physics-Guided MNIST DiT")
    print("="*70)
    
    # Configuration
    config = get_config()
    config.DATASET_NAME = "mnist"
    config.IMG_SIZE = 28
    config.PATCH = 7
    config.USE_PHYSICS_LOSS = True
    config.MODEL_SUFFIX = "_physics"  # Add suffix for checkpoint naming
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
    dataset_info = load_dataset("mnist", config.WORKDIR, config.SEED)
    
    # Initialize model
    dit_model = DiT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=10,
        dropout=config.DROP
    ).to(device)
    
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    # Train
    trainer = DiTTrainer(config, dataset_info, device)
    trained_model = trainer.train(dit_model, ddpm, epochs=epochs, use_wandb=use_wandb)
    
    print(f"\n✓ Physics-guided MNIST DiT training complete!")
    print(f"✓ Checkpoints saved in: {config.CHECKPOINT_DIR}")
    
    return trained_model, ddpm, config


def main():
    parser = argparse.ArgumentParser(
        description="Train DiT on MNIST with optional physics-guided losses"
    )
    parser.add_argument(
        'experiment',
        choices=['baseline', 'physics', 'both'],
        help='Which experiment to run'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
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
        priors_path = "./work_dir/mnist_priors.pt"
        if not os.path.exists(priors_path):
            print("\n" + "="*70)
            print("ERROR: MNIST priors not found!")
            print("="*70)
            print(f"\nPlease run the following command first:")
            print(f"  python mnist_stats.py")
            print(f"\nThis will precompute the physics priors needed for training.")
            print("="*70 + "\n")
            return
    
    # Run experiments
    if args.experiment == 'baseline':
        train_mnist_baseline(args.epochs, args.use_wandb)
    
    elif args.experiment == 'physics':
        train_mnist_physics_guided(
            args.epochs, args.use_wandb,
            args.lambda_vert, args.lambda_occ,
            args.lambda_smooth, args.lambda_handover
        )
    
    elif args.experiment == 'both':
        print("\n" + "="*70)
        print("Running BOTH experiments sequentially")
        print("="*70)
        
        # Experiment A: Baseline
        train_mnist_baseline(args.epochs, args.use_wandb)
        
        # Experiment B: Physics-guided
        train_mnist_physics_guided(
            args.epochs, args.use_wandb,
            args.lambda_vert, args.lambda_occ,
            args.lambda_smooth, args.lambda_handover
        )
        
        print("\n" + "="*70)
        print("Both experiments completed!")
        print("="*70)


if __name__ == "__main__":
    main()
