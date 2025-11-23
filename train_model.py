#!/usr/bin/env python3
"""
Train individual models (DiT or GAN) on a dataset with checkpointing

This script allows you to train just the generative models without running
the full experiment. Useful for:
- Model development and hyperparameter tuning
- Pre-training models for later use
- Testing on custom datasets
"""

import argparse
import torch

from config import Config, get_config
from data_loader import load_dataset
from architectures import DiT, Generator, Discriminator
from diffusion import DDPM
from trainers import DiTTrainer, GANTrainer
from utils import set_seed, get_device

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_dit(config: Config, dataset_name: str, epochs: int = None, 
              use_wandb: bool = False):
    """Train DiT model on a dataset"""
    # Setup
    set_seed(config.SEED)
    device = get_device()
    
    # Load dataset
    dataset_info = load_dataset(dataset_name, config.WORKDIR, config.SEED)
    
    # Initialize model
    dit_model = DiT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=dataset_info.num_classes,
        dropout=config.DROP
    ).to(device)
    
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config=vars(config),
            name=f"dit_{dataset_name}_{epochs or config.GEN_EPOCHS}_epochs"
        )
    
    # Train
    trainer = DiTTrainer(config, dataset_info, device)
    trained_model = trainer.train(dit_model, ddpm, epochs=epochs, use_wandb=use_wandb)
    
    # Finish wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\n✓ DiT training complete!")
    print(f"✓ Checkpoints saved in: {config.CHECKPOINT_DIR}")


def train_gan(config: Config, dataset_name: str, epochs: int = None,
              use_wandb: bool = False):
    """Train GAN model on a dataset"""
    # Setup
    set_seed(config.SEED)
    device = get_device()
    
    # Load dataset
    dataset_info = load_dataset(dataset_name, config.WORKDIR, config.SEED)
    
    # Initialize models
    generator = Generator(
        latent_dim=config.GAN_LATENT_DIM,
        num_features=dataset_info.num_features,
        num_classes=dataset_info.num_classes,
        hidden=config.GAN_HIDDEN
    ).to(device)
    
    discriminator = Discriminator(
        num_features=dataset_info.num_features,
        num_classes=dataset_info.num_classes,
        hidden=config.GAN_HIDDEN
    ).to(device)
    
    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config=vars(config),
            name=f"gan_{dataset_name}_{epochs or config.GEN_EPOCHS}_epochs"
        )
    
    # Train
    trainer = GANTrainer(config, dataset_info, device)
    trained_generator = trainer.train(
        generator, discriminator, epochs=epochs, use_wandb=use_wandb
    )
    
    # Finish wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\n✓ GAN training complete!")
    print(f"✓ Checkpoints saved in: {config.CHECKPOINT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Train individual models (DiT or GAN)"
    )
    parser.add_argument(
        'model',
        choices=['dit', 'gan'],
        help='Model to train'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='indoor',
        choices=['indoor', 'outdoor', 'uci', 'powder'],
        help='Dataset to train on'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: from config)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='Wandb project name'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    # Override config from args
    if args.wandb_project is not None:
        config.WANDB_PROJECT = args.wandb_project
    
    use_wandb = config.USE_WANDB and not args.no_wandb and WANDB_AVAILABLE
    
    # Normalize dataset name
    dataset_map = {
        'indoor': 'uci',
        'uci': 'uci',
        'outdoor': 'powder',
        'powder': 'powder'
    }
    dataset_name = dataset_map[args.dataset.lower()]
    
    # Train model
    if args.model == 'dit':
        train_dit(config, dataset_name, args.epochs, use_wandb)
    elif args.model == 'gan':
        train_gan(config, dataset_name, args.epochs, use_wandb)


if __name__ == "__main__":
    main()
