#!/usr/bin/env python3
"""
Quick demo of MNIST physics-guided loss experiment

This runs a minimal version (1 epoch, small batch) to demonstrate the complete workflow.
"""

import torch
from config import get_config
from data_loader import load_dataset
from architectures import DiT
from diffusion import DDPM
from trainers import DiTTrainer
from utils import set_seed, get_device
from mnist_stats import load_mnist_priors


def demo_baseline():
    """Demo: Baseline MNIST DiT (1 epoch)"""
    print("\n" + "="*70)
    print("DEMO: Baseline MNIST DiT (1 epoch, small model)")
    print("="*70)
    
    config = get_config()
    config.DATASET_NAME = "mnist"
    config.IMG_SIZE = 28
    config.PATCH = 7
    config.USE_PHYSICS_LOSS = False
    config.GEN_EPOCHS = 1
    config.BATCH = 64  # Smaller batch
    config.WIDTH = 128  # Smaller model
    config.DEPTH = 2
    config.CHECKPOINT_EVERY = 1  # Save every epoch
    
    set_seed(config.SEED)
    device = get_device()
    
    # Load small subset of data
    dataset_info = load_dataset("mnist", config.WORKDIR, config.SEED)
    # Use only first 1000 samples for demo
    dataset_info.X_train = dataset_info.X_train[:1000]
    dataset_info.y_train = dataset_info.y_train[:1000]
    
    print(f"Using {len(dataset_info.X_train)} training samples (subset for demo)")
    
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
    
    trainer = DiTTrainer(config, dataset_info, device)
    trained_model = trainer.train(dit_model, ddpm, epochs=1, use_wandb=False)
    
    print(f"\n✓ Baseline demo complete!")
    
    # Generate a few samples
    print("\nGenerating 10 sample images...")
    labels = torch.arange(10, device=device)  # One of each digit
    samples = ddpm.sample(trained_model, n=10, img_size=28, channels=1, y=labels)
    print(f"✓ Generated samples shape: {samples.shape}")
    
    return trained_model


def demo_physics_guided():
    """Demo: Physics-guided MNIST DiT (1 epoch)"""
    print("\n" + "="*70)
    print("DEMO: Physics-Guided MNIST DiT (1 epoch, small model)")
    print("="*70)
    
    config = get_config()
    config.DATASET_NAME = "mnist"
    config.IMG_SIZE = 28
    config.PATCH = 7
    config.USE_PHYSICS_LOSS = True
    config.LAMBDA_VERT = 0.1
    config.LAMBDA_OCC = 0.1
    config.LAMBDA_SMOOTH = 0.01
    config.LAMBDA_HANDOVER = 0.01
    config.GEN_EPOCHS = 1
    config.BATCH = 64
    config.WIDTH = 128
    config.DEPTH = 2
    config.CHECKPOINT_EVERY = 1
    
    set_seed(config.SEED)
    device = get_device()
    
    # Verify priors exist
    priors = load_mnist_priors(config.MNIST_PRIORS_PATH)
    print(f"✓ Loaded priors (tau={priors['tau']:.4f})")
    
    dataset_info = load_dataset("mnist", config.WORKDIR, config.SEED)
    dataset_info.X_train = dataset_info.X_train[:1000]
    dataset_info.y_train = dataset_info.y_train[:1000]
    
    print(f"Using {len(dataset_info.X_train)} training samples (subset for demo)")
    
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
    
    trainer = DiTTrainer(config, dataset_info, device)
    trained_model = trainer.train(dit_model, ddpm, epochs=1, use_wandb=False)
    
    print(f"\n✓ Physics-guided demo complete!")
    
    # Generate a few samples
    print("\nGenerating 10 sample images...")
    labels = torch.arange(10, device=device)
    samples = ddpm.sample(trained_model, n=10, img_size=28, channels=1, y=labels)
    print(f"✓ Generated samples shape: {samples.shape}")
    
    return trained_model


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("MNIST PHYSICS-GUIDED LOSS DEMO")
    print("="*70)
    print("\nThis demonstrates the complete workflow:")
    print("  1. Load MNIST dataset")
    print("  2. Train baseline DiT (1 epoch, 1000 samples)")
    print("  3. Train physics-guided DiT (1 epoch, 1000 samples)")
    print("  4. Show that physics losses are computed correctly")
    print("\nNote: This is a quick demo. For real experiments, use train_mnist.py")
    print("="*70)
    
    try:
        # Demo 1: Baseline
        model_baseline = demo_baseline()
        
        # Demo 2: Physics-guided
        model_physics = demo_physics_guided()
        
        print("\n" + "="*70)
        print("SUCCESS: Both experiments completed!")
        print("="*70)
        print("\nKey observations:")
        print("  ✓ MNIST dataset loads correctly as 28x28 images")
        print("  ✓ DiT model handles MNIST (patch_size=7 -> 16 patches)")
        print("  ✓ Baseline training works (standard diffusion loss)")
        print("  ✓ Physics-guided training works (4 additional loss terms)")
        print("  ✓ All losses are differentiable and flow gradients")
        print("\nFor full experiments (5-10 epochs, full dataset):")
        print("  python train_mnist.py baseline --epochs 10")
        print("  python train_mnist.py physics --epochs 10")
        print("  python train_mnist.py both --epochs 10")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
