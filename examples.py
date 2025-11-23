#!/usr/bin/env python3
"""
Example usage of the refactored DiT experiment code

This script demonstrates how to use individual components
for custom experiments and development.
"""

import torch
import numpy as np

from config import Config, get_config
from data_loader import load_dataset
from architectures import DiT, Generator, Discriminator, MLP
from diffusion import DDPM
from trainers import DiTTrainer, GANTrainer, MLPTrainer, CheckpointManager
from synthesis import DiTSynthesizer, GANSynthesizer, KrigingSynthesizer
from utils import set_seed, get_device, vecs_to_images


def example_1_load_and_explore_dataset():
    """Example 1: Load and explore a dataset"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Load and Explore Dataset")
    print("="*70)
    
    config = get_config()
    set_seed(config.SEED)
    
    # Load UCI Indoor dataset
    dataset = load_dataset('uci', config.WORKDIR, config.SEED)
    
    print(f"\nDataset: {dataset.name}")
    print(f"Features: {dataset.num_features}")
    print(f"Classes: {dataset.num_classes}")
    print(f"Training samples: {len(dataset.y_train)}")
    print(f"Test samples: {len(dataset.y_test)}")
    print(f"Train class distribution: {np.bincount(dataset.y_train)}")


def example_2_train_dit_with_custom_settings():
    """Example 2: Train DiT with custom hyperparameters"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Train DiT with Custom Settings")
    print("="*70)
    
    # Create custom config
    config = Config(
        GEN_EPOCHS=500,        # Shorter training for demo
        CHECKPOINT_EVERY=50,   # Checkpoint more frequently
        BATCH=64,              # Smaller batch size
        WIDTH=128,             # Smaller model
        DEPTH=2,               # Fewer layers
    )
    
    device = get_device()
    set_seed(config.SEED)
    
    # Load dataset
    dataset = load_dataset('uci', config.WORKDIR, config.SEED)
    
    # Create model
    model = DiT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=dataset.num_classes,
        dropout=config.DROP
    ).to(device)
    
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    # Train with custom settings
    trainer = DiTTrainer(config, dataset, device)
    trained_model = trainer.train(model, ddpm, epochs=100, use_wandb=False)
    
    print("\n✓ Training complete!")
    print(f"Checkpoints saved in: {config.CHECKPOINT_DIR}")


def example_3_generate_synthetic_samples():
    """Example 3: Generate synthetic samples using trained models"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Generate Synthetic Samples")
    print("="*70)
    
    config = get_config()
    device = get_device()
    set_seed(config.SEED)
    
    dataset = load_dataset('uci', config.WORKDIR, config.SEED)
    
    # For demo, we'll use Kriging which doesn't need pre-trained models
    print("\nGenerating 10 synthetic samples per class using Kriging...")
    
    synthesizer = KrigingSynthesizer()
    X_synth, y_synth = synthesizer.synthesize(
        dataset.X_train,
        dataset.y_train,
        dataset.num_features,
        dataset.num_classes,
        num_samples_per_class=10
    )
    
    print(f"Generated {len(X_synth)} synthetic samples")
    print(f"Synthetic class distribution: {np.bincount(y_synth)}")
    print(f"Synthetic samples shape: {X_synth.shape}")
    print(f"Sample mean: {X_synth.mean():.4f}, std: {X_synth.std():.4f}")


def example_4_train_classifier():
    """Example 4: Train and evaluate MLP classifier"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Train and Evaluate MLP Classifier")
    print("="*70)
    
    config = Config(MLP_EPOCHS=50)  # Shorter training for demo
    device = get_device()
    set_seed(config.SEED)
    
    dataset = load_dataset('uci', config.WORKDIR, config.SEED)
    
    # Train MLP on real data
    print("\nTraining MLP classifier on real data...")
    trainer = MLPTrainer(config, device)
    
    accuracy = trainer.train_and_evaluate(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        dataset.num_features,
        dataset.num_classes,
        use_wandb=False
    )
    
    print(f"\n✓ Test Accuracy: {accuracy*100:.2f}%")


def example_5_checkpoint_management():
    """Example 5: Save and load model checkpoints"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Checkpoint Management")
    print("="*70)
    
    config = get_config()
    device = get_device()
    
    # Create a simple model
    model = MLP(in_dim=7, num_classes=4, hidden=128).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(config.CHECKPOINT_DIR, keep_last_n=3)
    
    # Simulate training and saving checkpoints
    print("\nSaving checkpoints...")
    for epoch in range(5):
        # Simulate some training
        fake_loss = 1.0 / (epoch + 1)
        
        checkpoint_mgr.save(
            model, optimizer, epoch, fake_loss, 
            model_name="example_mlp",
            is_best=(epoch == 2)  # Epoch 2 is the "best"
        )
        print(f"  Saved checkpoint at epoch {epoch} (loss: {fake_loss:.4f})")
    
    print("\nLoading best checkpoint...")
    epoch, loss = checkpoint_mgr.load(model, optimizer, "example_mlp")
    print(f"✓ Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")


def example_6_custom_experiment():
    """Example 6: Custom experiment with specific data ratio"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Experiment with 25% Real Data")
    print("="*70)
    
    config = Config(MLP_EPOCHS=30)
    device = get_device()
    set_seed(config.SEED)
    
    dataset = load_dataset('uci', config.WORKDIR, config.SEED)
    
    # Use only 25% of training data
    from sklearn.model_selection import StratifiedShuffleSplit
    
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.25, random_state=42)
    subset_idx, _ = next(sss.split(dataset.X_train, dataset.y_train))
    
    X_subset = dataset.X_train[subset_idx]
    y_subset = dataset.y_train[subset_idx]
    
    print(f"\nUsing {len(X_subset)} samples (25% of {len(dataset.X_train)})")
    
    # Train classifier on subset
    trainer = MLPTrainer(config, device)
    accuracy = trainer.train_and_evaluate(
        X_subset, y_subset,
        dataset.X_test, dataset.y_test,
        dataset.num_features, dataset.num_classes,
        use_wandb=False
    )
    
    print(f"✓ Accuracy with 25% data: {accuracy*100:.2f}%")
    
    # Now augment with Kriging synthetic data
    print("\nAugmenting with synthetic data...")
    
    # Calculate how many synthetic samples we need
    total_per_class = len(dataset.y_train) // dataset.num_classes
    real_per_class = len(y_subset) // dataset.num_classes
    synth_per_class = total_per_class - real_per_class
    
    synthesizer = KrigingSynthesizer()
    X_synth, y_synth = synthesizer.synthesize(
        X_subset, y_subset,
        dataset.num_features, dataset.num_classes,
        synth_per_class
    )
    
    # Combine real and synthetic
    X_combined = np.concatenate([X_subset, X_synth], axis=0)
    y_combined = np.concatenate([y_subset, y_synth], axis=0)
    
    print(f"Combined: {len(X_subset)} real + {len(X_synth)} synthetic = {len(X_combined)} total")
    
    # Train on combined data
    accuracy_augmented = trainer.train_and_evaluate(
        X_combined, y_combined,
        dataset.X_test, dataset.y_test,
        dataset.num_features, dataset.num_classes,
        use_wandb=False
    )
    
    print(f"✓ Accuracy with augmentation: {accuracy_augmented*100:.2f}%")
    print(f"✓ Improvement: {(accuracy_augmented - accuracy)*100:.2f}%")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("DiT Data Fidelity - Usage Examples")
    print("="*70)
    
    examples = [
        ("Load and Explore Dataset", example_1_load_and_explore_dataset),
        ("Generate Synthetic Samples", example_3_generate_synthetic_samples),
        ("Train Classifier", example_4_train_classifier),
        ("Checkpoint Management", example_5_checkpoint_management),
        ("Custom Experiment", example_6_custom_experiment),
        # Skip example 2 (full DiT training) as it takes too long
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning quick examples (skipping full training)...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✓ All examples complete!")
    print("="*70)


if __name__ == "__main__":
    main()
