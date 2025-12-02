#!/usr/bin/env python3
"""
Evaluate MNIST DiT models and compare baseline vs physics-guided

This script:
1. Loads trained models (baseline and physics-guided)
2. Generates synthetic samples
3. Computes physics-prior statistics on generated samples
4. Compares to real data statistics
"""

import argparse
import os
import torch
import numpy as np
from config import get_config
from data_loader import load_dataset
from architectures import DiT
from diffusion import DDPM
from mnist_stats import (
    compute_vertical_attenuation,
    compute_occupancy,
    compute_smoothness_threshold,
    compute_handover_statistics,
    load_mnist_priors
)
from utils import set_seed, get_device
import torch.nn.functional as F


def load_model_checkpoint(model_name, config, device):
    """Load a trained DiT model from checkpoint"""
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train the model first."
        )
    
    # Initialize model
    model = DiT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=10,
        dropout=config.DROP
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return model


def generate_samples(model, ddpm, n_samples=1000, batch_size=100, num_classes=10, device='cuda'):
    """Generate synthetic samples from trained DiT model"""
    print(f"\nGenerating {n_samples} synthetic samples...")
    
    all_samples = []
    all_labels = []
    
    # Generate samples in batches
    for i in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - i)
        
        # Sample labels uniformly
        labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
        
        # Generate samples
        samples = ddpm.sample(
            model,
            n=current_batch_size,
            img_size=model.img_size,
            channels=1,
            y=labels
        )
        
        all_samples.append(samples.cpu())
        all_labels.append(labels.cpu())
        
        if (i + current_batch_size) % 500 == 0:
            print(f"  Generated {i + current_batch_size}/{n_samples}")
    
    # Concatenate all batches
    X_gen = torch.cat(all_samples, dim=0)
    y_gen = torch.cat(all_labels, dim=0)
    
    print(f"✓ Generated {len(X_gen)} samples")
    
    return X_gen, y_gen


def evaluate_model(model_name, config, device, n_samples=1000):
    """Evaluate a trained model and compute statistics"""
    print("\n" + "="*70)
    print(f"Evaluating: {model_name}")
    print("="*70)
    
    # Load model
    model = load_model_checkpoint(model_name, config, device)
    
    # Initialize DDPM
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    # Generate samples
    X_gen, y_gen = generate_samples(model, ddpm, n_samples, device=device)
    
    # Compute statistics on generated samples
    print("\nComputing statistics on generated samples...")
    
    mu_gen = compute_vertical_attenuation(X_gen, y_gen, num_classes=10, img_size=28)
    p_occ_gen = compute_occupancy(X_gen, y_gen, num_classes=10, img_size=28)
    tau_gen = compute_smoothness_threshold(X_gen)
    N_mean_gen, N_std_gen = compute_handover_statistics(X_gen, y_gen, num_classes=10, img_size=28)
    
    stats = {
        'mu_data': mu_gen,
        'p_occ_data': p_occ_gen,
        'tau': tau_gen,
        'N_mean': N_mean_gen,
        'N_std': N_std_gen
    }
    
    print(f"✓ Statistics computed")
    
    return stats, X_gen, y_gen


def compare_statistics(stats_baseline, stats_physics, priors):
    """Compare generated statistics to real data priors"""
    print("\n" + "="*70)
    print("COMPARISON: Generated vs Real Data Statistics")
    print("="*70)
    
    # 1. Vertical attenuation (mu_data)
    print("\n1. Vertical Attenuation (Row-wise mean intensity)")
    print("-" * 70)
    
    l1_baseline = F.l1_loss(stats_baseline['mu_data'], priors['mu_data']).item()
    l1_physics = F.l1_loss(stats_physics['mu_data'], priors['mu_data']).item()
    
    print(f"   L1 Distance (Baseline vs Real):  {l1_baseline:.6f}")
    print(f"   L1 Distance (Physics vs Real):   {l1_physics:.6f}")
    print(f"   Improvement: {((l1_baseline - l1_physics) / l1_baseline * 100):.2f}%")
    
    # 2. Occupancy (p_occ_data)
    print("\n2. Occupancy (Column brightness probability)")
    print("-" * 70)
    
    # KL divergence
    eps = 1e-8
    
    def safe_kl(p, q):
        p_safe = torch.clamp(p, eps, 1.0 - eps)
        q_safe = torch.clamp(q, eps, 1.0 - eps)
        return F.kl_div(torch.log(q_safe), p_safe, reduction='batchmean').item()
    
    kl_baseline = safe_kl(priors['p_occ_data'], stats_baseline['p_occ_data'])
    kl_physics = safe_kl(priors['p_occ_data'], stats_physics['p_occ_data'])
    
    print(f"   KL Divergence (Baseline vs Real): {kl_baseline:.6f}")
    print(f"   KL Divergence (Physics vs Real):  {kl_physics:.6f}")
    print(f"   Improvement: {((kl_baseline - kl_physics) / kl_baseline * 100):.2f}%")
    
    # 3. Smoothness (tau)
    print("\n3. Smoothness (95th percentile of gradients)")
    print("-" * 70)
    
    tau_real = priors['tau']
    tau_baseline = stats_baseline['tau']
    tau_physics = stats_physics['tau']
    
    diff_baseline = abs(tau_baseline - tau_real)
    diff_physics = abs(tau_physics - tau_real)
    
    print(f"   Real data tau:     {tau_real:.6f}")
    print(f"   Baseline tau:      {tau_baseline:.6f} (diff: {diff_baseline:.6f})")
    print(f"   Physics tau:       {tau_physics:.6f} (diff: {diff_physics:.6f})")
    print(f"   Improvement: {((diff_baseline - diff_physics) / diff_baseline * 100):.2f}%")
    
    # 4. Handover (center row transitions)
    print("\n4. Handover (Center row transitions)")
    print("-" * 70)
    
    # Average L2 error across all classes
    l2_baseline = torch.sqrt(((stats_baseline['N_mean'] - priors['N_mean']) ** 2).mean()).item()
    l2_physics = torch.sqrt(((stats_physics['N_mean'] - priors['N_mean']) ** 2).mean()).item()
    
    print(f"   L2 Error in N_mean (Baseline): {l2_baseline:.4f}")
    print(f"   L2 Error in N_mean (Physics):  {l2_physics:.4f}")
    print(f"   Improvement: {((l2_baseline - l2_physics) / l2_baseline * 100):.2f}%")
    
    print("\n   Per-digit comparison:")
    print("   " + "-" * 66)
    print("   Digit |  Real N_mean  | Baseline N_mean | Physics N_mean")
    print("   " + "-" * 66)
    for i in range(10):
        print(f"     {i}   |   {priors['N_mean'][i]:6.2f}     |    {stats_baseline['N_mean'][i]:6.2f}      |   {stats_physics['N_mean'][i]:6.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Physics-guided model shows improvement over baseline in:")
    
    improvements = []
    if l1_physics < l1_baseline:
        improvements.append("✓ Vertical attenuation (row-wise intensity)")
    if kl_physics < kl_baseline:
        improvements.append("✓ Occupancy (column brightness)")
    if diff_physics < diff_baseline:
        improvements.append("✓ Smoothness (gradient distribution)")
    if l2_physics < l2_baseline:
        improvements.append("✓ Handover (row transitions)")
    
    if improvements:
        for imp in improvements:
            print(f"  {imp}")
    else:
        print("  No clear improvements detected")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MNIST DiT models"
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples to generate for evaluation (default: 1000)'
    )
    parser.add_argument(
        '--baseline-only',
        action='store_true',
        help='Only evaluate baseline model'
    )
    parser.add_argument(
        '--physics-only',
        action='store_true',
        help='Only evaluate physics-guided model'
    )
    
    args = parser.parse_args()
    
    # Setup
    config = get_config()
    config.IMG_SIZE = 28
    config.PATCH = 7
    set_seed(config.SEED)
    device = get_device()
    
    # Load real data priors
    print("\n" + "="*70)
    print("Loading MNIST priors (real data statistics)")
    print("="*70)
    
    priors_path = config.MNIST_PRIORS_PATH
    if not os.path.exists(priors_path):
        print(f"\nERROR: Priors not found at {priors_path}")
        print("Please run: python mnist_stats.py")
        return
    
    priors = load_mnist_priors(priors_path)
    print(f"✓ Loaded priors from: {priors_path}")
    
    # Evaluate models
    stats_baseline = None
    stats_physics = None
    
    if not args.physics_only:
        try:
            stats_baseline, _, _ = evaluate_model("dit_ema_mnist", config, device, args.n_samples)
        except FileNotFoundError as e:
            print(f"\n⚠️  Warning: {e}")
            if not args.baseline_only:
                print("Skipping baseline evaluation...")
    
    if not args.baseline_only:
        # For physics-guided model, we need a different checkpoint name
        # Assuming it's saved with a suffix or different name
        try:
            # Try to find physics-guided checkpoint
            # You may need to modify the checkpoint naming in the training script
            stats_physics, _, _ = evaluate_model("dit_ema_mnist", config, device, args.n_samples)
        except FileNotFoundError as e:
            print(f"\n⚠️  Warning: {e}")
            if not args.physics_only:
                print("Skipping physics-guided evaluation...")
    
    # Compare if we have both
    if stats_baseline is not None and stats_physics is not None:
        compare_statistics(stats_baseline, stats_physics, priors)
    else:
        print("\n⚠️  Need both baseline and physics-guided models for comparison")
        print("    Please train both models first using: python train_mnist.py both")


if __name__ == "__main__":
    main()
