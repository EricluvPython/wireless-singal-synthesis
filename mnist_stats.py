#!/usr/bin/env python3
"""
Precompute MNIST statistics for physics-guided loss functions

This module computes:
1. mu_data[f, i]: Mean pixel intensity per row i for each digit f
2. p_occ_data[f, j]: Probability of bright pixels in column j for each digit f
3. tau: 95th percentile of spatial gradients (smoothness threshold)
4. N_mean[f], N_std[f]: Mean and std of transitions in center row for each digit f
"""

import argparse
import os
import torch
import numpy as np
from data_loader import load_dataset
from config import get_config


def compute_vertical_attenuation(X_train, y_train, num_classes=10, img_size=28):
    """
    Compute mu_data[f, i] = average pixel intensity of row i for digit f
    
    Args:
        X_train: Training images (N, 1, 28, 28)
        y_train: Training labels (N,)
        num_classes: Number of digit classes (10)
        img_size: Image size (28)
    
    Returns:
        mu_data: (num_classes, img_size) tensor
    """
    mu_data = torch.zeros(num_classes, img_size)
    
    for f in range(num_classes):
        # Get all images for this digit
        mask = y_train == f
        images_f = X_train[mask]  # (N_f, 1, 28, 28)
        
        if len(images_f) > 0:
            # Average across all samples and the channel dimension
            # Result: (28, 28) -> mean over rows -> (28,)
            mu_data[f] = images_f.mean(dim=0).squeeze(0).mean(dim=1)  # Mean over columns -> row averages
    
    return mu_data


def compute_occupancy(X_train, y_train, num_classes=10, img_size=28, threshold=0.5):
    """
    Compute p_occ_data[f, j] = probability of bright pixels in column j for digit f
    
    A pixel is "bright" if its value > threshold.
    
    Args:
        X_train: Training images (N, 1, 28, 28)
        y_train: Training labels (N,)
        num_classes: Number of digit classes (10)
        img_size: Image size (28)
        threshold: Brightness threshold
    
    Returns:
        p_occ_data: (num_classes, img_size) tensor
    """
    p_occ_data = torch.zeros(num_classes, img_size)
    
    for f in range(num_classes):
        mask = y_train == f
        images_f = X_train[mask]  # (N_f, 1, 28, 28)
        
        if len(images_f) > 0:
            # Binarize: is pixel bright?
            bright = (images_f > threshold).float()  # (N_f, 1, 28, 28)
            
            # For each column j, compute: has bright pixel in that column
            # bright.squeeze(1): (N_f, 28, 28)
            # bright.squeeze(1).max(dim=1): max over rows -> (N_f, 28) - bool for each column
            has_bright_in_col = bright.squeeze(1).max(dim=1)[0]  # (N_f, 28)
            
            # Probability = fraction of samples with bright pixels in column j
            p_occ_data[f] = has_bright_in_col.float().mean(dim=0)  # (28,)
    
    return p_occ_data


def compute_smoothness_threshold(X_train, percentile=95):
    """
    Compute tau = 95th percentile of spatial gradients across training set
    
    Args:
        X_train: Training images (N, 1, 28, 28)
        percentile: Percentile to use (95)
    
    Returns:
        tau: Scalar threshold value
    """
    # Compute horizontal and vertical differences
    # Horizontal: |I(i, j+1) - I(i, j)|
    delta_h = torch.abs(X_train[:, :, :, 1:] - X_train[:, :, :, :-1])
    
    # Vertical: |I(i+1, j) - I(i, j)|
    delta_v = torch.abs(X_train[:, :, 1:, :] - X_train[:, :, :-1, :])
    
    # Flatten all differences
    all_diffs = torch.cat([delta_h.flatten(), delta_v.flatten()])
    
    # Sample if too large (to avoid memory issues)
    if len(all_diffs) > 10_000_000:
        indices = torch.randperm(len(all_diffs))[:10_000_000]
        all_diffs = all_diffs[indices]
    
    # Compute 95th percentile using numpy (more memory efficient)
    tau = np.percentile(all_diffs.numpy(), percentile)
    
    return float(tau)


def count_transitions(row, threshold=0.5):
    """
    Count transitions (0->1 or 1->0) in a binarized row
    
    Args:
        row: 1D array or tensor
        threshold: Binarization threshold
    
    Returns:
        Number of transitions
    """
    binary = (row > threshold).float()
    # Compute differences between consecutive elements
    diffs = torch.abs(binary[1:] - binary[:-1])
    # Count non-zero differences (transitions)
    return diffs.sum().item()


def compute_handover_statistics(X_train, y_train, num_classes=10, img_size=28, threshold=0.5):
    """
    Compute N_mean[f], N_std[f] = mean and std of transitions in center row for each digit f
    
    Args:
        X_train: Training images (N, 1, 28, 28)
        y_train: Training labels (N,)
        num_classes: Number of digit classes (10)
        img_size: Image size (28)
        threshold: Binarization threshold
    
    Returns:
        N_mean: (num_classes,) tensor
        N_std: (num_classes,) tensor
    """
    N_mean = torch.zeros(num_classes)
    N_std = torch.zeros(num_classes)
    
    center_row = img_size // 2  # Row 14 for size 28
    
    for f in range(num_classes):
        mask = y_train == f
        images_f = X_train[mask]  # (N_f, 1, 28, 28)
        
        if len(images_f) > 0:
            # Extract center row for all images
            center_rows = images_f[:, 0, center_row, :]  # (N_f, 28)
            
            # Count transitions for each image
            transitions = []
            for row in center_rows:
                n_trans = count_transitions(row, threshold)
                transitions.append(n_trans)
            
            transitions = torch.tensor(transitions, dtype=torch.float32)
            N_mean[f] = transitions.mean()
            N_std[f] = transitions.std()
    
    return N_mean, N_std


def precompute_mnist_priors(workdir="./work_dir", save_path=None, seed=42):
    """
    Precompute all MNIST priors and save to disk
    
    Args:
        workdir: Working directory
        save_path: Path to save priors (default: workdir/mnist_priors.pt)
        seed: Random seed
    """
    print("\n" + "="*70)
    print("Precomputing MNIST Physics Priors")
    print("="*70)
    
    # Load MNIST dataset
    from data_loader import load_dataset
    dataset_info = load_dataset("mnist", workdir, seed)
    
    # Convert to tensors
    X_train = torch.from_numpy(dataset_info.X_train).float()
    y_train = torch.from_numpy(dataset_info.y_train).long()
    
    num_classes = dataset_info.num_classes
    img_size = X_train.shape[2]  # Should be 28
    
    print(f"\nComputing priors from {len(X_train)} training samples")
    print(f"Image size: {img_size}x{img_size}, Classes: {num_classes}\n")
    
    # 1. Vertical attenuation (mu_data)
    print("1. Computing vertical attenuation statistics (mu_data)...")
    mu_data = compute_vertical_attenuation(X_train, y_train, num_classes, img_size)
    print(f"   mu_data shape: {mu_data.shape}")
    
    # 2. Occupancy (p_occ_data)
    print("2. Computing occupancy statistics (p_occ_data)...")
    p_occ_data = compute_occupancy(X_train, y_train, num_classes, img_size)
    print(f"   p_occ_data shape: {p_occ_data.shape}")
    
    # 3. Smoothness (tau)
    print("3. Computing smoothness threshold (tau)...")
    tau = compute_smoothness_threshold(X_train)
    print(f"   tau = {tau:.6f}")
    
    # 4. Handover (N_mean, N_std)
    print("4. Computing handover statistics (N_mean, N_std)...")
    N_mean, N_std = compute_handover_statistics(X_train, y_train, num_classes, img_size)
    print(f"   N_mean shape: {N_mean.shape}")
    print(f"   N_std shape: {N_std.shape}")
    
    # Save priors
    if save_path is None:
        save_path = os.path.join(workdir, "mnist_priors.pt")
    
    priors = {
        'mu_data': mu_data,
        'p_occ_data': p_occ_data,
        'tau': tau,
        'N_mean': N_mean,
        'N_std': N_std,
        'img_size': img_size,
        'num_classes': num_classes,
        'threshold': 0.5  # brightness threshold used
    }
    
    torch.save(priors, save_path)
    print(f"\n✓ Priors saved to: {save_path}")
    print("="*70)
    
    return priors


def load_mnist_priors(path="./work_dir/mnist_priors.pt"):
    """Load precomputed MNIST priors from disk"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MNIST priors not found at {path}. "
            f"Please run mnist_stats.py first to precompute priors."
        )
    
    priors = torch.load(path)
    return priors


def main():
    parser = argparse.ArgumentParser(
        description="Precompute MNIST physics priors"
    )
    parser.add_argument(
        '--workdir',
        type=str,
        default='./work_dir',
        help='Working directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for priors (default: workdir/mnist_priors.pt)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Precompute and save
    priors = precompute_mnist_priors(
        workdir=args.workdir,
        save_path=args.output,
        seed=args.seed
    )
    
    # Print summary
    print("\nPrior Statistics Summary:")
    print("-" * 70)
    print(f"mu_data (row averages per digit):")
    for f in range(priors['num_classes']):
        print(f"  Digit {f}: mean={priors['mu_data'][f].mean():.4f}, "
              f"std={priors['mu_data'][f].std():.4f}")
    
    print(f"\np_occ_data (column brightness probability):")
    for f in range(priors['num_classes']):
        print(f"  Digit {f}: mean={priors['p_occ_data'][f].mean():.4f}, "
              f"std={priors['p_occ_data'][f].std():.4f}")
    
    print(f"\ntau (smoothness threshold): {priors['tau']:.6f}")
    
    print(f"\nN_mean (center row transitions):")
    for f in range(priors['num_classes']):
        print(f"  Digit {f}: {priors['N_mean'][f]:.2f} ± {priors['N_std'][f]:.2f}")


if __name__ == "__main__":
    main()
