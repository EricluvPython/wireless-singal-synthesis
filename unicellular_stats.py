#!/usr/bin/env python3
"""
Precompute physics priors for UniCellular dataset

This script computes the statistics needed for physics-guided losses:
1. Vertical attenuation (mu_data): floor-wise RSS patterns
2. Occupancy (p_occ_data): cell tower presence probability per floor
3. Smoothness (tau): RSS gradient threshold
4. Handover (N_mean, N_std): floor transition statistics
"""

import argparse
import os
import torch
import numpy as np
from config import get_config
from data_loader import load_dataset
from utils import set_seed


def compute_vertical_attenuation(X, y, num_classes, num_features):
    """
    Compute floor-wise mean RSS per cell tower (vertical attenuation)
    
    For RSS data, we compute mean signal strength per floor per cell tower.
    This is analogous to row-wise intensity in images.
    
    Args:
        X: RSS data (N, num_features) 
        y: Floor labels (N,)
        num_classes: Number of floors
        num_features: Number of cell towers
    
    Returns:
        mu_data: (num_classes, num_features) - mean RSS per floor per tower
    """
    mu_data = torch.zeros(num_classes, num_features)
    
    for c in range(num_classes):
        class_mask = (y == c)
        if class_mask.sum() > 0:
            class_samples = X[class_mask]
            # Mean RSS per cell tower for this floor
            mu_data[c] = class_samples.mean(dim=0)
    
    return mu_data


def compute_occupancy(X, y, num_classes, num_features, threshold=-100):
    """
    Compute cell tower occupancy probability per floor
    
    Probability that each cell tower is "visible" (RSS > threshold) on each floor.
    This is analogous to column brightness in images.
    
    Args:
        X: RSS data (N, num_features)
        y: Floor labels (N,)
        num_classes: Number of floors
        num_features: Number of cell towers
        threshold: RSS threshold for "visibility" (default: -100 dBm)
    
    Returns:
        p_occ_data: (num_classes, num_features) - probability of visibility
    """
    p_occ_data = torch.zeros(num_classes, num_features)
    
    for c in range(num_classes):
        class_mask = (y == c)
        if class_mask.sum() > 0:
            class_samples = X[class_mask]
            # Fraction of samples where RSS > threshold
            visible = (class_samples > threshold).float()
            p_occ_data[c] = visible.mean(dim=0)
    
    return p_occ_data


def compute_smoothness_threshold(X, percentile=95):
    """
    Compute RSS variation threshold (smoothness)
    
    For RSS data, we look at variation between consecutive cell towers
    (when sorted by signal strength).
    
    Args:
        X: RSS data (N, num_features)
        percentile: Percentile for threshold (default: 95)
    
    Returns:
        tau: Scalar threshold value
    """
    # Sort RSS values per sample and compute differences
    X_sorted = torch.sort(X, dim=1, descending=True)[0]
    deltas = torch.abs(X_sorted[:, 1:] - X_sorted[:, :-1])
    
    # Compute percentile
    tau = torch.quantile(deltas.flatten(), percentile / 100.0).item()
    
    return tau


def compute_handover_statistics(X, y, num_classes):
    """
    Compute floor transition statistics
    
    For RSS data, we compute how many cell towers "hand over" (change significantly)
    when moving between floors. We approximate this by looking at RSS differences
    between floors.
    
    Args:
        X: RSS data (N, num_features)
        y: Floor labels (N,)
        num_classes: Number of floors
    
    Returns:
        N_mean: (num_classes,) - mean number of significant transitions per floor
        N_std: (num_classes,) - std dev of transitions per floor
    """
    N_mean = torch.zeros(num_classes)
    N_std = torch.zeros(num_classes)
    
    threshold = 10.0  # dBm threshold for "significant" change
    
    for c in range(num_classes):
        class_mask = (y == c)
        if class_mask.sum() > 0:
            class_samples = X[class_mask]
            
            # Count significant variations within each sample
            # (differences between strongest and weakest signals)
            sorted_rss = torch.sort(class_samples, dim=1, descending=True)[0]
            # Count how many towers have RSS > threshold from minimum
            min_rss = sorted_rss[:, -1:]  # Weakest signal
            transitions = ((sorted_rss - min_rss) > threshold).sum(dim=1).float()
            
            N_mean[c] = transitions.mean()
            N_std[c] = transitions.std() if len(transitions) > 1 else 0.0
    
    return N_mean, N_std


def main():
    parser = argparse.ArgumentParser(
        description="Precompute physics priors for UniCellular dataset"
    )
    parser.add_argument(
        '--building',
        choices=['deeb', 'alexu'],
        default='deeb',
        help='Building to use (deeb=9 floors, alexu=7 floors, default: deeb)'
    )
    
    args = parser.parse_args()
    
    # Setup
    config = get_config()
    set_seed(config.SEED)
    
    # Load dataset
    print("\n" + "="*70)
    print(f"Loading UniCellular Dataset ({args.building.upper()})")
    print("="*70)
    
    dataset_info = load_dataset(f"unicellular_{args.building}", config.WORKDIR, config.SEED)
    
    # Convert to torch tensors
    X_train = torch.from_numpy(dataset_info.X_train).float()
    y_train = torch.from_numpy(dataset_info.y_train).long()
    
    num_classes = dataset_info.num_classes
    num_features = dataset_info.num_features
    
    print(f"\nDataset: {dataset_info.name}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Features: {num_features} cell towers")
    print(f"  Classes: {num_classes} floors")
    
    # Compute statistics
    print("\n" + "="*70)
    print("Computing Physics Priors")
    print("="*70)
    
    print("\n1. Vertical Attenuation (floor-wise mean RSS per tower)...")
    mu_data = compute_vertical_attenuation(X_train, y_train, num_classes, num_features)
    print(f"   Shape: {mu_data.shape}")
    print(f"   Range: [{mu_data.min():.2f}, {mu_data.max():.2f}]")
    
    print("\n2. Occupancy (cell tower visibility probability per floor)...")
    p_occ_data = compute_occupancy(X_train, y_train, num_classes, num_features)
    print(f"   Shape: {p_occ_data.shape}")
    print(f"   Range: [{p_occ_data.min():.4f}, {p_occ_data.max():.4f}]")
    
    print("\n3. Smoothness (RSS variation threshold)...")
    tau = compute_smoothness_threshold(X_train)
    print(f"   tau (95th percentile): {tau:.4f}")
    
    print("\n4. Handover (floor transition statistics)...")
    N_mean, N_std = compute_handover_statistics(X_train, y_train, num_classes)
    print(f"   N_mean shape: {N_mean.shape}")
    print(f"   N_mean range: [{N_mean.min():.2f}, {N_mean.max():.2f}]")
    print(f"   N_std range: [{N_std.min():.2f}, {N_std.max():.2f}]")
    
    # Print per-floor statistics
    print(f"\n   Per-floor transition statistics:")
    print(f"   Floor | N_mean | N_std")
    print(f"   " + "-" * 30)
    for i in range(num_classes):
        print(f"     {i}   | {N_mean[i]:6.2f} | {N_std[i]:5.2f}")
    
    # Save priors
    priors = {
        'mu_data': mu_data,
        'p_occ_data': p_occ_data,
        'tau': tau,
        'N_mean': N_mean,
        'N_std': N_std,
        'num_classes': num_classes,
        'num_features': num_features,
        'dataset_name': dataset_info.name,
    }
    
    output_path = os.path.join(config.WORKDIR, f"unicellular_{args.building}_priors.pt")
    torch.save(priors, output_path)
    
    print("\n" + "="*70)
    print(f"✓ Priors saved to: {output_path}")
    print("="*70)
    
    print("\nSummary:")
    print(f"  Dataset: {dataset_info.name}")
    print(f"  Floors: {num_classes}")
    print(f"  Cell towers: {num_features}")
    print(f"  Priors computed and saved successfully!")
    print("\nNext steps:")
    print(f"  1. Train baseline: python train_unicellular.py baseline --building {args.building}")
    print(f"  2. Train physics:  python train_unicellular.py physics --building {args.building}")
    print(f"  3. Or run both:    python train_unicellular.py both --building {args.building}")


def load_unicellular_priors(priors_path):
    """Load precomputed UniCellular priors"""
    if not os.path.exists(priors_path):
        raise FileNotFoundError(f"Priors not found: {priors_path}")
    
    priors = torch.load(priors_path)
    return priors


if __name__ == "__main__":
    main()
