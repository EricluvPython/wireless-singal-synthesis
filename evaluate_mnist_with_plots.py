#!/usr/bin/env python3
"""
Evaluate MNIST DiT models with comprehensive visualizations

This script:
1. Loads trained models (baseline and physics-guided)
2. Generates synthetic samples
3. Computes and visualizes physics-prior statistics
4. Calculates FID scores
5. Creates comparison plots
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_model_checkpoint(checkpoint_name, config, device):
    """Load a trained DiT model from checkpoint"""
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{checkpoint_name}_best.pt")
    
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


def calculate_fid(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images
    
    For MNIST, we use a simplified version based on pixel statistics
    """
    # Flatten images
    real_flat = real_images.reshape(real_images.shape[0], -1).numpy()
    gen_flat = generated_images.reshape(generated_images.shape[0], -1).numpy()
    
    # Calculate mean and covariance
    mu_real = np.mean(real_flat, axis=0)
    mu_gen = np.mean(gen_flat, axis=0)
    
    sigma_real = np.cov(real_flat, rowvar=False)
    sigma_gen = np.cov(gen_flat, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_gen
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    if not np.isfinite(covmean).all():
        print("Warning: FID calculation resulted in singular product; adding epsilon")
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_gen + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    
    return fid


def plot_generated_samples(X_baseline, y_baseline, X_physics, y_physics, X_real, y_real, save_dir):
    """Plot examples of generated images from both models and real data"""
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    
    # Plot one example of each digit for each model
    for digit in range(10):
        # Real
        real_idx = (y_real == digit).nonzero(as_tuple=True)[0][0]
        axes[0, digit].imshow(X_real[real_idx, 0], cmap='gray')
        axes[0, digit].axis('off')
        if digit == 0:
            axes[0, digit].set_title('Real', fontsize=10, loc='left')
        
        # Baseline
        baseline_idx = (y_baseline == digit).nonzero(as_tuple=True)[0][0]
        axes[1, digit].imshow(X_baseline[baseline_idx, 0], cmap='gray')
        axes[1, digit].axis('off')
        if digit == 0:
            axes[1, digit].set_title('Baseline', fontsize=10, loc='left')
        
        # Physics
        physics_idx = (y_physics == digit).nonzero(as_tuple=True)[0][0]
        axes[2, digit].imshow(X_physics[physics_idx, 0], cmap='gray')
        axes[2, digit].axis('off')
        if digit == 0:
            axes[2, digit].set_title('Physics', fontsize=10, loc='left')
        
        # Add digit label on top
        axes[0, digit].text(0.5, 1.1, str(digit), ha='center', va='bottom', 
                           transform=axes[0, digit].transAxes, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generated_samples.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'generated_samples.png')}")
    plt.close()


def plot_vertical_attenuation(mu_baseline, mu_physics, mu_real, save_dir):
    """Plot comparison of vertical attenuation (row-wise intensity)"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for digit in range(10):
        ax = axes[digit]
        rows = np.arange(28)
        
        ax.plot(rows, mu_real[digit].numpy(), 'k-', linewidth=2, label='Real', alpha=0.8)
        ax.plot(rows, mu_baseline[digit].numpy(), 'b--', linewidth=1.5, label='Baseline', alpha=0.7)
        ax.plot(rows, mu_physics[digit].numpy(), 'r:', linewidth=1.5, label='Physics', alpha=0.7)
        
        ax.set_title(f'Digit {digit}', fontweight='bold')
        ax.set_xlabel('Row')
        ax.set_ylabel('Mean Intensity')
        ax.grid(True, alpha=0.3)
        
        if digit == 0:
            ax.legend(loc='best', fontsize=8)
    
    plt.suptitle('Vertical Attenuation: Row-wise Mean Intensity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vertical_attenuation.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'vertical_attenuation.png')}")
    plt.close()


def plot_occupancy(p_occ_baseline, p_occ_physics, p_occ_real, save_dir):
    """Plot comparison of occupancy (column brightness probability)"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for digit in range(10):
        ax = axes[digit]
        cols = np.arange(28)
        
        ax.plot(cols, p_occ_real[digit].numpy(), 'k-', linewidth=2, label='Real', alpha=0.8)
        ax.plot(cols, p_occ_baseline[digit].numpy(), 'b--', linewidth=1.5, label='Baseline', alpha=0.7)
        ax.plot(cols, p_occ_physics[digit].numpy(), 'r:', linewidth=1.5, label='Physics', alpha=0.7)
        
        ax.set_title(f'Digit {digit}', fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Brightness Prob.')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        if digit == 0:
            ax.legend(loc='best', fontsize=8)
    
    plt.suptitle('Occupancy: Column Brightness Probability', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'occupancy.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'occupancy.png')}")
    plt.close()


def plot_smoothness(tau_baseline, tau_physics, tau_real, X_baseline, X_physics, X_real, save_dir):
    """Plot smoothness comparison (gradient distribution)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Compute gradients for all datasets
    def compute_gradients(X):
        delta_h = torch.abs(X[:, :, :, 1:] - X[:, :, :, :-1])
        delta_v = torch.abs(X[:, :, 1:, :] - X[:, :, :-1, :])
        return torch.cat([delta_h.flatten(), delta_v.flatten()]).numpy()
    
    grad_real = compute_gradients(X_real[:1000])  # Subsample for speed
    grad_baseline = compute_gradients(X_baseline[:1000])
    grad_physics = compute_gradients(X_physics[:1000])
    
    # Plot 1: Histogram of gradients
    bins = np.linspace(0, 1, 50)
    ax1.hist(grad_real, bins=bins, alpha=0.5, label='Real', density=True, color='black')
    ax1.hist(grad_baseline, bins=bins, alpha=0.5, label='Baseline', density=True, color='blue')
    ax1.hist(grad_physics, bins=bins, alpha=0.5, label='Physics', density=True, color='red')
    ax1.axvline(tau_real, color='black', linestyle='-', linewidth=2, label=f'Real τ={tau_real:.3f}')
    ax1.axvline(tau_baseline, color='blue', linestyle='--', linewidth=2, label=f'Base τ={tau_baseline:.3f}')
    ax1.axvline(tau_physics, color='red', linestyle=':', linewidth=2, label=f'Phys τ={tau_physics:.3f}')
    ax1.set_xlabel('Gradient Magnitude')
    ax1.set_ylabel('Density')
    ax1.set_title('Gradient Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tau comparison bar chart
    models = ['Real', 'Baseline', 'Physics']
    taus = [tau_real, tau_baseline, tau_physics]
    colors = ['black', 'blue', 'red']
    
    bars = ax2.bar(models, taus, color=colors, alpha=0.6, edgecolor='black')
    ax2.set_ylabel('τ (95th percentile)')
    ax2.set_title('Smoothness Threshold Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, tau in zip(bars, taus):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{tau:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Smoothness: Spatial Gradient Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'smoothness.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'smoothness.png')}")
    plt.close()


def plot_handover(N_mean_baseline, N_mean_physics, N_mean_real, 
                  N_std_baseline, N_std_physics, N_std_real, save_dir):
    """Plot handover comparison (center row transitions)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    digits = np.arange(10)
    width = 0.25
    
    # Plot 1: Mean transitions
    ax1.bar(digits - width, N_mean_real.numpy(), width, label='Real', 
            color='black', alpha=0.6, edgecolor='black')
    ax1.bar(digits, N_mean_baseline.numpy(), width, label='Baseline', 
            color='blue', alpha=0.6, edgecolor='black')
    ax1.bar(digits + width, N_mean_physics.numpy(), width, label='Physics', 
            color='red', alpha=0.6, edgecolor='black')
    
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Mean Transitions')
    ax1.set_title('Mean Center Row Transitions')
    ax1.set_xticks(digits)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Standard deviation
    ax2.bar(digits - width, N_std_real.numpy(), width, label='Real', 
            color='black', alpha=0.6, edgecolor='black')
    ax2.bar(digits, N_std_baseline.numpy(), width, label='Baseline', 
            color='blue', alpha=0.6, edgecolor='black')
    ax2.bar(digits + width, N_std_physics.numpy(), width, label='Physics', 
            color='red', alpha=0.6, edgecolor='black')
    
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Std Dev of Transitions')
    ax2.set_title('Std Dev of Center Row Transitions')
    ax2.set_xticks(digits)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Handover: Center Row Transition Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'handover.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'handover.png')}")
    plt.close()


def plot_summary_metrics(l1_baseline, l1_physics, kl_baseline, kl_physics,
                         tau_diff_baseline, tau_diff_physics, l2_baseline, l2_physics, 
                         fid_baseline, fid_physics, save_dir):
    """Plot summary of all metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    metrics = [
        ('L1 Distance\n(Vert. Atten.)', [l1_baseline, l1_physics], 'lower is better'),
        ('KL Divergence\n(Occupancy)', [kl_baseline, kl_physics], 'lower is better'),
        ('Tau Difference\n(Smoothness)', [tau_diff_baseline, tau_diff_physics], 'lower is better'),
        ('L2 Error\n(Handover)', [l2_baseline, l2_physics], 'lower is better'),
        ('FID Score', [fid_baseline, fid_physics], 'lower is better'),
    ]
    
    for idx, (name, values, note) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        bars = ax.bar(['Baseline', 'Physics'], values, color=['blue', 'red'], 
                     alpha=0.6, edgecolor='black')
        ax.set_ylabel('Value')
        ax.set_title(name, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Add note
        ax.text(0.5, -0.15, note, ha='center', va='top', 
               transform=ax.transAxes, fontsize=8, style='italic', color='gray')
        
        # Highlight better model
        better_idx = 0 if values[0] < values[1] else 1
        bars[better_idx].set_edgecolor('green')
        bars[better_idx].set_linewidth(3)
    
    # Remove the last empty subplot
    axes[1, 2].axis('off')
    
    plt.suptitle('Summary: Physics-Guided vs Baseline Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_metrics.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'summary_metrics.png')}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MNIST DiT models with visualizations"
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples to generate for evaluation (default: 1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directory to save evaluation plots (default: ./evaluation_results)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup
    config = get_config()
    config.IMG_SIZE = 28
    config.PATCH = 7
    set_seed(config.SEED)
    device = get_device()
    
    # Load real data priors
    print("\n" + "="*70)
    print("Loading MNIST priors and real data")
    print("="*70)
    
    priors = load_mnist_priors(config.MNIST_PRIORS_PATH)
    print(f"✓ Loaded priors from: {config.MNIST_PRIORS_PATH}")
    
    # Load real MNIST data for comparison
    dataset_info = load_dataset("mnist", config.WORKDIR, config.SEED)
    X_real = torch.from_numpy(dataset_info.X_test[:args.n_samples]).float()
    y_real = torch.from_numpy(dataset_info.y_test[:args.n_samples]).long()
    print(f"✓ Loaded {len(X_real)} real test samples")
    
    # Initialize DDPM
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    # Load and evaluate baseline model
    print("\n" + "="*70)
    print("Evaluating Baseline Model")
    print("="*70)
    
    model_baseline = load_model_checkpoint("dit_ema_mnist_baseline", config, device)
    X_baseline, y_baseline = generate_samples(model_baseline, ddpm, args.n_samples, device=device)
    
    print("Computing statistics...")
    mu_baseline = compute_vertical_attenuation(X_baseline, y_baseline, 10, 28)
    p_occ_baseline = compute_occupancy(X_baseline, y_baseline, 10, 28)
    tau_baseline = compute_smoothness_threshold(X_baseline)
    N_mean_baseline, N_std_baseline = compute_handover_statistics(X_baseline, y_baseline, 10, 28)
    
    # Calculate FID for baseline
    fid_baseline = calculate_fid(X_real, X_baseline)
    print(f"✓ Baseline FID: {fid_baseline:.4f}")
    
    # Load and evaluate physics-guided model
    print("\n" + "="*70)
    print("Evaluating Physics-Guided Model")
    print("="*70)
    
    model_physics = load_model_checkpoint("dit_ema_mnist_physics", config, device)
    X_physics, y_physics = generate_samples(model_physics, ddpm, args.n_samples, device=device)
    
    print("Computing statistics...")
    mu_physics = compute_vertical_attenuation(X_physics, y_physics, 10, 28)
    p_occ_physics = compute_occupancy(X_physics, y_physics, 10, 28)
    tau_physics = compute_smoothness_threshold(X_physics)
    N_mean_physics, N_std_physics = compute_handover_statistics(X_physics, y_physics, 10, 28)
    
    # Calculate FID for physics
    fid_physics = calculate_fid(X_real, X_physics)
    print(f"✓ Physics FID: {fid_physics:.4f}")
    
    # Compute comparison metrics
    l1_baseline = F.l1_loss(mu_baseline, priors['mu_data']).item()
    l1_physics = F.l1_loss(mu_physics, priors['mu_data']).item()
    
    eps = 1e-8
    def safe_kl(p, q):
        p_safe = torch.clamp(p, eps, 1.0 - eps)
        q_safe = torch.clamp(q, eps, 1.0 - eps)
        return F.kl_div(torch.log(q_safe), p_safe, reduction='batchmean').item()
    
    kl_baseline = safe_kl(priors['p_occ_data'], p_occ_baseline)
    kl_physics = safe_kl(priors['p_occ_data'], p_occ_physics)
    
    tau_diff_baseline = abs(tau_baseline - priors['tau'])
    tau_diff_physics = abs(tau_physics - priors['tau'])
    
    l2_baseline = torch.sqrt(((N_mean_baseline - priors['N_mean']) ** 2).mean()).item()
    l2_physics = torch.sqrt(((N_mean_physics - priors['N_mean']) ** 2).mean()).item()
    
    # Create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations")
    print("="*70)
    
    plot_generated_samples(X_baseline, y_baseline, X_physics, y_physics, 
                          X_real, y_real, args.output_dir)
    
    plot_vertical_attenuation(mu_baseline, mu_physics, priors['mu_data'], args.output_dir)
    
    plot_occupancy(p_occ_baseline, p_occ_physics, priors['p_occ_data'], args.output_dir)
    
    plot_smoothness(tau_baseline, tau_physics, priors['tau'], 
                   X_baseline, X_physics, X_real, args.output_dir)
    
    plot_handover(N_mean_baseline, N_mean_physics, priors['N_mean'],
                 N_std_baseline, N_std_physics, priors['N_std'], args.output_dir)
    
    plot_summary_metrics(l1_baseline, l1_physics, kl_baseline, kl_physics,
                        tau_diff_baseline, tau_diff_physics, l2_baseline, l2_physics,
                        fid_baseline, fid_physics, args.output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nMetric Comparison (Baseline vs Physics):")
    print(f"  FID Score:              {fid_baseline:.4f} vs {fid_physics:.4f}")
    print(f"  L1 (Vert. Atten.):      {l1_baseline:.6f} vs {l1_physics:.6f}")
    print(f"  KL (Occupancy):         {kl_baseline:.6f} vs {kl_physics:.6f}")
    print(f"  Tau Diff (Smoothness):  {tau_diff_baseline:.6f} vs {tau_diff_physics:.6f}")
    print(f"  L2 (Handover):          {l2_baseline:.4f} vs {l2_physics:.4f}")
    
    print(f"\nAll plots saved to: {args.output_dir}/")
    print("  - generated_samples.png")
    print("  - vertical_attenuation.png")
    print("  - occupancy.png")
    print("  - smoothness.png")
    print("  - handover.png")
    print("  - summary_metrics.png")
    print("="*70)


if __name__ == "__main__":
    main()
