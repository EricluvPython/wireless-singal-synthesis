#!/usr/bin/env python3
"""
Evaluate UniCellular DiT models with comprehensive visualizations

This script:
1. Loads trained models (baseline and physics-guided)
2. Generates synthetic RSS fingerprints
3. Computes and visualizes physics-prior statistics
4. Compares baseline vs physics-guided models
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
from unicellular_stats import (
    compute_vertical_attenuation,
    compute_occupancy,
    compute_smoothness_threshold,
    compute_handover_statistics,
    load_unicellular_priors
)
from utils import set_seed, get_device
import torch.nn.functional as F

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_model_checkpoint(checkpoint_name, config, device, num_classes, img_size, patch_size):
    """Load a trained DiT model from checkpoint"""
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{checkpoint_name}_best.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train the model first."
        )
    
    # Initialize model
    model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=num_classes,
        dropout=config.DROP
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return model


def generate_samples(model, ddpm, n_samples, num_classes, num_features, img_size, 
                    batch_size=100, device='cuda'):
    """Generate synthetic RSS fingerprints from trained DiT model"""
    print(f"\nGenerating {n_samples} synthetic samples...")
    
    all_samples = []
    all_labels = []
    
    # Generate samples in batches
    for i in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - i)
        
        # Sample labels uniformly across floors
        labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
        
        # Generate samples (will be in image format)
        samples_img = ddpm.sample(
            model,
            n=current_batch_size,
            img_size=img_size,
            channels=1,
            y=labels
        )
        
        # Reshape from image back to RSS vector: (N, 1, H, W) -> (N, H*W)
        samples_flat = samples_img.reshape(current_batch_size, -1)
        
        # Extract only the original features (remove padding)
        samples_rss = samples_flat[:, :num_features]
        
        all_samples.append(samples_rss.cpu())
        all_labels.append(labels.cpu())
        
        if (i + current_batch_size) % 500 == 0:
            print(f"  Generated {i + current_batch_size}/{n_samples}")
    
    # Concatenate all batches
    X_gen = torch.cat(all_samples, dim=0)
    y_gen = torch.cat(all_labels, dim=0)
    
    print(f"✓ Generated {len(X_gen)} samples")
    print(f"  RSS range: [{X_gen.min():.2f}, {X_gen.max():.2f}]")
    
    return X_gen, y_gen


def calculate_fid(real_rss, generated_rss):
    """
    Calculate Fréchet Inception Distance (FID) for RSS fingerprints
    Based on feature statistics
    """
    real_flat = real_rss.numpy()
    gen_flat = generated_rss.numpy()
    
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
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_gen + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    
    return fid


def plot_rss_heatmaps(X_baseline, y_baseline, X_physics, y_physics, X_real, y_real, 
                     num_classes, num_features, save_dir):
    """Plot RSS heatmaps for real vs generated data"""
    fig, axes = plt.subplots(3, num_classes, figsize=(2*num_classes, 6))
    
    if num_classes == 1:
        axes = axes[:, np.newaxis]
    
    for floor in range(num_classes):
        # Real
        real_idx = (y_real == floor).nonzero(as_tuple=True)[0]
        if len(real_idx) > 0:
            real_sample = X_real[real_idx[0]].numpy().reshape(1, -1)
            im = axes[0, floor].imshow(real_sample, cmap='viridis', aspect='auto', 
                                       vmin=-3, vmax=3)
            axes[0, floor].set_yticks([])
            axes[0, floor].set_xlabel('Cell Tower')
            if floor == 0:
                axes[0, floor].set_ylabel('Real', fontsize=10)
            axes[0, floor].set_title(f'Floor {floor}', fontsize=9)
        
        # Baseline
        baseline_idx = (y_baseline == floor).nonzero(as_tuple=True)[0]
        if len(baseline_idx) > 0:
            baseline_sample = X_baseline[baseline_idx[0]].numpy().reshape(1, -1)
            axes[1, floor].imshow(baseline_sample, cmap='viridis', aspect='auto',
                                 vmin=-3, vmax=3)
            axes[1, floor].set_yticks([])
            axes[1, floor].set_xlabel('Cell Tower')
            if floor == 0:
                axes[1, floor].set_ylabel('Baseline', fontsize=10)
        
        # Physics
        physics_idx = (y_physics == floor).nonzero(as_tuple=True)[0]
        if len(physics_idx) > 0:
            physics_sample = X_physics[physics_idx[0]].numpy().reshape(1, -1)
            axes[2, floor].imshow(physics_sample, cmap='viridis', aspect='auto',
                                 vmin=-3, vmax=3)
            axes[2, floor].set_yticks([])
            axes[2, floor].set_xlabel('Cell Tower')
            if floor == 0:
                axes[2, floor].set_ylabel('Physics', fontsize=10)
    
    # Add colorbar
    plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Normalized RSS')
    
    plt.savefig(os.path.join(save_dir, 'rss_heatmaps.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'rss_heatmaps.png')}")
    plt.close()


def plot_vertical_attenuation(mu_baseline, mu_physics, mu_real, num_classes, save_dir):
    """Plot floor-wise mean RSS per cell tower"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    num_features = mu_real.shape[1]
    cell_indices = np.arange(num_features)
    
    for floor in range(min(num_classes, 9)):
        ax = axes[floor]
        
        ax.plot(cell_indices, mu_real[floor].numpy(), 'k-', linewidth=2, 
               label='Real', alpha=0.8, marker='o')
        ax.plot(cell_indices, mu_baseline[floor].numpy(), 'b--', linewidth=1.5, 
               label='Baseline', alpha=0.7, marker='s')
        ax.plot(cell_indices, mu_physics[floor].numpy(), 'r:', linewidth=1.5, 
               label='Physics', alpha=0.7, marker='^')
        
        ax.set_title(f'Floor {floor}', fontweight='bold')
        ax.set_xlabel('Cell Tower ID')
        ax.set_ylabel('Mean RSS (normalized)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_classes, 9):
        axes[i].axis('off')
    
    plt.suptitle('Vertical Attenuation: Floor-wise Mean RSS per Cell Tower', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vertical_attenuation.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'vertical_attenuation.png')}")
    plt.close()


def plot_occupancy(p_occ_baseline, p_occ_physics, p_occ_real, num_classes, save_dir):
    """Plot cell tower visibility probability per floor"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    num_features = p_occ_real.shape[1]
    cell_indices = np.arange(num_features)
    
    for floor in range(min(num_classes, 9)):
        ax = axes[floor]
        
        ax.plot(cell_indices, p_occ_real[floor].numpy(), 'k-', linewidth=2, 
               label='Real', alpha=0.8, marker='o')
        ax.plot(cell_indices, p_occ_baseline[floor].numpy(), 'b--', linewidth=1.5, 
               label='Baseline', alpha=0.7, marker='s')
        ax.plot(cell_indices, p_occ_physics[floor].numpy(), 'r:', linewidth=1.5, 
               label='Physics', alpha=0.7, marker='^')
        
        ax.set_title(f'Floor {floor}', fontweight='bold')
        ax.set_xlabel('Cell Tower ID')
        ax.set_ylabel('Visibility Probability')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_classes, 9):
        axes[i].axis('off')
    
    plt.suptitle('Occupancy: Cell Tower Visibility Probability per Floor', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'occupancy.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'occupancy.png')}")
    plt.close()


def plot_smoothness(tau_baseline, tau_physics, tau_real, X_baseline, X_physics, X_real, save_dir):
    """Plot RSS variation comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Compute variations
    def compute_variations(X):
        X_sorted = torch.sort(X, dim=1, descending=True)[0]
        deltas = torch.abs(X_sorted[:, 1:] - X_sorted[:, :-1])
        return deltas.flatten().numpy()
    
    var_real = compute_variations(X_real[:1000])
    var_baseline = compute_variations(X_baseline[:1000])
    var_physics = compute_variations(X_physics[:1000])
    
    # Plot 1: Histogram
    bins = np.linspace(0, 3, 50)
    ax1.hist(var_real, bins=bins, alpha=0.5, label='Real', density=True, color='black')
    ax1.hist(var_baseline, bins=bins, alpha=0.5, label='Baseline', density=True, color='blue')
    ax1.hist(var_physics, bins=bins, alpha=0.5, label='Physics', density=True, color='red')
    ax1.axvline(tau_real, color='black', linestyle='-', linewidth=2, label=f'Real τ={tau_real:.3f}')
    ax1.axvline(tau_baseline, color='blue', linestyle='--', linewidth=2, label=f'Base τ={tau_baseline:.3f}')
    ax1.axvline(tau_physics, color='red', linestyle=':', linewidth=2, label=f'Phys τ={tau_physics:.3f}')
    ax1.set_xlabel('RSS Variation')
    ax1.set_ylabel('Density')
    ax1.set_title('RSS Variation Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tau comparison
    models = ['Real', 'Baseline', 'Physics']
    taus = [tau_real, tau_baseline, tau_physics]
    colors = ['black', 'blue', 'red']
    
    bars = ax2.bar(models, taus, color=colors, alpha=0.6, edgecolor='black')
    ax2.set_ylabel('τ (95th percentile)')
    ax2.set_title('Smoothness Threshold Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, tau in zip(bars, taus):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{tau:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Smoothness: RSS Variation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'smoothness.png'), bbox_inches='tight', dpi=150)
    print(f"✓ Saved: {os.path.join(save_dir, 'smoothness.png')}")
    plt.close()


def plot_handover(N_mean_baseline, N_mean_physics, N_mean_real,
                 N_std_baseline, N_std_physics, N_std_real, num_classes, save_dir):
    """Plot floor transition statistics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    floors = np.arange(num_classes)
    width = 0.25
    
    # Plot 1: Mean transitions
    ax1.bar(floors - width, N_mean_real.numpy(), width, label='Real',
           color='black', alpha=0.6, edgecolor='black')
    ax1.bar(floors, N_mean_baseline.numpy(), width, label='Baseline',
           color='blue', alpha=0.6, edgecolor='black')
    ax1.bar(floors + width, N_mean_physics.numpy(), width, label='Physics',
           color='red', alpha=0.6, edgecolor='black')
    
    ax1.set_xlabel('Floor')
    ax1.set_ylabel('Mean Transitions')
    ax1.set_title('Mean RSS Transitions per Floor')
    ax1.set_xticks(floors)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Std dev
    ax2.bar(floors - width, N_std_real.numpy(), width, label='Real',
           color='black', alpha=0.6, edgecolor='black')
    ax2.bar(floors, N_std_baseline.numpy(), width, label='Baseline',
           color='blue', alpha=0.6, edgecolor='black')
    ax2.bar(floors + width, N_std_physics.numpy(), width, label='Physics',
           color='red', alpha=0.6, edgecolor='black')
    
    ax2.set_xlabel('Floor')
    ax2.set_ylabel('Std Dev of Transitions')
    ax2.set_title('Std Dev of RSS Transitions')
    ax2.set_xticks(floors)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Handover: Floor Transition Statistics', fontsize=14, fontweight='bold')
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
        description="Evaluate UniCellular DiT models with visualizations"
    )
    parser.add_argument(
        '--building',
        choices=['deeb', 'alexu'],
        default='deeb',
        help='Building to evaluate (default: deeb)'
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
        default='./evaluation_results_unicellular',
        help='Directory to save evaluation plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = f"{args.output_dir}_{args.building}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup
    config = get_config()
    set_seed(config.SEED)
    device = get_device()
    
    # Load dataset
    print("\n" + "="*70)
    print(f"Loading UniCellular Dataset ({args.building.upper()})")
    print("="*70)
    
    dataset_info = load_dataset(f"unicellular_{args.building}", config.WORKDIR, config.SEED)
    num_features = dataset_info.num_features
    num_classes = dataset_info.num_classes
    
    # Calculate image size
    import math
    img_size = int(math.ceil(math.sqrt(num_features)))
    if img_size % 2 == 1:
        img_size += 1
    patch_size = img_size // 4
    if patch_size == 0:
        patch_size = 1
    
    print(f"\nDataset: {dataset_info.name}")
    print(f"  Features: {num_features} cell towers")
    print(f"  Classes: {num_classes} floors")
    print(f"  Image size: {img_size}x{img_size} (patch: {patch_size})")
    
    # Load priors
    priors_path = os.path.join(config.WORKDIR, f"unicellular_{args.building}_priors.pt")
    priors = load_unicellular_priors(priors_path)
    print(f"✓ Loaded priors from: {priors_path}")
    
    # Load real data
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
    
    baseline_name = f"dit_ema_unicellular_{args.building}_baseline"
    model_baseline = load_model_checkpoint(baseline_name, config, device, 
                                          num_classes, img_size, patch_size)
    X_baseline, y_baseline = generate_samples(model_baseline, ddpm, args.n_samples,
                                              num_classes, num_features, img_size, device=device)
    
    print("Computing statistics...")
    mu_baseline = compute_vertical_attenuation(X_baseline, y_baseline, num_classes, num_features)
    p_occ_baseline = compute_occupancy(X_baseline, y_baseline, num_classes, num_features)
    tau_baseline = compute_smoothness_threshold(X_baseline)
    N_mean_baseline, N_std_baseline = compute_handover_statistics(X_baseline, y_baseline, num_classes)
    
    fid_baseline = calculate_fid(X_real, X_baseline)
    print(f"✓ Baseline FID: {fid_baseline:.4f}")
    
    # Load and evaluate physics-guided model
    print("\n" + "="*70)
    print("Evaluating Physics-Guided Model")
    print("="*70)
    
    physics_name = f"dit_ema_unicellular_{args.building}_physics"
    model_physics = load_model_checkpoint(physics_name, config, device,
                                         num_classes, img_size, patch_size)
    X_physics, y_physics = generate_samples(model_physics, ddpm, args.n_samples,
                                           num_classes, num_features, img_size, device=device)
    
    print("Computing statistics...")
    mu_physics = compute_vertical_attenuation(X_physics, y_physics, num_classes, num_features)
    p_occ_physics = compute_occupancy(X_physics, y_physics, num_classes, num_features)
    tau_physics = compute_smoothness_threshold(X_physics)
    N_mean_physics, N_std_physics = compute_handover_statistics(X_physics, y_physics, num_classes)
    
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
    
    plot_rss_heatmaps(X_baseline, y_baseline, X_physics, y_physics,
                     X_real, y_real, num_classes, num_features, output_dir)
    
    plot_vertical_attenuation(mu_baseline, mu_physics, priors['mu_data'], 
                             num_classes, output_dir)
    
    plot_occupancy(p_occ_baseline, p_occ_physics, priors['p_occ_data'],
                  num_classes, output_dir)
    
    plot_smoothness(tau_baseline, tau_physics, priors['tau'],
                   X_baseline, X_physics, X_real, output_dir)
    
    plot_handover(N_mean_baseline, N_mean_physics, priors['N_mean'],
                 N_std_baseline, N_std_physics, priors['N_std'],
                 num_classes, output_dir)
    
    plot_summary_metrics(l1_baseline, l1_physics, kl_baseline, kl_physics,
                        tau_diff_baseline, tau_diff_physics, l2_baseline, l2_physics,
                        fid_baseline, fid_physics, output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\nBuilding: {args.building.upper()}")
    print(f"Metric Comparison (Baseline vs Physics):")
    print(f"  FID Score:              {fid_baseline:.4f} vs {fid_physics:.4f}")
    print(f"  L1 (Vert. Atten.):      {l1_baseline:.6f} vs {l1_physics:.6f}")
    print(f"  KL (Occupancy):         {kl_baseline:.6f} vs {kl_physics:.6f}")
    print(f"  Tau Diff (Smoothness):  {tau_diff_baseline:.6f} vs {tau_diff_physics:.6f}")
    print(f"  L2 (Handover):          {l2_baseline:.4f} vs {l2_physics:.4f}")
    
    print(f"\nAll plots saved to: {output_dir}/")
    print("  - rss_heatmaps.png")
    print("  - vertical_attenuation.png")
    print("  - occupancy.png")
    print("  - smoothness.png")
    print("  - handover.png")
    print("  - summary_metrics.png")
    print("="*70)


if __name__ == "__main__":
    main()
