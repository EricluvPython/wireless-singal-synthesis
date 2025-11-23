#!/usr/bin/env python3
"""
Measure model statistics for the 3 main generative methods: DiT, GAN, Kriging
Focused on UCI Indoor dataset only
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configurations
IMG_SIZE = 16
CHANNELS = 1
PATCH = 4
WIDTH = 256
DEPTH = 4
HEADS = 4
DROP = 0.1
TIMESTEPS = 400

# ========================================
# MODEL DEFINITIONS
# ========================================

class DiT(nn.Module):
    """Diffusion Transformer for conditional generation"""
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH, in_channels=CHANNELS,
                 d_model=WIDTH, depth=DEPTH, num_heads=HEADS, 
                 num_classes=4, dropout=DROP):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.d_model = d_model
        
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.class_embed = nn.Embedding(num_classes, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_size * patch_size * in_channels)
        )
        
    def forward(self, x, t, y=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        t_emb = self.get_time_embedding(t, self.d_model).to(x.device)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)
        
        if y is not None:
            c_emb = self.class_embed(y).unsqueeze(1)
            x = x + t_emb + c_emb
        else:
            x = x + t_emb
        
        x = self.transformer(x)
        x = self.final_layer(x)
        
        x = x.view(B, self.num_patches, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3)
        
        H = W = self.img_size // self.patch_size
        x = x.reshape(B, -1, H, W, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(B, -1, self.img_size, self.img_size)
        
        return x
    
    def get_time_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class Generator(nn.Module):
    """Conditional Generator for RSS data"""
    def __init__(self, latent_dim=100, num_features=7, num_classes=4, hidden=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        
        self.class_embed = nn.Embedding(num_classes, latent_dim)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, num_features),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_emb = self.class_embed(labels)
        x = torch.cat([z, label_emb], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """Conditional Discriminator for RSS data"""
    def __init__(self, num_features=7, num_classes=4, hidden=128):
        super().__init__()
        
        self.class_embed = nn.Embedding(num_classes, num_features)
        
        self.net = nn.Sequential(
            nn.Linear(num_features * 2, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        label_emb = self.class_embed(labels)
        x_concat = torch.cat([x, label_emb], dim=1)
        return self.net(x_concat)


# ========================================
# MEASUREMENT FUNCTIONS
# ========================================

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def measure_dit_inference_per_sample(dit_model, num_classes=4, timesteps=TIMESTEPS, num_runs=50):
    """Measure DiT inference time for generating one sample (full diffusion process)"""
    dit_model.eval()
    
    times = []
    
    for _ in range(num_runs):
        x = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
        y = torch.randint(0, num_classes, (1,), device=device)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            # Full denoising loop (400 timesteps)
            for t_idx in reversed(range(timesteps)):
                t = torch.full((1,), t_idx, dtype=torch.long, device=device)
                _ = dit_model(x, t, y=y)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


def measure_gan_inference_per_sample(generator, latent_dim=100, num_classes=4, num_runs=100):
    """Measure GAN inference time for generating one sample"""
    generator.eval()
    
    z = torch.randn(1, latent_dim, device=device)
    y = torch.randint(0, num_classes, (1,), device=device)
    
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            _ = generator(z, y)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


def measure_kriging_inference_per_sample(num_features=7, num_runs=100):
    """Measure Kriging (RBF interpolation) inference time for generating one sample"""
    # Simulate kriging with 20 reference samples per class
    num_ref_samples = 20
    X_ref = np.random.randn(num_ref_samples, num_features).astype(np.float32)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        
        # Kriging: select 2 random samples and interpolate
        idx1, idx2 = np.random.choice(num_ref_samples, 2, replace=True)
        alpha = np.random.rand()
        interpolated = alpha * X_ref[idx1] + (1 - alpha) * X_ref[idx2]
        interpolated += np.random.normal(0, 0.05, num_features)
        
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


# ========================================
# MAIN MEASUREMENT
# ========================================

def main():
    print(f"\n{'='*70}")
    print("MODEL STATISTICS - UCI Indoor Dataset")
    print("Generative Methods: DiT, GAN, Kriging")
    print(f"{'='*70}\n")
    
    results = []
    
    # ========================================
    # 1. DiT
    # ========================================
    print("[1/3] DiT Model...")
    dit_model = DiT(num_classes=4).to(device)
    dit_params, dit_trainable = count_parameters(dit_model)
    dit_size = get_model_size_mb(dit_model)
    dit_time, dit_std = measure_dit_inference_per_sample(dit_model, num_classes=4, num_runs=20)
    
    print(f"  Parameters: {dit_params:,}")
    print(f"  Model Size: {dit_size:.2f} MB")
    print(f"  Inference Time: {dit_time*1000:.2f} ± {dit_std*1000:.2f} ms/sample")
    
    results.append({
        'Method': 'DiT',
        'Parameters': f"{dit_params:,}",
        'Model Size (MB)': f"{dit_size:.2f}",
        'Inference Time (ms/sample)': f"{dit_time*1000:.2f} ± {dit_std*1000:.2f}"
    })
    
    # ========================================
    # 2. GAN (Generator + Discriminator combined)
    # ========================================
    print("\n[2/3] GAN (Generator + Discriminator)...")
    generator = Generator(latent_dim=100, num_features=7, num_classes=4, hidden=128).to(device)
    discriminator = Discriminator(num_features=7, num_classes=4, hidden=128).to(device)
    
    gen_params, _ = count_parameters(generator)
    disc_params, _ = count_parameters(discriminator)
    gan_total_params = gen_params + disc_params
    
    gen_size = get_model_size_mb(generator)
    disc_size = get_model_size_mb(discriminator)
    gan_total_size = gen_size + disc_size
    
    # For inference time, we only care about generator (used for synthesis)
    gan_time, gan_std = measure_gan_inference_per_sample(generator, latent_dim=100, num_classes=4)
    
    print(f"  Generator Parameters: {gen_params:,}")
    print(f"  Discriminator Parameters: {disc_params:,}")
    print(f"  Total Parameters: {gan_total_params:,}")
    print(f"  Total Model Size: {gan_total_size:.2f} MB")
    print(f"  Inference Time (Generator): {gan_time*1000:.2f} ± {gan_std*1000:.2f} ms/sample")
    
    results.append({
        'Method': 'GAN',
        'Parameters': f"{gan_total_params:,}",
        'Model Size (MB)': f"{gan_total_size:.2f}",
        'Inference Time (ms/sample)': f"{gan_time*1000:.2f} ± {gan_std*1000:.2f}"
    })
    
    # ========================================
    # 3. Kriging (Statistical Interpolation)
    # ========================================
    print("\n[3/3] Kriging (Statistical Interpolation)...")
    kriging_time, kriging_std = measure_kriging_inference_per_sample(num_features=7, num_runs=100)
    
    print(f"  Parameters: N/A (non-parametric)")
    print(f"  Model Size: N/A")
    print(f"  Inference Time: {kriging_time*1000:.4f} ± {kriging_std*1000:.4f} ms/sample")
    
    results.append({
        'Method': 'Kriging',
        'Parameters': '—',
        'Model Size (MB)': '—',
        'Inference Time (ms/sample)': f"{kriging_time*1000:.4f} ± {kriging_std*1000:.4f}"
    })
    
    # ========================================
    # Create and save table
    # ========================================
    df = pd.DataFrame(results)
    
    output_path = "./work_dir/model_comparison_table.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))
    
    print(f"\n✓ Table saved to: {output_path}")
    
    # Also create a LaTeX-friendly version
    print(f"\n{'='*70}")
    print("LaTeX FORMAT")
    print(f"{'='*70}\n")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Method & Parameters & Model Size (MB) & Inference Time (ms/sample) \\\\")
    print("\\hline")
    for _, row in df.iterrows():
        print(f"{row['Method']} & {row['Parameters']} & {row['Model Size (MB)']} & {row['Inference Time (ms/sample)']} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Model complexity and inference time comparison for generative methods on UCI Indoor dataset.}")
    print("\\label{tab:model_comparison}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
