#!/usr/bin/env python3
"""
Measure model statistics: parameters, size, and inference time
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import from the main experiment file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configurations (matching dit_data_fidelity_experiment.py)
IMG_SIZE = 16
CHANNELS = 1
PATCH = 4
WIDTH = 256
DEPTH = 4
HEADS = 4
DROP = 0.1
TIMESTEPS = 400

# ========================================
# MODEL DEFINITIONS (from main experiment)
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
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Class embedding (for conditional generation)
        self.class_embed = nn.Embedding(num_classes, d_model)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Output projection
        self.final_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_size * patch_size * in_channels)
        )
        
    def forward(self, x, t, y=None):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Time embedding
        t_emb = self.get_time_embedding(t, self.d_model).to(x.device)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)  # (B, 1, d_model)
        
        # Class embedding (if provided)
        if y is not None:
            c_emb = self.class_embed(y).unsqueeze(1)  # (B, 1, d_model)
            x = x + t_emb + c_emb
        else:
            x = x + t_emb
        
        # Transformer
        x = self.transformer(x)
        
        # Output projection
        x = self.final_layer(x)  # (B, num_patches, patch_size^2 * channels)
        
        # Reshape to image
        x = x.view(B, self.num_patches, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3)  # (B, channels, num_patches, patch_size, patch_size)
        
        # Reconstruct image from patches
        H = W = self.img_size // self.patch_size
        x = x.reshape(B, -1, H, W, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, C, H, patch_size, W, patch_size)
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
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, latent_dim)
        
        # Generator network
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),  # *2 for noise + class embedding
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden, num_features),
            nn.Tanh()  # Output in [-1, 1] range like standardized data
        )
    
    def forward(self, z, labels):
        # Embed class labels
        label_emb = self.class_embed(labels)
        # Concatenate noise and label embedding
        x = torch.cat([z, label_emb], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """Conditional Discriminator for RSS data"""
    def __init__(self, num_features=7, num_classes=4, hidden=128):
        super().__init__()
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, num_features)
        
        # Discriminator network
        self.net = nn.Sequential(
            nn.Linear(num_features * 2, hidden),  # *2 for data + class embedding
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
        # Embed class labels
        label_emb = self.class_embed(labels)
        # Concatenate data and label embedding
        x_concat = torch.cat([x, label_emb], dim=1)
        return self.net(x_concat)


class MLP(nn.Module):
    """Simple MLP for downstream classification"""
    def __init__(self, in_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


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


def measure_inference_time(model, input_data, num_runs=100, warmup=10):
    """Measure inference time per sample"""
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*input_data) if isinstance(input_data, tuple) else model(input_data)
    
    # Actual timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            
            _ = model(*input_data) if isinstance(input_data, tuple) else model(input_data)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            times.append(end - start)
    
    # Return average time per sample
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


def measure_dit_inference_per_sample(dit_model, num_classes=4, timesteps=TIMESTEPS, num_runs=50):
    """Measure DiT inference time for generating one sample"""
    dit_model.eval()
    
    # Simulate DDPM sampling for 1 sample (full denoising process)
    times = []
    
    for _ in range(num_runs):
        x = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
        y = torch.randint(0, num_classes, (1,), device=device)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        
        with torch.no_grad():
            # Full denoising loop (this is what happens during generation)
            for t_idx in reversed(range(timesteps)):
                t = torch.full((1,), t_idx, dtype=torch.long, device=device)
                _ = dit_model(x, t, y=y)
                # Note: We skip the actual denoising math to just measure model forward passes
        
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
    
    avg_time, std_time = measure_inference_time(generator, (z, y), num_runs=num_runs)
    return avg_time, std_time


# ========================================
# MAIN MEASUREMENT
# ========================================

def main():
    print(f"\n{'='*70}")
    print("MODEL STATISTICS MEASUREMENT")
    print(f"{'='*70}\n")
    
    results = []
    
    # ========================================
    # UCI Indoor Configuration (7 features, 4 classes)
    # ========================================
    print("Measuring UCI Indoor models (7 features, 4 classes)...")
    
    # DiT
    print("\n[1/4] DiT Model...")
    dit_uci = DiT(num_classes=4).to(device)
    dit_params, dit_trainable = count_parameters(dit_uci)
    dit_size = get_model_size_mb(dit_uci)
    dit_time, dit_std = measure_dit_inference_per_sample(dit_uci, num_classes=4, num_runs=20)
    
    print(f"  Parameters: {dit_params:,} ({dit_trainable:,} trainable)")
    print(f"  Size: {dit_size:.2f} MB")
    print(f"  Inference time: {dit_time*1000:.2f} ± {dit_std*1000:.2f} ms/sample")
    
    results.append({
        'Dataset': 'UCI Indoor',
        'Method': 'DiT',
        'Total Parameters': dit_params,
        'Trainable Parameters': dit_trainable,
        'Model Size (MB)': dit_size,
        'Inference Time (ms/sample)': dit_time * 1000,
        'Inference Std (ms)': dit_std * 1000
    })
    
    # GAN Generator
    print("\n[2/4] GAN Generator...")
    gan_gen_uci = Generator(latent_dim=100, num_features=7, num_classes=4, hidden=128).to(device)
    gan_params, gan_trainable = count_parameters(gan_gen_uci)
    gan_size = get_model_size_mb(gan_gen_uci)
    gan_time, gan_std = measure_gan_inference_per_sample(gan_gen_uci, latent_dim=100, num_classes=4)
    
    print(f"  Parameters: {gan_params:,} ({gan_trainable:,} trainable)")
    print(f"  Size: {gan_size:.2f} MB")
    print(f"  Inference time: {gan_time*1000:.2f} ± {gan_std*1000:.2f} ms/sample")
    
    results.append({
        'Dataset': 'UCI Indoor',
        'Method': 'GAN (Generator)',
        'Total Parameters': gan_params,
        'Trainable Parameters': gan_trainable,
        'Model Size (MB)': gan_size,
        'Inference Time (ms/sample)': gan_time * 1000,
        'Inference Std (ms)': gan_std * 1000
    })
    
    # GAN Discriminator
    print("\n[3/4] GAN Discriminator...")
    gan_disc_uci = Discriminator(num_features=7, num_classes=4, hidden=128).to(device)
    disc_params, disc_trainable = count_parameters(gan_disc_uci)
    disc_size = get_model_size_mb(gan_disc_uci)
    
    # Measure discriminator inference
    x = torch.randn(1, 7, device=device)
    y = torch.randint(0, 4, (1,), device=device)
    disc_time, disc_std = measure_inference_time(gan_disc_uci, (x, y))
    
    print(f"  Parameters: {disc_params:,} ({disc_trainable:,} trainable)")
    print(f"  Size: {disc_size:.2f} MB")
    print(f"  Inference time: {disc_time*1000:.2f} ± {disc_std*1000:.2f} ms/sample")
    
    results.append({
        'Dataset': 'UCI Indoor',
        'Method': 'GAN (Discriminator)',
        'Total Parameters': disc_params,
        'Trainable Parameters': disc_trainable,
        'Model Size (MB)': disc_size,
        'Inference Time (ms/sample)': disc_time * 1000,
        'Inference Std (ms)': disc_std * 1000
    })
    
    # MLP Classifier
    print("\n[4/4] MLP Classifier...")
    mlp_uci = MLP(in_dim=7, num_classes=4, hidden=256).to(device)
    mlp_params, mlp_trainable = count_parameters(mlp_uci)
    mlp_size = get_model_size_mb(mlp_uci)
    
    x_mlp = torch.randn(1, 7, device=device)
    mlp_time, mlp_std = measure_inference_time(mlp_uci, x_mlp)
    
    print(f"  Parameters: {mlp_params:,} ({mlp_trainable:,} trainable)")
    print(f"  Size: {mlp_size:.2f} MB")
    print(f"  Inference time: {mlp_time*1000:.2f} ± {mlp_std*1000:.2f} ms/sample")
    
    results.append({
        'Dataset': 'UCI Indoor',
        'Method': 'MLP Classifier',
        'Total Parameters': mlp_params,
        'Trainable Parameters': mlp_trainable,
        'Model Size (MB)': mlp_size,
        'Inference Time (ms/sample)': mlp_time * 1000,
        'Inference Std (ms)': mlp_std * 1000
    })
    
    # ========================================
    # POWDER Outdoor Configuration (25 features, 16 classes)
    # ========================================
    print("\n" + "="*70)
    print("Measuring POWDER Outdoor models (25 features, 16 classes)...")
    
    # DiT (same architecture, different num_classes)
    print("\n[1/4] DiT Model...")
    dit_powder = DiT(num_classes=16).to(device)
    dit_params_p, dit_trainable_p = count_parameters(dit_powder)
    dit_size_p = get_model_size_mb(dit_powder)
    dit_time_p, dit_std_p = measure_dit_inference_per_sample(dit_powder, num_classes=16, num_runs=20)
    
    print(f"  Parameters: {dit_params_p:,} ({dit_trainable_p:,} trainable)")
    print(f"  Size: {dit_size_p:.2f} MB")
    print(f"  Inference time: {dit_time_p*1000:.2f} ± {dit_std_p*1000:.2f} ms/sample")
    
    results.append({
        'Dataset': 'POWDER Outdoor',
        'Method': 'DiT',
        'Total Parameters': dit_params_p,
        'Trainable Parameters': dit_trainable_p,
        'Model Size (MB)': dit_size_p,
        'Inference Time (ms/sample)': dit_time_p * 1000,
        'Inference Std (ms)': dit_std_p * 1000
    })
    
    # GAN Generator
    print("\n[2/4] GAN Generator...")
    gan_gen_powder = Generator(latent_dim=100, num_features=25, num_classes=16, hidden=128).to(device)
    gan_params_p, gan_trainable_p = count_parameters(gan_gen_powder)
    gan_size_p = get_model_size_mb(gan_gen_powder)
    gan_time_p, gan_std_p = measure_gan_inference_per_sample(gan_gen_powder, latent_dim=100, num_classes=16)
    
    print(f"  Parameters: {gan_params_p:,} ({gan_trainable_p:,} trainable)")
    print(f"  Size: {gan_size_p:.2f} MB")
    print(f"  Inference time: {gan_time_p*1000:.2f} ± {gan_std_p*1000:.2f} ms/sample")
    
    results.append({
        'Dataset': 'POWDER Outdoor',
        'Method': 'GAN (Generator)',
        'Total Parameters': gan_params_p,
        'Trainable Parameters': gan_trainable_p,
        'Model Size (MB)': gan_size_p,
        'Inference Time (ms/sample)': gan_time_p * 1000,
        'Inference Std (ms)': gan_std_p * 1000
    })
    
    # GAN Discriminator
    print("\n[3/4] GAN Discriminator...")
    gan_disc_powder = Discriminator(num_features=25, num_classes=16, hidden=128).to(device)
    disc_params_p, disc_trainable_p = count_parameters(gan_disc_powder)
    disc_size_p = get_model_size_mb(gan_disc_powder)
    
    x_p = torch.randn(1, 25, device=device)
    y_p = torch.randint(0, 16, (1,), device=device)
    disc_time_p, disc_std_p = measure_inference_time(gan_disc_powder, (x_p, y_p))
    
    print(f"  Parameters: {disc_params_p:,} ({disc_trainable_p:,} trainable)")
    print(f"  Size: {disc_size_p:.2f} MB")
    print(f"  Inference time: {disc_time_p*1000:.2f} ± {disc_std_p*1000:.2f} ms/sample")
    
    results.append({
        'Dataset': 'POWDER Outdoor',
        'Method': 'GAN (Discriminator)',
        'Total Parameters': disc_params_p,
        'Trainable Parameters': disc_trainable_p,
        'Model Size (MB)': disc_size_p,
        'Inference Time (ms/sample)': disc_time_p * 1000,
        'Inference Std (ms)': disc_std_p * 1000
    })
    
    # MLP Classifier
    print("\n[4/4] MLP Classifier...")
    mlp_powder = MLP(in_dim=25, num_classes=16, hidden=256).to(device)
    mlp_params_p, mlp_trainable_p = count_parameters(mlp_powder)
    mlp_size_p = get_model_size_mb(mlp_powder)
    
    x_mlp_p = torch.randn(1, 25, device=device)
    mlp_time_p, mlp_std_p = measure_inference_time(mlp_powder, x_mlp_p)
    
    print(f"  Parameters: {mlp_params_p:,} ({mlp_trainable_p:,} trainable)")
    print(f"  Size: {mlp_size_p:.2f} MB")
    print(f"  Inference time: {mlp_time_p*1000:.2f} ± {mlp_std_p*1000:.2f} ms/sample")
    
    results.append({
        'Dataset': 'POWDER Outdoor',
        'Method': 'MLP Classifier',
        'Total Parameters': mlp_params_p,
        'Trainable Parameters': mlp_trainable_p,
        'Model Size (MB)': mlp_size_p,
        'Inference Time (ms/sample)': mlp_time_p * 1000,
        'Inference Std (ms)': mlp_std_p * 1000
    })
    
    # ========================================
    # Save results
    # ========================================
    df = pd.DataFrame(results)
    
    # Reorder columns
    df = df[['Dataset', 'Method', 'Total Parameters', 'Trainable Parameters', 
             'Model Size (MB)', 'Inference Time (ms/sample)', 'Inference Std (ms)']]
    
    output_path = "./work_dir/model_statistics.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Create a clean table for presentation
    print(f"\n{'='*70}")
    print("CLEAN TABLE (for paper/presentation)")
    print(f"{'='*70}\n")
    
    # Format for better readability
    clean_df = df.copy()
    clean_df['Parameters'] = clean_df['Total Parameters'].apply(lambda x: f"{x:,}")
    clean_df['Size'] = clean_df['Model Size (MB)'].apply(lambda x: f"{x:.2f} MB")
    clean_df['Inference Time'] = clean_df.apply(
        lambda row: f"{row['Inference Time (ms/sample)']:.2f} ± {row['Inference Std (ms)']:.2f} ms",
        axis=1
    )
    
    clean_table = clean_df[['Dataset', 'Method', 'Parameters', 'Size', 'Inference Time']]
    print(clean_table.to_string(index=False))
    
    # Save clean table
    clean_output_path = "./work_dir/model_statistics_clean.csv"
    clean_table.to_csv(clean_output_path, index=False)
    print(f"\n✓ Clean table saved to: {clean_output_path}")
    
    print("\n✓ Model statistics measurement complete!")


if __name__ == "__main__":
    main()
