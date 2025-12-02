#!/usr/bin/env python3
"""
DiT Data Fidelity Experiment (Replicating GAN Paper Methodology)

Following the methodology from "Indoor Localization Using Generative Adversarial 
Network and Bayesian Filtering" (Rajen Bhatt dataset).

Key Differences from Original Approach:
1. Train generative models (DiT, GAN) ONCE on 100% training data
2. For each experiment (5%, 10%, 25%, etc.), use X% real + synthetic to fill gap
3. Compare DiT vs GAN vs Kriging (statistical interpolation) vs Real-Only baseline
4. Evaluate with MLP classifier on test set

This tests: "Can generative models trained on historical data reduce data 
collection overhead by filling gaps with synthetic samples?"
"""

import os
import sys
import time
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from scipy.interpolate import Rbf
# from tqdm import tqdm

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================================
# CONFIG
# ========================================
@dataclass
class Config:
    WORKDIR: str = "./work_dir"
    IMG_SIZE: int = 16           # rasterized "image" is 16x16
    CHANNELS: int = 1           # grayscale
    PATCH: int = 4              # for DiT: 4x4 patches -> (16/4)^2=16 tokens
    WIDTH: int = 256            # transformer d_model
    DEPTH: int = 4              # transformer layers
    HEADS: int = 4              # attention heads
    DROP: float = 0.1
    BATCH: int = 128
    GEN_EPOCHS: int = 2000      # epochs for DiT training
    MLP_EPOCHS: int = 100       # epochs for MLP classifier
    LR: float = 2e-4
    BETA_START: float = 1e-4
    BETA_END: float = 0.02
    TIMESTEPS: int = 400
    EMA: bool = True
    EMA_DECAY: float = 0.999
    VEC_CLIP: float = 3.0       # map standardized features linearly into [-1,1]
    TRAIN_RATIOS: List[float] = None  # will be set to [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    
    def __post_init__(self):
        if self.TRAIN_RATIOS is None:
            self.TRAIN_RATIOS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0]

cfg = Config()
os.makedirs(cfg.WORKDIR, exist_ok=True)
print(f"Work directory: {cfg.WORKDIR}")
print(f"Training ratios: {cfg.TRAIN_RATIOS}")

# ========================================
# DATA LOADING
# ========================================

class DatasetInfo:
    def __init__(self, name, num_features, num_classes, X_train, X_test, y_train, y_test, scaler):
        self.name = name
        self.num_features = num_features
        self.num_classes = num_classes
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler

def load_uci_dataset():
    """Load UCI WiFi Localization dataset (Indoor, 7 APs, 4 rooms)
    
    Dataset format: Each row is one sample with RSS from 7 APs + room label
    Columns 0-6: RSS values from 7 different APs
    Column 7: Room ID (1-4)
    
    Original paper used this dataset with:
    - 2000 samples total
    - 7 features (RSS from 7 APs)
    - 4 classes (4 different rooms)
    """
    print("\n=== Loading UCI WiFi Dataset (Indoor) ===")
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00422/wifi_localization.txt"
    raw_path = os.path.join(cfg.WORKDIR, "wifi_localization.txt")
    
    if not os.path.exists(raw_path):
        print(f"Downloading from {uci_url}...")
        urllib.request.urlretrieve(uci_url, raw_path)
        print(f"Saved to: {raw_path}")
    
    # Load data
    df = pd.read_csv(raw_path, header=None, sep=r"\s+")
    
    # Features: columns 0-6 (RSS from 7 APs)
    # Labels: column 7 (room ID: 1-4)
    X = df.iloc[:, :7].values.astype(np.float32)
    y = df.iloc[:, 7].values.astype(np.int64) - 1  # Convert to 0-indexed (0-3)
    
    num_features = 7
    num_classes = 4
    
    print(f"Total samples: {X.shape[0]}")
    print(f"Features: {num_features} APs, Classes: {num_classes} rooms")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Stratified split: 50/50 train/test (following GAN paper methodology)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    # Standardize using train stats only
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_std = scaler.transform(X_te).astype(np.float32)
    
    print(f"Train: {X_tr_std.shape}, Test: {X_te_std.shape}")
    print(f"Class distribution - Train: {np.bincount(y_tr)}")
    print(f"Class distribution - Test: {np.bincount(y_te)}")
    
    return DatasetInfo("UCI_Indoor", num_features, num_classes, X_tr_std, X_te_std, y_tr, y_te, scaler)

def load_powder_dataset():
    """Load POWDER Outdoor RSS dataset"""
    print("\n=== Loading POWDER Outdoor RSS Dataset ===")
    
    powder_url = "https://zenodo.org/api/records/10962857/files/separated_data.zip/content"
    zip_path = os.path.join(cfg.WORKDIR, "separated_data.zip")
    extract_path = os.path.join(cfg.WORKDIR, "powder_data")
    
    if not os.path.exists(extract_path):
        if not os.path.exists(zip_path):
            print(f"Downloading POWDER dataset from Zenodo...")
            urllib.request.urlretrieve(powder_url, zip_path)
            print(f"Saved to: {zip_path}")
        
        print("Extracting data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted to: {extract_path}")
    
    # Load the train/test split (using random split)
    train_file = os.path.join(extract_path, "separated_data", "train_test_splits", "random_split", "random_train.json")
    test_file = os.path.join(extract_path, "separated_data", "train_test_splits", "random_split", "random_test.json")
    
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    def extract_features_labels(data_dict, max_receivers=25):
        """Extract RSS features and location-based labels from POWDER data"""
        samples = []
        labels = []
        
        for timestamp, sample in data_dict.items():
            rx_data = sample['rx_data']
            tx_coords = sample['tx_coords']
            
            if len(tx_coords) != 1:  # Only single-transmitter samples
                continue
            
            # Create RSS vector (pad/truncate to max_receivers)
            # rx_data format: [[rss, lat, lon, name], ...]
            rss_values = [-120.0] * max_receivers  # Default to very low RSS
            for i, rx in enumerate(rx_data[:max_receivers]):
                rss_val = float(rx[0])  # First element is RSS
                # Validate RSS value is reasonable
                if not np.isnan(rss_val) and not np.isinf(rss_val):
                    rss_values[i] = rss_val
            
            # Create location-based label (grid-based discretization)
            # tx_coords format: [[lat, lon], ...]
            tx_coord = tx_coords[0]
            if len(tx_coord) >= 2:
                tx_lat = float(tx_coord[0])
                tx_lon = float(tx_coord[1])
            else:
                continue  # Skip malformed data
            
            # Grid-based labeling (4x4 grid = 16 location classes for fine-grained localization)
            samples.append(rss_values)
            labels.append((tx_lat, tx_lon))
        
        return np.array(samples, dtype=np.float32), labels
    
    X_tr_raw, train_coords = extract_features_labels(train_data)
    X_te_raw, test_coords = extract_features_labels(test_data)
    
    # Create location-based labels (4x4 grid = 16 labels for fine-grained localization)
    all_coords = train_coords + test_coords
    all_lats = [c[0] for c in all_coords]
    all_lons = [c[1] for c in all_coords]
    
    # Create grid boundaries (4x4 = 16 cells)
    min_lat, max_lat = np.min(all_lats), np.max(all_lats)
    min_lon, max_lon = np.min(all_lons), np.max(all_lons)
    
    lat_bins = np.linspace(min_lat, max_lat, 5)  # 5 edges for 4 bins
    lon_bins = np.linspace(min_lon, max_lon, 5)
    
    def coords_to_label(lat, lon):
        # Digitize returns 1-based indices, we want 0-based
        lat_idx = np.digitize(lat, lat_bins) - 1
        lon_idx = np.digitize(lon, lon_bins) - 1
        # Clamp to valid range [0, 3] for 4x4 grid
        lat_idx = np.clip(lat_idx, 0, 3)
        lon_idx = np.clip(lon_idx, 0, 3)
        # Convert 2D grid position to single label (0-15)
        return lat_idx * 4 + lon_idx
    
    y_tr = np.array([coords_to_label(lat, lon) for lat, lon in train_coords], dtype=np.int64)
    y_te = np.array([coords_to_label(lat, lon) for lat, lon in test_coords], dtype=np.int64)
    
    # Standardize
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_te_std = scaler.transform(X_te_raw).astype(np.float32)
    
    print(f"Train: {X_tr_std.shape}, Test: {X_te_std.shape}")
    print(f"Features: {X_tr_std.shape[1]}, Classes: {len(np.unique(y_tr))}")
    print(f"Class distribution - Train: {np.bincount(y_tr)}")
    print(f"Class distribution - Test: {np.bincount(y_te)}")
    
    return DatasetInfo("POWDER_Outdoor", X_tr_std.shape[1], 16, X_tr_std, X_te_std, y_tr, y_te, scaler)

# ========================================
# RASTERIZATION UTILITIES
# ========================================

def vecs_to_images(vecs_std: np.ndarray, clip=cfg.VEC_CLIP) -> torch.Tensor:
    """Convert standardized feature vectors to 8x8 images in [-1,1]"""
    v = np.clip(vecs_std, -clip, clip) / clip  # -> [-1,1]
    B, num_features = v.shape
    
    imgs = np.zeros((B, 1, cfg.IMG_SIZE, cfg.IMG_SIZE), dtype=np.float32)
    total_pixels = cfg.IMG_SIZE * cfg.IMG_SIZE
    
    if num_features <= cfg.IMG_SIZE:
        # Fill first row with features, pad rest
        row = np.zeros((B, cfg.IMG_SIZE), dtype=np.float32)
        row[:, :num_features] = v
        # Tile vertically
        tile = np.repeat(row[:, None, :], cfg.IMG_SIZE, axis=1)  # (B, 8, 8)
        imgs[:, 0, :, :] = tile
    else:
        # Reshape features to fill the image
        for i in range(B):
            flat = np.zeros(total_pixels, dtype=np.float32)
            flat[:num_features] = v[i]
            imgs[i, 0, :, :] = flat.reshape(cfg.IMG_SIZE, cfg.IMG_SIZE)
    
    return torch.from_numpy(imgs)

def images_to_vecs_std(imgs: torch.Tensor, num_features: int, clip=cfg.VEC_CLIP) -> np.ndarray:
    """Convert 8x8 images back to standardized feature vectors"""
    x = imgs[:, 0, :, :]  # (B, 8, 8)
    B = x.shape[0]
    
    if num_features <= cfg.IMG_SIZE:
        # Extract from first row
        cols = []
        for j in range(num_features):
            cols.append(x[:, :, j].mean(dim=1))  # average over rows
        v_scaled = torch.stack(cols, dim=1)
    else:
        # Extract from flattened image
        flat = x.reshape(B, -1)[:, :num_features]
        v_scaled = flat
    
    vecs_std = (v_scaled * clip).cpu().numpy().astype(np.float32)
    return vecs_std

# ========================================
# DIFFUSION MODELS
# ========================================

class DDPM:
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, timesteps=cfg.TIMESTEPS, beta_start=cfg.BETA_START, beta_end=cfg.BETA_END):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise
    
    @torch.no_grad()
    def sample(self, model, n, img_size, channels, y=None):
        model.eval()
        x = torch.randn(n, channels, img_size, img_size, device=device)
        
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((n,), t_idx, dtype=torch.long, device=device)
            pred_noise = model(x, t, y=y)
            
            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bar[t_idx]
            beta = self.betas[t_idx]
            
            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1.0 / torch.sqrt(alpha)) * (
                x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise
        
        return x

# ========================================
# DiT MODEL
# ========================================

class DiT(nn.Module):
    """Diffusion Transformer for conditional generation"""
    def __init__(self, img_size=cfg.IMG_SIZE, patch_size=cfg.PATCH, in_channels=cfg.CHANNELS,
                 d_model=cfg.WIDTH, depth=cfg.DEPTH, num_heads=cfg.HEADS, 
                 num_classes=4, dropout=cfg.DROP):
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

# ========================================
# GAN MODEL (Conditional GAN for comparison)
# ========================================

class Generator(nn.Module):
    """Conditional Generator for RSS data"""
    def __init__(self, latent_dim=100, num_features=4, num_classes=7, hidden=128):
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
    def __init__(self, num_features=4, num_classes=7, hidden=128):
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

# ========================================
# MLP CLASSIFIER
# ========================================

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
# TRAINING FUNCTIONS
# ========================================

def train_dit(dataset_info: DatasetInfo, epochs=cfg.GEN_EPOCHS):
    """Train DiT model on dataset"""
    print(f"\n{'='*70}")
    print(f"Training DiT on {dataset_info.name}")
    print(f"{'='*70}")
    
    # Prepare data
    Xtr_img = vecs_to_images(dataset_info.X_train)
    ytr_t = torch.from_numpy(dataset_info.y_train)
    
    # Adaptive batch size: use smaller batch for small datasets
    batch_size = min(cfg.BATCH, max(16, len(dataset_info.X_train) // 4))
    drop_last = len(dataset_info.X_train) > batch_size * 2  # Only drop last if we have enough data
    
    train_loader = DataLoader(
        TensorDataset(Xtr_img, ytr_t), 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=drop_last
    )
    
    print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")
    
    # Initialize model and DDPM
    model = DiT(num_classes=dataset_info.num_classes).to(device)
    ddpm = DDPM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
    
    # EMA
    ema_model = None
    if cfg.EMA:
        ema_model = DiT(num_classes=dataset_info.num_classes).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad = False
    
    # Best model tracking
    best_loss = float('inf')
    best_model_state = None
    best_ema_state = None
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x0, y in train_loader:
            x0, y = x0.to(device), y.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, cfg.TIMESTEPS, (x0.shape[0],), device=device)
            
            # Add noise
            xt, noise = ddpm.add_noise(x0, t)
            
            # Predict noise
            pred_noise = model(xt, t, y=y)
            
            # Loss
            loss = F.mse_loss(pred_noise, noise)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # EMA update
            if cfg.EMA:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(cfg.EMA_DECAY).add_(p.data, alpha=1 - cfg.EMA_DECAY)
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if cfg.EMA:
                best_ema_state = {k: v.cpu().clone() for k, v in ema_model.state_dict().items()}
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        if cfg.EMA and best_ema_state is not None:
            ema_model.load_state_dict({k: v.to(device) for k, v in best_ema_state.items()})
        print(f"✓ Loaded best model (loss: {best_loss:.4f})")
    
    print(f"✓ DiT training complete in {time.time() - start_time:.1f}s")
    
    return (ema_model if cfg.EMA else model), ddpm

def train_gan(dataset_info: DatasetInfo, epochs=cfg.GEN_EPOCHS, latent_dim=100):
    """Train Conditional GAN on dataset"""
    print(f"\n{'='*70}")
    print(f"Training GAN on {dataset_info.name}")
    print(f"{'='*70}")
    
    # Prepare data
    X_train = torch.from_numpy(dataset_info.X_train).float()
    y_train = torch.from_numpy(dataset_info.y_train).long()
    
    batch_size = min(cfg.BATCH, max(16, len(dataset_info.X_train) // 4))
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")
    
    # Initialize models
    generator = Generator(latent_dim=latent_dim, 
                         num_features=dataset_info.num_features,
                         num_classes=dataset_info.num_classes).to(device)
    discriminator = Discriminator(num_features=dataset_info.num_features,
                                 num_classes=dataset_info.num_classes).to(device)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    # Best model tracking
    best_g_loss = float('inf')
    best_generator_state = None
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for real_data, real_labels in train_loader:
            real_data = real_data.to(device)
            real_labels = real_labels.to(device)
            batch_size_actual = real_data.size(0)
            
            # Create labels for real and fake data
            real_target = torch.ones(batch_size_actual, 1, device=device)
            fake_target = torch.zeros(batch_size_actual, 1, device=device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real data
            d_real = discriminator(real_data, real_labels)
            d_real_loss = criterion(d_real, real_target)
            
            # Fake data
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_data = generator(z, real_labels)
            d_fake = discriminator(fake_data.detach(), real_labels)
            d_fake_loss = criterion(d_fake, fake_target)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_data = generator(z, real_labels)
            d_fake = discriminator(fake_data, real_labels)
            g_loss = criterion(d_fake, real_target)  # Want discriminator to think fake is real
            
            g_loss.backward()
            g_optimizer.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
        
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        # Save best generator
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            best_generator_state = {k: v.cpu().clone() for k, v in generator.state_dict().items()}
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} | G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f} | Time: {elapsed:.1f}s")
    
    # Load best generator
    if best_generator_state is not None:
        generator.load_state_dict({k: v.to(device) for k, v in best_generator_state.items()})
        print(f"✓ Loaded best generator (loss: {best_g_loss:.4f})")
    
    print(f"✓ GAN training complete in {time.time() - start_time:.1f}s")
    
    return generator, latent_dim

# ========================================
# SYNTHESIS & EVALUATION
# ========================================

@torch.no_grad()
def synthesize_with_dit(model, ddpm, dataset_info: DatasetInfo, num_samples_per_class: int):
    """Generate synthetic samples using trained DiT"""
    model.eval()
    all_imgs = []
    all_labels = []
    
    for class_idx in range(dataset_info.num_classes):
        labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long, device=device)
        imgs = ddpm.sample(model, n=num_samples_per_class, 
                          img_size=cfg.IMG_SIZE, channels=cfg.CHANNELS, y=labels)
        all_imgs.append(imgs)
        all_labels.append(labels)
    
    imgs = torch.cat(all_imgs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Convert back to feature vectors
    vecs_std = images_to_vecs_std(imgs, dataset_info.num_features)
    return vecs_std, labels.cpu().numpy()

@torch.no_grad()
def synthesize_with_gan(generator, latent_dim: int, dataset_info: DatasetInfo, 
                       num_samples_per_class: int):
    """Generate synthetic samples using trained GAN"""
    generator.eval()
    all_samples = []
    all_labels = []
    
    for class_idx in range(dataset_info.num_classes):
        # Generate noise
        z = torch.randn(num_samples_per_class, latent_dim, device=device)
        labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long, device=device)
        
        # Generate samples
        fake_data = generator(z, labels)
        all_samples.append(fake_data.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    X_synth = np.concatenate(all_samples, axis=0)
    y_synth = np.concatenate(all_labels, axis=0)
    
    return X_synth, y_synth

def synthesize_with_kriging(X_real: np.ndarray, y_real: np.ndarray,
                           num_features: int, num_classes: int,
                           num_samples_per_class: int):
    """
    Generate synthetic samples using Kriging (RBF interpolation)
    
    For each class:
    1. Fit RBF interpolator to real samples in that class
    2. Generate synthetic samples by perturbing real samples and interpolating
    """
    all_samples = []
    all_labels = []
    
    for class_idx in range(num_classes):
        # Get real samples for this class
        class_mask = y_real == class_idx
        X_class = X_real[class_mask]
        
        if len(X_class) == 0:
            # No samples for this class, skip
            continue
        
        if len(X_class) == 1:
            # Only one sample, just add noise around it
            noise = np.random.normal(0, 0.1, (num_samples_per_class, num_features))
            synth_samples = X_class + noise
        else:
            # Use RBF interpolation
            # Create synthetic "query points" by perturbing existing samples
            synth_samples = []
            
            for _ in range(num_samples_per_class):
                # Randomly select two real samples
                idx1, idx2 = np.random.choice(len(X_class), 2, replace=True)
                # Interpolate between them with random weight + noise
                alpha = np.random.rand()
                interpolated = alpha * X_class[idx1] + (1 - alpha) * X_class[idx2]
                # Add small Gaussian noise
                interpolated += np.random.normal(0, 0.05, num_features)
                synth_samples.append(interpolated)
            
            synth_samples = np.array(synth_samples, dtype=np.float32)
        
        all_samples.append(synth_samples)
        all_labels.append(np.full(num_samples_per_class, class_idx, dtype=np.int64))
    
    X_synth = np.concatenate(all_samples, axis=0)
    y_synth = np.concatenate(all_labels, axis=0)
    
    return X_synth, y_synth

def train_and_evaluate_mlp(X_train_std: np.ndarray, y_train: np.ndarray,
                          X_test_std: np.ndarray, y_test: np.ndarray,
                          num_features: int, num_classes: int,
                          epochs=cfg.MLP_EPOCHS, lr=2e-3, batch_size=128):
    """Train MLP classifier and return test accuracy"""
    
    mlp = MLP(in_dim=num_features, num_classes=num_classes, hidden=128).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_std).float(), 
                     torch.from_numpy(y_train).long()),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    
    # Training
    mlp.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = mlp(xb)
            loss = criterion(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    # Evaluation
    mlp.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test_std).float().to(device)
        y_test_t = torch.from_numpy(y_test).long().to(device)
        logits = mlp(X_test_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# ========================================
# MAIN EXPERIMENT (Following GAN Paper Methodology)
# ========================================

def main():
    print(f"\n{'='*70}")
    print("DiT DATA FIDELITY EXPERIMENT")
    print("(Replicating GAN Paper Methodology)")
    print(f"{'='*70}")
    print(f"Training ratios: {cfg.TRAIN_RATIOS}")
    print(f"DiT/GAN epochs: {cfg.GEN_EPOCHS}")
    print(f"MLP epochs: {cfg.MLP_EPOCHS}")
    
    # Load datasets
    uci_data = load_uci_dataset()
    powder_data = load_powder_dataset()
    
    datasets = {
        'Indoor': uci_data,
        'Outdoor': powder_data
    }
    
    # Run experiments
    results = []
    
    for name, dataset_info in datasets.items():
        print(f"\n{'='*70}")
        print(f"Experiments on {dataset_info.name}")
        print(f"{'='*70}")
        
        # ============================================================
        # PHASE 1: Train generative models ONCE on 100% training data
        # ============================================================
        print(f"\n{'*'*70}")
        print("PHASE 1: Training Generative Models on 100% Data")
        print(f"{'*'*70}")
        
        # Train DiT on full dataset
        print("\n[1/2] Training DiT on 100% training data...")
        dit_model, ddpm = train_dit(dataset_info)
        
        # Train GAN on full dataset
        print("\n[2/2] Training GAN on 100% training data...")
        gan_generator, latent_dim = train_gan(dataset_info)
        
        # ============================================================
        # PHASE 2: For each train_ratio, use X% real + synthetic
        # ============================================================
        print(f"\n{'*'*70}")
        print("PHASE 2: Testing with Varying Real Data Percentages")
        print(f"{'*'*70}")
        
        # Get total samples per class for target
        total_train = len(dataset_info.y_train)
        samples_per_class_target = total_train // dataset_info.num_classes
        
        for train_ratio in cfg.TRAIN_RATIOS:
            print(f"\n{'─'*70}")
            print(f"Experiment with {int(train_ratio*100)}% real data")
            print(f"{'─'*70}")
            
            # Create stratified subset of X% real data
            if train_ratio < 1.0:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
                subset_indices, _ = next(sss.split(dataset_info.X_train, dataset_info.y_train))
            else:
                subset_indices = np.arange(total_train)
            
            X_real_subset = dataset_info.X_train[subset_indices]
            y_real_subset = dataset_info.y_train[subset_indices]
            
            # Calculate how many samples per class we have and need
            real_samples_per_class = len(y_real_subset) // dataset_info.num_classes
            synthetic_samples_per_class = samples_per_class_target - real_samples_per_class
            
            print(f"Real data: {len(y_real_subset)} samples ({real_samples_per_class}/class)")
            if synthetic_samples_per_class > 0:
                print(f"Synthetic target: {synthetic_samples_per_class}/class to reach {samples_per_class_target}/class")
            
            # ========================================
            # Condition 1: Real Only Baseline
            # ========================================
            print(f"\n[1/5] Real Only Baseline (X% real data, no augmentation)...")
            real_only_accuracy = train_and_evaluate_mlp(
                X_real_subset, y_real_subset,
                dataset_info.X_test, dataset_info.y_test,
                dataset_info.num_features, dataset_info.num_classes
            )
            print(f"✓ Real Only Accuracy: {real_only_accuracy*100:.2f}%")
            
            results.append({
                'dataset': dataset_info.name,
                'method': 'Real_Only',
                'train_ratio': train_ratio,
                'num_real': len(y_real_subset),
                'num_synthetic': 0,
                'total_samples': len(y_real_subset),
                'accuracy': real_only_accuracy
            })
            
            # ========================================
            # Condition 2: DiT Augmented
            # ========================================
            if synthetic_samples_per_class > 0:
                print(f"\n[2/5] DiT Augmented (X% real + DiT synthetic)...")
                
                # Generate synthetic data
                X_dit_synth, y_dit_synth = synthesize_with_dit(
                    dit_model, ddpm, dataset_info, synthetic_samples_per_class
                )
                
                # Combine real + synthetic
                X_dit_combined = np.concatenate([X_real_subset, X_dit_synth], axis=0)
                y_dit_combined = np.concatenate([y_real_subset, y_dit_synth], axis=0)
                
                print(f"  Combined: {len(y_real_subset)} real + {len(y_dit_synth)} synthetic = {len(y_dit_combined)} total")
                
                dit_accuracy = train_and_evaluate_mlp(
                    X_dit_combined, y_dit_combined,
                    dataset_info.X_test, dataset_info.y_test,
                    dataset_info.num_features, dataset_info.num_classes
                )
                print(f"✓ DiT Augmented Accuracy: {dit_accuracy*100:.2f}%")
                
                results.append({
                    'dataset': dataset_info.name,
                    'method': 'DiT_Augmented',
                    'train_ratio': train_ratio,
                    'num_real': len(y_real_subset),
                    'num_synthetic': len(y_dit_synth),
                    'total_samples': len(y_dit_combined),
                    'accuracy': dit_accuracy
                })
            
            # ========================================
            # Condition 3: GAN Augmented
            # ========================================
            if synthetic_samples_per_class > 0:
                print(f"\n[3/5] GAN Augmented (X% real + GAN synthetic)...")
                
                # Generate synthetic data
                X_gan_synth, y_gan_synth = synthesize_with_gan(
                    gan_generator, latent_dim, dataset_info, synthetic_samples_per_class
                )
                
                # Combine real + synthetic
                X_gan_combined = np.concatenate([X_real_subset, X_gan_synth], axis=0)
                y_gan_combined = np.concatenate([y_real_subset, y_gan_synth], axis=0)
                
                print(f"  Combined: {len(y_real_subset)} real + {len(y_gan_synth)} synthetic = {len(y_gan_combined)} total")
                
                gan_accuracy = train_and_evaluate_mlp(
                    X_gan_combined, y_gan_combined,
                    dataset_info.X_test, dataset_info.y_test,
                    dataset_info.num_features, dataset_info.num_classes
                )
                print(f"✓ GAN Augmented Accuracy: {gan_accuracy*100:.2f}%")
                
                results.append({
                    'dataset': dataset_info.name,
                    'method': 'GAN_Augmented',
                    'train_ratio': train_ratio,
                    'num_real': len(y_real_subset),
                    'num_synthetic': len(y_gan_synth),
                    'total_samples': len(y_gan_combined),
                    'accuracy': gan_accuracy
                })
            
            # ========================================
            # Condition 4: Kriging Augmented
            # ========================================
            if synthetic_samples_per_class > 0:
                print(f"\n[4/5] Kriging Augmented (X% real + Kriging interpolation)...")
                
                # Generate synthetic data using kriging
                X_krig_synth, y_krig_synth = synthesize_with_kriging(
                    X_real_subset, y_real_subset,
                    dataset_info.num_features, dataset_info.num_classes,
                    synthetic_samples_per_class
                )
                
                # Combine real + synthetic
                X_krig_combined = np.concatenate([X_real_subset, X_krig_synth], axis=0)
                y_krig_combined = np.concatenate([y_real_subset, y_krig_synth], axis=0)
                
                print(f"  Combined: {len(y_real_subset)} real + {len(y_krig_synth)} synthetic = {len(y_krig_combined)} total")
                
                krig_accuracy = train_and_evaluate_mlp(
                    X_krig_combined, y_krig_combined,
                    dataset_info.X_test, dataset_info.y_test,
                    dataset_info.num_features, dataset_info.num_classes
                )
                print(f"✓ Kriging Augmented Accuracy: {krig_accuracy*100:.2f}%")
                
                results.append({
                    'dataset': dataset_info.name,
                    'method': 'Kriging_Augmented',
                    'train_ratio': train_ratio,
                    'num_real': len(y_real_subset),
                    'num_synthetic': len(y_krig_synth),
                    'total_samples': len(y_krig_combined),
                    'accuracy': krig_accuracy
                })
            
            # ========================================
            # Condition 5: Oracle (100% real - run once)
            # ========================================
            if train_ratio == 1.0:
                print(f"\n[5/5] Oracle (100% real data - upper bound)...")
                oracle_accuracy = train_and_evaluate_mlp(
                    dataset_info.X_train, dataset_info.y_train,
                    dataset_info.X_test, dataset_info.y_test,
                    dataset_info.num_features, dataset_info.num_classes
                )
                print(f"✓ Oracle (100% Real) Accuracy: {oracle_accuracy*100:.2f}%")
                
                # Add oracle for all ratios (for plotting reference line)
                for ratio in cfg.TRAIN_RATIOS:
                    results.append({
                        'dataset': dataset_info.name,
                        'method': 'Oracle_100%',
                        'train_ratio': ratio,
                        'num_real': len(dataset_info.y_train),
                        'num_synthetic': 0,
                        'total_samples': len(dataset_info.y_train),
                        'accuracy': oracle_accuracy
                    })
    
    # ============================================================
    # Save and visualize results
    # ============================================================
    results_df = pd.DataFrame(results)
    results_path = os.path.join(cfg.WORKDIR, "dit_data_fidelity_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
    # Create visualization
    dataset_names = results_df['dataset'].unique()
    num_datasets = len(dataset_names)
    
    fig, axes = plt.subplots(1, num_datasets, figsize=(10 * num_datasets, 6))
    if num_datasets == 1:
        axes = [axes]  # Make it iterable
    
    # Plot each method
    methods = [
        ('Real_Only', 'Real Only (X%)', 's', '--', '#E63946'),
        ('DiT_Augmented', 'DiT Augmented', 'o', '-', '#2E86AB'),
        ('GAN_Augmented', 'GAN Augmented', '^', '-', '#F77F00'),
        ('Kriging_Augmented', 'Kriging Augmented', 'D', '-', '#9D4EDD'),
    ]
    
    for idx, dataset_name in enumerate(dataset_names):
        ax = axes[idx]
        
        for method_key, label, marker, linestyle, color in methods:
            method_data = results_df[(results_df['dataset'] == dataset_name) & 
                                    (results_df['method'] == method_key)]
            if not method_data.empty:
                ax.plot(method_data['train_ratio'] * 100, method_data['accuracy'] * 100, 
                       marker=marker, linewidth=2, markersize=8, label=label,
                       linestyle=linestyle, color=color, alpha=0.8)
        
        # Plot Oracle reference line
        oracle = results_df[(results_df['dataset'] == dataset_name) & 
                           (results_df['method'] == 'Oracle_100%')]
        if not oracle.empty:
            ax.axhline(y=oracle['accuracy'].iloc[0] * 100, 
                      linestyle=':', linewidth=2, color='#06A77D', 
                      label='Oracle (100% Real)', alpha=0.8)
        
        ax.set_xlabel('Real Data Used (%)', fontsize=13)
        ax.set_ylabel('MLP Test Accuracy (%)', fontsize=13)
        ax.set_title(f'{dataset_name}: Generative Models vs Statistical Interpolation', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plot_path = os.path.join(cfg.WORKDIR, "dit_data_fidelity_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    plt.show()
    
    print("\n✓ DiT data fidelity experiment complete!")

if __name__ == "__main__":
    main()
