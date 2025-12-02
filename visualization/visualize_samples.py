#!/usr/bin/env python3
"""
Visualize RSS samples from Indoor and Outdoor datasets

This script:
1. Loads raw RSS data from both datasets
2. Selects 4 sample vectors from each
3. Prints the raw RSS values
4. Converts them to 16x16 images (as used by DiT)
5. Saves visualization showing both raw vectors and images
"""

import os
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# Set random seed for reproducibility
np.random.seed(42)

# Config
WORKDIR = "./work_dir"
IMG_SIZE = 8
VEC_CLIP = 3.0

os.makedirs(WORKDIR, exist_ok=True)

# ========================================
# DATA LOADING (copied from experiment)
# ========================================

def load_uci_dataset():
    """Load UCI WiFi Localization dataset (Indoor, 7 APs, 4 rooms)"""
    print("\n=== Loading UCI WiFi Dataset (Indoor) ===")
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00422/wifi_localization.txt"
    raw_path = os.path.join(WORKDIR, "wifi_localization.txt")
    
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
    
    print(f"Total samples: {X.shape[0]}")
    print(f"Features: 7 APs, Classes: 4 rooms")
    
    # Stratified split: 50/50 train/test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    # Standardize using train stats only
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr).astype(np.float32)
    
    print(f"Train: {X_tr_std.shape}")
    
    return X_tr, X_tr_std, y_tr, scaler, 7  # raw, standardized, labels, scaler, num_features

def load_powder_dataset():
    """Load POWDER Outdoor RSS dataset"""
    print("\n=== Loading POWDER Outdoor RSS Dataset ===")
    
    powder_url = "https://zenodo.org/api/records/10962857/files/separated_data.zip/content"
    zip_path = os.path.join(WORKDIR, "separated_data.zip")
    extract_path = os.path.join(WORKDIR, "powder_data")
    
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
    
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
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
            rss_values = [-120.0] * max_receivers  # Default to very low RSS
            for i, rx in enumerate(rx_data[:max_receivers]):
                rss_val = float(rx[0])  # First element is RSS
                if not np.isnan(rss_val) and not np.isinf(rss_val):
                    rss_values[i] = rss_val
            
            # Create location-based label
            tx_coord = tx_coords[0]
            if len(tx_coord) >= 2:
                tx_lat = float(tx_coord[0])
                tx_lon = float(tx_coord[1])
            else:
                continue
            
            samples.append(rss_values)
            labels.append((tx_lat, tx_lon))
        
        return np.array(samples, dtype=np.float32), labels
    
    X_tr_raw, train_coords = extract_features_labels(train_data)
    
    # Create labels (simplified - just use first coordinate)
    y_tr = np.array([i % 16 for i in range(len(train_coords))], dtype=np.int64)
    
    # Standardize
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr_raw).astype(np.float32)
    
    print(f"Train: {X_tr_std.shape}")
    print(f"Features: {X_tr_std.shape[1]}")
    
    return X_tr_raw, X_tr_std, y_tr, scaler, 25  # raw, standardized, labels, scaler, num_features

# ========================================
# RASTERIZATION (copied from experiment)
# ========================================

def vecs_to_images(vecs_std: np.ndarray, num_features: int, clip=VEC_CLIP) -> np.ndarray:
    """Convert standardized feature vectors to 8x8 images in [-1,1]"""
    v = np.clip(vecs_std, -clip, clip) / clip  # -> [-1,1]
    B = v.shape[0]
    
    imgs = np.zeros((B, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    total_pixels = IMG_SIZE * IMG_SIZE
    
    if num_features <= IMG_SIZE:
        # Fill first row with features, pad rest
        row = np.zeros((B, IMG_SIZE), dtype=np.float32)
        row[:, :num_features] = v
        # Tile vertically
        tile = np.repeat(row[:, None, :], IMG_SIZE, axis=1)  # (B, 8, 8)
        imgs[:, 0, :, :] = tile
    else:
        # Reshape features to fill the image
        for i in range(B):
            flat = np.zeros(total_pixels, dtype=np.float32)
            flat[:num_features] = v[i]
            imgs[i, 0, :, :] = flat.reshape(IMG_SIZE, IMG_SIZE)
    
    return imgs

# ========================================
# VISUALIZATION
# ========================================

def visualize_samples(X_raw, X_std, y, dataset_name, num_features, num_samples=4):
    """Visualize RSS samples as rasterized images in a 2x2 grid"""
    
    print(f"\n{'='*70}")
    print(f"Visualizing {dataset_name} Samples")
    print(f"{'='*70}")
    
    # Select random samples from different classes
    unique_classes = np.unique(y)
    selected_indices = []
    
    for i in range(min(num_samples, len(unique_classes))):
        class_idx = unique_classes[i]
        class_samples = np.where(y == class_idx)[0]
        if len(class_samples) > 0:
            selected_indices.append(np.random.choice(class_samples))
    
    # If we need more samples, just pick random ones
    while len(selected_indices) < num_samples:
        idx = np.random.randint(0, len(y))
        if idx not in selected_indices:
            selected_indices.append(idx)
    
    selected_indices = selected_indices[:num_samples]
    
    # Get selected samples
    X_raw_samples = X_raw[selected_indices]
    X_std_samples = X_std[selected_indices]
    y_samples = y[selected_indices]
    
    # Convert to images
    imgs = vecs_to_images(X_std_samples, num_features)
    
    # Print raw RSS vectors
    print(f"\nRaw RSS Vectors:")
    print(f"{'─'*70}")
    for i, (raw_vec, label) in enumerate(zip(X_raw_samples, y_samples)):
        print(f"\nSample {i+1} (Class {label}):")
        if num_features <= 7:
            # Indoor - print all values
            print(f"  RSS values: {raw_vec}")
        else:
            # Outdoor - print first 10 and summary
            print(f"  RSS values (first 10): {raw_vec[:10]}")
            print(f"  RSS range: [{raw_vec.min():.2f}, {raw_vec.max():.2f}]")
            print(f"  RSS mean: {raw_vec.mean():.2f}, std: {raw_vec.std():.2f}")
    
    # Choose colormap based on dataset
    cmap = 'RdBu_r' if dataset_name == 'Indoor' else 'viridis'
    
    # Create 2x2 grid of images
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        im = ax.imshow(imgs[i, 0], cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
        
        # Add gridlines
        ax.set_xticks(np.arange(-.5, IMG_SIZE, 1), minor=True)
        ax.set_yticks(np.arange(-.5, IMG_SIZE, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.5)
        ax.tick_params(which='both', size=0, labelsize=0)
        
        # Add colorbar for each image
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
    
    plt.tight_layout(pad=0.3)
    
    # Save figure
    output_path = os.path.join(WORKDIR, f"{dataset_name.lower()}_samples.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    
    return output_path

# ========================================
# MAIN
# ========================================

def main():
    print("="*70)
    print("RSS SAMPLE VISUALIZATION")
    print("="*70)
    
    # Load and visualize Indoor dataset
    X_raw_indoor, X_std_indoor, y_indoor, scaler_indoor, num_feat_indoor = load_uci_dataset()
    indoor_path = visualize_samples(X_raw_indoor, X_std_indoor, y_indoor, 
                                    "Indoor", num_feat_indoor, num_samples=4)
    
    # Load and visualize Outdoor dataset
    X_raw_outdoor, X_std_outdoor, y_outdoor, scaler_outdoor, num_feat_outdoor = load_powder_dataset()
    outdoor_path = visualize_samples(X_raw_outdoor, X_std_outdoor, y_outdoor, 
                                     "Outdoor", num_feat_outdoor, num_samples=4)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Indoor samples: {indoor_path}")
    print(f"Outdoor samples: {outdoor_path}")
    print("\nThese visualizations show how RSS vectors are converted to")
    print(f"{IMG_SIZE}x{IMG_SIZE} images for the DiT model:")
    print(f"  - Indoor: {num_feat_indoor} features → tiled vertically in image")
    print(f"  - Outdoor: {num_feat_outdoor} features → filled sequentially in image")
    print("="*70)

if __name__ == "__main__":
    main()
