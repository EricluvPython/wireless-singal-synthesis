#!/usr/bin/env python3
"""
Dataset loading and processing utilities
"""

import os
import json
import urllib.request
import zipfile
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class DatasetInfo:
    """Container for dataset information"""
    name: str
    num_features: int
    num_classes: int
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


class UCIWiFiDataset:
    """UCI WiFi Localization dataset loader (Indoor, 7 APs, 4 rooms)"""
    
    def __init__(self, workdir: str, seed: int = 42):
        self.workdir = workdir
        self.seed = seed
        self.url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00422/wifi_localization.txt"
        
    def download(self):
        """Download the dataset if not already present"""
        raw_path = os.path.join(self.workdir, "wifi_localization.txt")
        
        if not os.path.exists(raw_path):
            print(f"Downloading UCI WiFi dataset from {self.url}...")
            urllib.request.urlretrieve(self.url, raw_path)
            print(f"✓ Saved to: {raw_path}")
        
        return raw_path
    
    def load(self) -> DatasetInfo:
        """Load and preprocess the dataset"""
        print("\n" + "="*70)
        print("Loading UCI WiFi Dataset (Indoor)")
        print("="*70)
        
        raw_path = self.download()
        
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
        
        # Stratified split: 50/50 train/test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=self.seed)
        train_idx, test_idx = next(sss.split(X, y))
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Standardize using train stats only
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_std = scaler.transform(X_te).astype(np.float32)
        
        print(f"Train: {X_tr_std.shape}, Test: {X_te_std.shape}")
        print(f"Train class distribution: {np.bincount(y_tr)}")
        print(f"Test class distribution: {np.bincount(y_te)}")
        
        return DatasetInfo(
            name="UCI_Indoor",
            num_features=num_features,
            num_classes=num_classes,
            X_train=X_tr_std,
            X_test=X_te_std,
            y_train=y_tr,
            y_test=y_te,
            scaler=scaler
        )


class POWDERDataset:
    """POWDER Outdoor RSS dataset loader"""
    
    def __init__(self, workdir: str, seed: int = 42, max_receivers: int = 25):
        self.workdir = workdir
        self.seed = seed
        self.max_receivers = max_receivers
        self.url = "https://zenodo.org/api/records/10962857/files/separated_data.zip/content"
        
    def download(self):
        """Download and extract the dataset if not already present"""
        zip_path = os.path.join(self.workdir, "separated_data.zip")
        extract_path = os.path.join(self.workdir, "powder_data")
        
        if not os.path.exists(extract_path):
            if not os.path.exists(zip_path):
                print(f"Downloading POWDER dataset from Zenodo...")
                urllib.request.urlretrieve(self.url, zip_path)
                print(f"✓ Saved to: {zip_path}")
            
            print("Extracting data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"✓ Extracted to: {extract_path}")
        
        return extract_path
    
    def extract_features_labels(self, data_dict):
        """Extract RSS features and location-based labels from POWDER data"""
        samples = []
        labels = []
        
        for timestamp, sample in data_dict.items():
            rx_data = sample['rx_data']
            tx_coords = sample['tx_coords']
            
            if len(tx_coords) != 1:  # Only single-transmitter samples
                continue
            
            # Create RSS vector (pad/truncate to max_receivers)
            rss_values = [-120.0] * self.max_receivers  # Default to very low RSS
            for i, rx in enumerate(rx_data[:self.max_receivers]):
                rss_val = float(rx[0])  # First element is RSS
                # Validate RSS value is reasonable
                if not np.isnan(rss_val) and not np.isinf(rss_val):
                    rss_values[i] = rss_val
            
            # Create location-based label
            tx_coord = tx_coords[0]
            if len(tx_coord) >= 2:
                tx_lat = float(tx_coord[0])
                tx_lon = float(tx_coord[1])
            else:
                continue  # Skip malformed data
            
            samples.append(rss_values)
            labels.append((tx_lat, tx_lon))
        
        return np.array(samples, dtype=np.float32), labels
    
    def coords_to_labels(self, train_coords, test_coords):
        """Convert coordinates to grid-based labels (4x4 grid = 16 classes)"""
        all_coords = train_coords + test_coords
        all_lats = [c[0] for c in all_coords]
        all_lons = [c[1] for c in all_coords]
        
        # Create grid boundaries (4x4 = 16 cells)
        min_lat, max_lat = np.min(all_lats), np.max(all_lats)
        min_lon, max_lon = np.min(all_lons), np.max(all_lons)
        
        lat_bins = np.linspace(min_lat, max_lat, 5)  # 5 edges for 4 bins
        lon_bins = np.linspace(min_lon, max_lon, 5)
        
        def coords_to_label(lat, lon):
            lat_idx = np.digitize(lat, lat_bins) - 1
            lon_idx = np.digitize(lon, lon_bins) - 1
            lat_idx = np.clip(lat_idx, 0, 3)
            lon_idx = np.clip(lon_idx, 0, 3)
            return lat_idx * 4 + lon_idx
        
        y_tr = np.array([coords_to_label(lat, lon) for lat, lon in train_coords], dtype=np.int64)
        y_te = np.array([coords_to_label(lat, lon) for lat, lon in test_coords], dtype=np.int64)
        
        return y_tr, y_te
    
    def load(self) -> DatasetInfo:
        """Load and preprocess the dataset"""
        print("\n" + "="*70)
        print("Loading POWDER Outdoor RSS Dataset")
        print("="*70)
        
        extract_path = self.download()
        
        # Load the train/test split (using random split)
        train_file = os.path.join(extract_path, "separated_data", "train_test_splits", 
                                 "random_split", "random_train.json")
        test_file = os.path.join(extract_path, "separated_data", "train_test_splits", 
                                "random_split", "random_test.json")
        
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        X_tr_raw, train_coords = self.extract_features_labels(train_data)
        X_te_raw, test_coords = self.extract_features_labels(test_data)
        
        y_tr, y_te = self.coords_to_labels(train_coords, test_coords)
        
        # Standardize
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_te_std = scaler.transform(X_te_raw).astype(np.float32)
        
        print(f"Train: {X_tr_std.shape}, Test: {X_te_std.shape}")
        print(f"Features: {X_tr_std.shape[1]}, Classes: {len(np.unique(y_tr))}")
        print(f"Train class distribution: {np.bincount(y_tr)}")
        print(f"Test class distribution: {np.bincount(y_te)}")
        
        return DatasetInfo(
            name="POWDER_Outdoor",
            num_features=X_tr_std.shape[1],
            num_classes=16,
            X_train=X_tr_std,
            X_test=X_te_std,
            y_train=y_tr,
            y_test=y_te,
            scaler=scaler
        )


def load_dataset(dataset_name: str, workdir: str, seed: int = 42) -> DatasetInfo:
    """Load a dataset by name"""
    if dataset_name.lower() in ['uci', 'indoor', 'uci_indoor']:
        loader = UCIWiFiDataset(workdir, seed)
        return loader.load()
    elif dataset_name.lower() in ['powder', 'outdoor', 'powder_outdoor']:
        loader = POWDERDataset(workdir, seed)
        return loader.load()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
