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
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets, transforms


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


class UniCellularDataset:
    """UniCellular indoor RSS dataset loader (cellular signals, multiple floors)"""
    
    def __init__(self, workdir: str, seed: int = 42, building: str = 'deeb', max_cells: int = 20):
        self.workdir = workdir
        self.seed = seed
        self.building = building.lower()
        self.max_cells = max_cells
        
        # Dataset paths
        self.data_root = os.path.join(workdir, "..", "data", "Unicellular_Dataset", "TestBedData")
        
    def load_fingerprints(self, json_path):
        """Load fingerprints from JSON file and convert to RSS matrix"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        fingerprints = data['Fingerprints']
        
        # First pass: collect all unique cell IDs
        all_cell_ids = set()
        for fp in fingerprints.values():
            for tx in fp['scan']['transmitterInfoList']:
                cell_id = tx['id']
                # Filter out invalid cell IDs
                if cell_id != 2147483647:  # This is a placeholder value
                    all_cell_ids.add(cell_id)
        
        # Sort cell IDs for consistent ordering
        cell_id_list = sorted(list(all_cell_ids))[:self.max_cells]
        cell_id_to_idx = {cid: idx for idx, cid in enumerate(cell_id_list)}
        
        # Second pass: create RSS matrix
        samples = []
        labels = []
        
        for fp in fingerprints.values():
            # Create RSS vector (default to minimum RSS)
            rss_vector = np.full(len(cell_id_list), -120.0, dtype=np.float32)
            
            # Fill in measured RSS values
            for tx in fp['scan']['transmitterInfoList']:
                cell_id = tx['id']
                if cell_id in cell_id_to_idx:
                    idx = cell_id_to_idx[cell_id]
                    rss = tx['rss']
                    # Validate RSS value
                    if not np.isnan(rss) and not np.isinf(rss) and -120 <= rss <= 0:
                        rss_vector[idx] = rss
            
            samples.append(rss_vector)
            # Use floor number as label (0-indexed)
            labels.append(fp['floorNumber'] - 1)
        
        return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.int64), cell_id_list
    
    def load(self) -> DatasetInfo:
        """Load and preprocess the UniCellular dataset"""
        print("\n" + "="*70)
        print(f"Loading UniCellular Dataset ({self.building.upper()})")
        print("="*70)
        
        # Select data file based on building
        if self.building == 'deeb':
            json_path = os.path.join(self.data_root, "DeebMall", 
                                    "DeebMall_4_Phones_9_Floors_1.json")
            num_floors = 9
        elif self.building == 'alexu':
            json_path = os.path.join(self.data_root, "AlexuElectrical", 
                                    "AlexuElectrical_4_Phones_7_Floors_1.json")
            num_floors = 7
        else:
            raise ValueError(f"Unknown building: {self.building}. Choose 'deeb' or 'alexu'")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Dataset not found at {json_path}\n"
                f"Please ensure the Unicellular_Dataset folder is in data/"
            )
        
        # Load fingerprints
        X, y, cell_ids = self.load_fingerprints(json_path)
        
        print(f"Building: {self.building.upper()}")
        print(f"Total samples: {X.shape[0]}")
        print(f"Features: {X.shape[1]} cell towers")
        print(f"Cell IDs: {cell_ids[:5]}..." if len(cell_ids) > 5 else f"Cell IDs: {cell_ids}")
        print(f"Classes: {num_floors} floors")
        print(f"RSS range: [{X[X > -120].min():.1f}, {X.max():.1f}] dBm")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Stratified train/test split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=self.seed)
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
            name=f"UniCellular_{self.building.upper()}",
            num_features=X_tr_std.shape[1],
            num_classes=num_floors,
            X_train=X_tr_std,
            X_test=X_te_std,
            y_train=y_tr,
            y_test=y_te,
            scaler=scaler
        )


class MNISTDataset:
    """MNIST dataset loader for physics-guided loss experiments"""
    
    def __init__(self, workdir: str, seed: int = 42):
        self.workdir = workdir
        self.seed = seed
        self.data_dir = os.path.join(workdir, "mnist_data")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load(self) -> DatasetInfo:
        """Load and preprocess MNIST dataset"""
        print("\n" + "="*70)
        print("Loading MNIST Dataset")
        print("="*70)
        
        # Load MNIST train and test sets
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])
        
        train_dataset = datasets.MNIST(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        # Convert to numpy arrays
        # MNIST images are 28x28, we'll keep them as is
        X_train = train_dataset.data.numpy().astype(np.float32) / 255.0  # Normalize to [0, 1]
        y_train = train_dataset.targets.numpy().astype(np.int64)
        
        X_test = test_dataset.data.numpy().astype(np.float32) / 255.0
        y_test = test_dataset.targets.numpy().astype(np.int64)
        
        # Add channel dimension: (N, 28, 28) -> (N, 1, 28, 28)
        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]
        
        num_classes = 10  # Digits 0-9
        
        print(f"Total train samples: {X_train.shape[0]}")
        print(f"Total test samples: {X_test.shape[0]}")
        print(f"Image shape: {X_train.shape[1:]}")
        print(f"Classes: {num_classes} (digits 0-9)")
        print(f"Train class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        # No scaler needed for MNIST (already normalized to [0, 1])
        scaler = None
        
        return DatasetInfo(
            name="MNIST",
            num_features=28 * 28,  # Flattened size
            num_classes=num_classes,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler
        )


def load_dataset(dataset_name: str, workdir: str, seed: int = 42, **kwargs) -> DatasetInfo:
    """Load a dataset by name"""
    if dataset_name.lower() in ['uci', 'indoor', 'uci_indoor']:
        loader = UCIWiFiDataset(workdir, seed)
        return loader.load()
    elif dataset_name.lower() in ['powder', 'outdoor', 'powder_outdoor']:
        loader = POWDERDataset(workdir, seed)
        return loader.load()
    elif dataset_name.lower() in ['unicellular', 'unicellular_deeb', 'deeb']:
        building = kwargs.get('building', 'deeb')
        loader = UniCellularDataset(workdir, seed, building=building)
        return loader.load()
    elif dataset_name.lower() in ['unicellular_alexu', 'alexu']:
        loader = UniCellularDataset(workdir, seed, building='alexu')
        return loader.load()
    elif dataset_name.lower() == 'mnist':
        loader = MNISTDataset(workdir, seed)
        return loader.load()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
