#!/usr/bin/env python3
"""
Synthesis utilities for generating samples with different methods
"""

import numpy as np
import torch

from config import Config
from data_loader import DatasetInfo
from utils import vecs_to_images, images_to_vecs_std


class DiTSynthesizer:
    """Generate synthetic samples using trained DiT model"""
    
    def __init__(self, model, ddpm, config: Config, device: str = 'cuda'):
        self.model = model
        self.ddpm = ddpm
        self.config = config
        self.device = device
    
    @torch.no_grad()
    def synthesize(self, dataset_info: DatasetInfo, num_samples_per_class: int):
        """Generate synthetic samples
        
        Args:
            dataset_info: Dataset information
            num_samples_per_class: Number of samples to generate per class
            
        Returns:
            X_synth: Synthetic feature vectors (standardized)
            y_synth: Synthetic labels
        """
        self.model.eval()
        all_imgs = []
        all_labels = []
        
        for class_idx in range(dataset_info.num_classes):
            labels = torch.full(
                (num_samples_per_class,), 
                class_idx, 
                dtype=torch.long, 
                device=self.device
            )
            imgs = self.ddpm.sample(
                self.model, 
                n=num_samples_per_class, 
                img_size=self.config.IMG_SIZE, 
                channels=self.config.CHANNELS, 
                y=labels
            )
            all_imgs.append(imgs)
            all_labels.append(labels)
        
        imgs = torch.cat(all_imgs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Convert back to feature vectors
        vecs_std = images_to_vecs_std(imgs, dataset_info.num_features, self.config.VEC_CLIP)
        
        return vecs_std, labels.cpu().numpy()


class GANSynthesizer:
    """Generate synthetic samples using trained GAN"""
    
    def __init__(self, generator, latent_dim: int, device: str = 'cuda'):
        self.generator = generator
        self.latent_dim = latent_dim
        self.device = device
    
    @torch.no_grad()
    def synthesize(self, dataset_info: DatasetInfo, num_samples_per_class: int):
        """Generate synthetic samples
        
        Args:
            dataset_info: Dataset information
            num_samples_per_class: Number of samples to generate per class
            
        Returns:
            X_synth: Synthetic feature vectors (standardized)
            y_synth: Synthetic labels
        """
        self.generator.eval()
        all_samples = []
        all_labels = []
        
        for class_idx in range(dataset_info.num_classes):
            # Generate noise
            z = torch.randn(num_samples_per_class, self.latent_dim, device=self.device)
            labels = torch.full(
                (num_samples_per_class,), 
                class_idx, 
                dtype=torch.long, 
                device=self.device
            )
            
            # Generate samples
            fake_data = self.generator(z, labels)
            all_samples.append(fake_data.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        
        X_synth = np.concatenate(all_samples, axis=0)
        y_synth = np.concatenate(all_labels, axis=0)
        
        return X_synth, y_synth


class KrigingSynthesizer:
    """Generate synthetic samples using Kriging (RBF interpolation)"""
    
    def __init__(self):
        pass
    
    def synthesize(self, X_real: np.ndarray, y_real: np.ndarray,
                  num_features: int, num_classes: int,
                  num_samples_per_class: int):
        """Generate synthetic samples using interpolation
        
        Args:
            X_real: Real feature vectors (standardized)
            y_real: Real labels
            num_features: Number of features
            num_classes: Number of classes
            num_samples_per_class: Number of samples to generate per class
            
        Returns:
            X_synth: Synthetic feature vectors (standardized)
            y_synth: Synthetic labels
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
                # Use interpolation between existing samples
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
