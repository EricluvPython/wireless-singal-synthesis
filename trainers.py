#!/usr/bin/env python3
"""
Training utilities with checkpointing and wandb integration
"""

import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from config import Config
from data_loader import DatasetInfo
from utils import vecs_to_images
from architectures import DiT, MLP


class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def save(self, model, optimizer, epoch, loss, model_name: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        # Save regular checkpoint
        filename = f"{model_name}_epoch_{epoch}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"{model_name}_best.pt"
            torch.save(checkpoint, best_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints(model_name)
        
        return filepath
    
    def load(self, model, optimizer, model_name: str, epoch: int = None):
        """Load model checkpoint"""
        if epoch is None:
            # Load best checkpoint
            filepath = self.checkpoint_dir / f"{model_name}_best.pt"
        else:
            filepath = self.checkpoint_dir / f"{model_name}_epoch_{epoch}.pt"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']
    
    def _cleanup_old_checkpoints(self, model_name: str):
        """Remove old checkpoints, keeping only the last N"""
        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{model_name}_epoch_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                checkpoint.unlink()


class DiTTrainer:
    """Trainer for DiT model with checkpointing and logging"""
    
    def __init__(self, config: Config, dataset_info: DatasetInfo, device: str = 'cuda'):
        self.config = config
        self.dataset_info = dataset_info
        self.device = device
        self.checkpoint_manager = CheckpointManager(
            config.CHECKPOINT_DIR, 
            config.KEEP_LAST_N
        )
        
    def train(self, model, ddpm, epochs: int = None, use_wandb: bool = False, 
              run_name: str = None):
        """Train DiT model
        
        Args:
            model: DiT model
            ddpm: DDPM instance
            epochs: Number of training epochs
            use_wandb: Whether to log to wandb
            run_name: Name for this training run
            
        Returns:
            Trained model and EMA model (if enabled)
        """
        if epochs is None:
            epochs = self.config.GEN_EPOCHS
            
        print(f"\n{'='*70}")
        print(f"Training DiT on {self.dataset_info.name}")
        print(f"{'='*70}")
        
        # Prepare data
        Xtr_img = vecs_to_images(
            self.dataset_info.X_train, 
            self.config.IMG_SIZE, 
            self.config.VEC_CLIP
        )
        ytr_t = torch.from_numpy(self.dataset_info.y_train)
        
        # Adaptive batch size
        batch_size = min(self.config.BATCH, max(16, len(self.dataset_info.X_train) // 4))
        drop_last = len(self.dataset_info.X_train) > batch_size * 2
        
        train_loader = DataLoader(
            TensorDataset(Xtr_img, ytr_t), 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=drop_last
        )
        
        print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.LR)
        
        # EMA model
        ema_model = None
        if self.config.EMA:
            ema_model = DiT(
                img_size=self.config.IMG_SIZE,
                patch_size=self.config.PATCH,
                in_channels=self.config.CHANNELS,
                d_model=self.config.WIDTH,
                depth=self.config.DEPTH,
                num_heads=self.config.HEADS,
                num_classes=self.dataset_info.num_classes,
                dropout=self.config.DROP
            ).to(self.device)
            ema_model.load_state_dict(model.state_dict())
            for p in ema_model.parameters():
                p.requires_grad = False
        
        # Best model tracking
        best_loss = float('inf')
        
        # Wandb logging
        if use_wandb and WANDB_AVAILABLE:
            wandb.watch(model, log='all', log_freq=100)
        
        # Training loop
        model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for x0, y in train_loader:
                x0, y = x0.to(self.device), y.to(self.device)
                
                # Sample random timesteps
                t = torch.randint(0, self.config.TIMESTEPS, (x0.shape[0],), device=self.device)
                
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
                if self.config.EMA:
                    with torch.no_grad():
                        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                            ema_p.data.mul_(self.config.EMA_DECAY).add_(
                                p.data, alpha=1 - self.config.EMA_DECAY
                            )
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            # Save checkpoint
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
            
            if (epoch + 1) % self.config.CHECKPOINT_EVERY == 0:
                model_name = f"dit_{self.dataset_info.name.lower()}"
                self.checkpoint_manager.save(
                    model, optimizer, epoch, avg_loss, model_name, is_best
                )
                
                if self.config.EMA:
                    ema_name = f"dit_ema_{self.dataset_info.name.lower()}"
                    self.checkpoint_manager.save(
                        ema_model, optimizer, epoch, avg_loss, ema_name, is_best
                    )
            
            # Logging
            if (epoch + 1) % 100 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Best: {best_loss:.4f} | Time: {elapsed:.1f}s")
                
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'dit/epoch': epoch,
                        'dit/loss': avg_loss,
                        'dit/best_loss': best_loss,
                        'dit/learning_rate': optimizer.param_groups[0]['lr']
                    })
        
        # Load best model
        try:
            model_name = f"dit_{self.dataset_info.name.lower()}"
            self.checkpoint_manager.load(model, optimizer, model_name)
            print(f"✓ Loaded best DiT model (loss: {best_loss:.4f})")
            
            if self.config.EMA:
                ema_name = f"dit_ema_{self.dataset_info.name.lower()}"
                self.checkpoint_manager.load(ema_model, optimizer, ema_name)
                print(f"✓ Loaded best EMA model")
        except FileNotFoundError:
            print("Warning: Best checkpoint not found, using final model")
        
        print(f"✓ DiT training complete in {time.time() - start_time:.1f}s")
        
        return ema_model if self.config.EMA else model


class GANTrainer:
    """Trainer for conditional GAN with checkpointing and logging"""
    
    def __init__(self, config: Config, dataset_info: DatasetInfo, device: str = 'cuda'):
        self.config = config
        self.dataset_info = dataset_info
        self.device = device
        self.checkpoint_manager = CheckpointManager(
            config.CHECKPOINT_DIR,
            config.KEEP_LAST_N
        )
    
    def train(self, generator, discriminator, epochs: int = None, 
              use_wandb: bool = False, run_name: str = None):
        """Train conditional GAN
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            epochs: Number of training epochs
            use_wandb: Whether to log to wandb
            run_name: Name for this training run
            
        Returns:
            Trained generator
        """
        if epochs is None:
            epochs = self.config.GEN_EPOCHS
            
        print(f"\n{'='*70}")
        print(f"Training GAN on {self.dataset_info.name}")
        print(f"{'='*70}")
        
        # Prepare data
        X_train = torch.from_numpy(self.dataset_info.X_train).float()
        y_train = torch.from_numpy(self.dataset_info.y_train).long()
        
        batch_size = min(self.config.BATCH, max(16, len(self.dataset_info.X_train) // 4))
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        print(f"Batch size: {batch_size}, Batches per epoch: {len(train_loader)}")
        
        # Optimizers
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        
        criterion = nn.BCELoss()
        
        # Best model tracking
        best_g_loss = float('inf')
        
        # Wandb logging
        if use_wandb and WANDB_AVAILABLE:
            wandb.watch(generator, log='all', log_freq=100)
            wandb.watch(discriminator, log='all', log_freq=100)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            for real_data, real_labels in train_loader:
                real_data = real_data.to(self.device)
                real_labels = real_labels.to(self.device)
                batch_size_actual = real_data.size(0)
                
                # Labels for real and fake data
                real_target = torch.ones(batch_size_actual, 1, device=self.device)
                fake_target = torch.zeros(batch_size_actual, 1, device=self.device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                # Real data
                d_real = discriminator(real_data, real_labels)
                d_real_loss = criterion(d_real, real_target)
                
                # Fake data
                z = torch.randn(batch_size_actual, self.config.GAN_LATENT_DIM, device=self.device)
                fake_data = generator(z, real_labels)
                d_fake = discriminator(fake_data.detach(), real_labels)
                d_fake_loss = criterion(d_fake, fake_target)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size_actual, self.config.GAN_LATENT_DIM, device=self.device)
                fake_data = generator(z, real_labels)
                d_fake = discriminator(fake_data, real_labels)
                g_loss = criterion(d_fake, real_target)
                
                g_loss.backward()
                g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
            
            avg_g_loss = epoch_g_loss / len(train_loader)
            avg_d_loss = epoch_d_loss / len(train_loader)
            
            # Save checkpoint
            is_best = avg_g_loss < best_g_loss
            if is_best:
                best_g_loss = avg_g_loss
            
            if (epoch + 1) % self.config.CHECKPOINT_EVERY == 0:
                gen_name = f"gan_gen_{self.dataset_info.name.lower()}"
                disc_name = f"gan_disc_{self.dataset_info.name.lower()}"
                
                self.checkpoint_manager.save(
                    generator, g_optimizer, epoch, avg_g_loss, gen_name, is_best
                )
                self.checkpoint_manager.save(
                    discriminator, d_optimizer, epoch, avg_d_loss, disc_name, False
                )
            
            # Logging
            if (epoch + 1) % 100 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | G_Loss: {avg_g_loss:.4f}, "
                      f"D_Loss: {avg_d_loss:.4f} | Best_G: {best_g_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")
                
                if use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'gan/epoch': epoch,
                        'gan/g_loss': avg_g_loss,
                        'gan/d_loss': avg_d_loss,
                        'gan/best_g_loss': best_g_loss,
                    })
        
        # Load best generator
        try:
            gen_name = f"gan_gen_{self.dataset_info.name.lower()}"
            self.checkpoint_manager.load(generator, g_optimizer, gen_name)
            print(f"✓ Loaded best generator (loss: {best_g_loss:.4f})")
        except FileNotFoundError:
            print("Warning: Best checkpoint not found, using final generator")
        
        print(f"✓ GAN training complete in {time.time() - start_time:.1f}s")
        
        return generator


class MLPTrainer:
    """Trainer for MLP classifier"""
    
    def __init__(self, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = device
    
    def train_and_evaluate(self, X_train_std, y_train, X_test_std, y_test,
                          num_features: int, num_classes: int, 
                          use_wandb: bool = False, prefix: str = ""):
        """Train MLP and evaluate on test set
        
        Args:
            X_train_std: Training features (standardized)
            y_train: Training labels
            X_test_std: Test features (standardized)
            y_test: Test labels
            num_features: Number of input features
            num_classes: Number of output classes
            use_wandb: Whether to log to wandb
            prefix: Prefix for wandb logging keys
            
        Returns:
            Test accuracy
        """
        mlp = MLP(
            in_dim=num_features, 
            num_classes=num_classes, 
            hidden=self.config.MLP_HIDDEN
        ).to(self.device)
        
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.config.MLP_LR)
        criterion = nn.CrossEntropyLoss()
        
        # DataLoader
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train_std).float(), 
                torch.from_numpy(y_train).long()
            ),
            batch_size=128, 
            shuffle=True, 
            drop_last=False
        )
        
        # Training
        mlp.train()
        for epoch in range(self.config.MLP_EPOCHS):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = mlp(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if use_wandb and WANDB_AVAILABLE and (epoch + 1) % 10 == 0:
                wandb.log({
                    f'{prefix}mlp/epoch': epoch,
                    f'{prefix}mlp/loss': epoch_loss / len(train_loader)
                })
        
        # Evaluation
        mlp.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test_std).float().to(self.device)
            y_test_t = torch.from_numpy(y_test).long().to(self.device)
            logits = mlp(X_test_t)
            y_pred = logits.argmax(dim=1).cpu().numpy()
            accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
