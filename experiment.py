#!/usr/bin/env python3
"""
Main experiment script for DiT Data Fidelity Experiment

This script orchestrates the full experiment:
1. Load datasets
2. Train generative models (DiT, GAN) once on 100% data
3. Run experiments with varying data percentages
4. Evaluate and visualize results
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import Config, get_config
from data_loader import load_dataset, DatasetInfo
from architectures import DiT, Generator, Discriminator
from diffusion import DDPM
from trainers import DiTTrainer, GANTrainer, MLPTrainer
from synthesis import DiTSynthesizer, GANSynthesizer, KrigingSynthesizer
from utils import set_seed, get_device


def train_generative_models(config: Config, dataset_info: DatasetInfo, 
                           device: str, use_wandb: bool = False):
    """Train DiT and GAN models on full dataset
    
    Args:
        config: Configuration object
        dataset_info: Dataset information
        device: Device to train on
        use_wandb: Whether to use wandb logging
        
    Returns:
        dit_model, ddpm, gan_generator, gan_latent_dim
    """
    print(f"\n{'*'*70}")
    print("PHASE 1: Training Generative Models on 100% Data")
    print(f"{'*'*70}")
    
    # Train DiT
    print("\n[1/2] Training DiT on 100% training data...")
    dit_model = DiT(
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH,
        in_channels=config.CHANNELS,
        d_model=config.WIDTH,
        depth=config.DEPTH,
        num_heads=config.HEADS,
        num_classes=dataset_info.num_classes,
        dropout=config.DROP
    ).to(device)
    
    ddpm = DDPM(
        timesteps=config.TIMESTEPS,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        device=device
    )
    
    dit_trainer = DiTTrainer(config, dataset_info, device)
    dit_model = dit_trainer.train(dit_model, ddpm, use_wandb=use_wandb)
    
    # Train GAN
    print("\n[2/2] Training GAN on 100% training data...")
    gan_generator = Generator(
        latent_dim=config.GAN_LATENT_DIM,
        num_features=dataset_info.num_features,
        num_classes=dataset_info.num_classes,
        hidden=config.GAN_HIDDEN
    ).to(device)
    
    gan_discriminator = Discriminator(
        num_features=dataset_info.num_features,
        num_classes=dataset_info.num_classes,
        hidden=config.GAN_HIDDEN
    ).to(device)
    
    gan_trainer = GANTrainer(config, dataset_info, device)
    gan_generator = gan_trainer.train(
        gan_generator, 
        gan_discriminator, 
        use_wandb=use_wandb
    )
    
    return dit_model, ddpm, gan_generator, config.GAN_LATENT_DIM


def run_experiment_for_ratio(config: Config, dataset_info: DatasetInfo,
                            dit_synthesizer: DiTSynthesizer,
                            gan_synthesizer: GANSynthesizer,
                            krig_synthesizer: KrigingSynthesizer,
                            mlp_trainer: MLPTrainer,
                            train_ratio: float,
                            use_wandb: bool = False):
    """Run experiment for a specific training data ratio
    
    Args:
        config: Configuration object
        dataset_info: Dataset information
        dit_synthesizer: DiT synthesizer
        gan_synthesizer: GAN synthesizer
        krig_synthesizer: Kriging synthesizer
        mlp_trainer: MLP trainer
        train_ratio: Fraction of real data to use
        use_wandb: Whether to log to wandb
        
    Returns:
        List of result dictionaries
    """
    print(f"\n{'─'*70}")
    print(f"Experiment with {int(train_ratio*100)}% real data")
    print(f"{'─'*70}")
    
    results = []
    
    # Create stratified subset of X% real data
    if train_ratio < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
        subset_indices, _ = next(sss.split(dataset_info.X_train, dataset_info.y_train))
    else:
        subset_indices = np.arange(len(dataset_info.y_train))
    
    X_real_subset = dataset_info.X_train[subset_indices]
    y_real_subset = dataset_info.y_train[subset_indices]
    
    # Calculate samples per class
    total_train = len(dataset_info.y_train)
    samples_per_class_target = total_train // dataset_info.num_classes
    real_samples_per_class = len(y_real_subset) // dataset_info.num_classes
    synthetic_samples_per_class = samples_per_class_target - real_samples_per_class
    
    print(f"Real data: {len(y_real_subset)} samples ({real_samples_per_class}/class)")
    if synthetic_samples_per_class > 0:
        print(f"Synthetic target: {synthetic_samples_per_class}/class")
    
    # 1. Real Only Baseline
    print(f"\n[1/5] Real Only Baseline...")
    real_only_accuracy = mlp_trainer.train_and_evaluate(
        X_real_subset, y_real_subset,
        dataset_info.X_test, dataset_info.y_test,
        dataset_info.num_features, dataset_info.num_classes,
        use_wandb=use_wandb, prefix=f"ratio_{int(train_ratio*100)}/real_only/"
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
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            f'ratio_{int(train_ratio*100)}/real_only/accuracy': real_only_accuracy
        })
    
    # 2. DiT Augmented
    if synthetic_samples_per_class > 0:
        print(f"\n[2/5] DiT Augmented...")
        X_dit_synth, y_dit_synth = dit_synthesizer.synthesize(
            dataset_info, synthetic_samples_per_class
        )
        
        X_dit_combined = np.concatenate([X_real_subset, X_dit_synth], axis=0)
        y_dit_combined = np.concatenate([y_real_subset, y_dit_synth], axis=0)
        
        print(f"  Combined: {len(y_real_subset)} real + {len(y_dit_synth)} synthetic")
        
        dit_accuracy = mlp_trainer.train_and_evaluate(
            X_dit_combined, y_dit_combined,
            dataset_info.X_test, dataset_info.y_test,
            dataset_info.num_features, dataset_info.num_classes,
            use_wandb=use_wandb, prefix=f"ratio_{int(train_ratio*100)}/dit/"
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
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f'ratio_{int(train_ratio*100)}/dit/accuracy': dit_accuracy
            })
    
    # 3. GAN Augmented
    if synthetic_samples_per_class > 0:
        print(f"\n[3/5] GAN Augmented...")
        X_gan_synth, y_gan_synth = gan_synthesizer.synthesize(
            dataset_info, synthetic_samples_per_class
        )
        
        X_gan_combined = np.concatenate([X_real_subset, X_gan_synth], axis=0)
        y_gan_combined = np.concatenate([y_real_subset, y_gan_synth], axis=0)
        
        print(f"  Combined: {len(y_real_subset)} real + {len(y_gan_synth)} synthetic")
        
        gan_accuracy = mlp_trainer.train_and_evaluate(
            X_gan_combined, y_gan_combined,
            dataset_info.X_test, dataset_info.y_test,
            dataset_info.num_features, dataset_info.num_classes,
            use_wandb=use_wandb, prefix=f"ratio_{int(train_ratio*100)}/gan/"
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
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f'ratio_{int(train_ratio*100)}/gan/accuracy': gan_accuracy
            })
    
    # 4. Kriging Augmented
    if synthetic_samples_per_class > 0:
        print(f"\n[4/5] Kriging Augmented...")
        X_krig_synth, y_krig_synth = krig_synthesizer.synthesize(
            X_real_subset, y_real_subset,
            dataset_info.num_features, dataset_info.num_classes,
            synthetic_samples_per_class
        )
        
        X_krig_combined = np.concatenate([X_real_subset, X_krig_synth], axis=0)
        y_krig_combined = np.concatenate([y_real_subset, y_krig_synth], axis=0)
        
        print(f"  Combined: {len(y_real_subset)} real + {len(y_krig_synth)} synthetic")
        
        krig_accuracy = mlp_trainer.train_and_evaluate(
            X_krig_combined, y_krig_combined,
            dataset_info.X_test, dataset_info.y_test,
            dataset_info.num_features, dataset_info.num_classes,
            use_wandb=use_wandb, prefix=f"ratio_{int(train_ratio*100)}/kriging/"
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
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f'ratio_{int(train_ratio*100)}/kriging/accuracy': krig_accuracy
            })
    
    # 5. Oracle (100% real - run once)
    if train_ratio == 1.0:
        print(f"\n[5/5] Oracle (100% real data)...")
        oracle_accuracy = mlp_trainer.train_and_evaluate(
            dataset_info.X_train, dataset_info.y_train,
            dataset_info.X_test, dataset_info.y_test,
            dataset_info.num_features, dataset_info.num_classes,
            use_wandb=use_wandb, prefix="oracle/"
        )
        print(f"✓ Oracle Accuracy: {oracle_accuracy*100:.2f}%")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({'oracle/accuracy': oracle_accuracy})
    
    return results


def visualize_results(results_df: pd.DataFrame, config: Config):
    """Create visualization of results
    
    Args:
        results_df: DataFrame with experiment results
        config: Configuration object
    """
    dataset_names = results_df['dataset'].unique()
    num_datasets = len(dataset_names)
    
    fig, axes = plt.subplots(1, num_datasets, figsize=(10 * num_datasets, 6))
    if num_datasets == 1:
        axes = [axes]
    
    methods = [
        ('Real_Only', 'Real Only (X%)', 's', '--', '#E63946'),
        ('DiT_Augmented', 'DiT Augmented', 'o', '-', '#2E86AB'),
        ('GAN_Augmented', 'GAN Augmented', '^', '-', '#F77F00'),
        ('Kriging_Augmented', 'Kriging Augmented', 'D', '-', '#9D4EDD'),
    ]
    
    for idx, dataset_name in enumerate(dataset_names):
        ax = axes[idx]
        
        for method_key, label, marker, linestyle, color in methods:
            method_data = results_df[
                (results_df['dataset'] == dataset_name) & 
                (results_df['method'] == method_key)
            ]
            if not method_data.empty:
                ax.plot(
                    method_data['train_ratio'] * 100, 
                    method_data['accuracy'] * 100, 
                    marker=marker, linewidth=2, markersize=8, label=label,
                    linestyle=linestyle, color=color, alpha=0.8
                )
        
        # Plot Oracle reference line
        oracle = results_df[
            (results_df['dataset'] == dataset_name) & 
            (results_df['method'] == 'Oracle_100%')
        ]
        if not oracle.empty:
            ax.axhline(
                y=oracle['accuracy'].iloc[0] * 100, 
                linestyle=':', linewidth=2, color='#06A77D', 
                label='Oracle (100% Real)', alpha=0.8
            )
        
        ax.set_xlabel('Real Data Used (%)', fontsize=13)
        ax.set_ylabel('MLP Test Accuracy (%)', fontsize=13)
        ax.set_title(
            f'{dataset_name}: Generative Models vs Statistical Interpolation', 
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plot_path = os.path.join(config.WORKDIR, "dit_data_fidelity_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    # Log to wandb if available
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"results_plot": wandb.Image(plot_path)})


def run_dataset_experiment(dataset_name: str, config: Config, device: str, 
                          use_wandb: bool = False):
    """Run complete experiment on one dataset
    
    Args:
        dataset_name: Name of dataset to use
        config: Configuration object
        device: Device to use
        use_wandb: Whether to use wandb logging
        
    Returns:
        List of result dictionaries
    """
    print(f"\n{'='*70}")
    print(f"Experiments on {dataset_name}")
    print(f"{'='*70}")
    
    # Load dataset
    dataset_info = load_dataset(dataset_name, config.WORKDIR, config.SEED)
    
    # Train generative models
    dit_model, ddpm, gan_generator, gan_latent_dim = train_generative_models(
        config, dataset_info, device, use_wandb
    )
    
    # Create synthesizers
    dit_synthesizer = DiTSynthesizer(dit_model, ddpm, config, device)
    gan_synthesizer = GANSynthesizer(gan_generator, gan_latent_dim, device)
    krig_synthesizer = KrigingSynthesizer()
    
    # Create MLP trainer
    mlp_trainer = MLPTrainer(config, device)
    
    # Run experiments for each ratio
    print(f"\n{'*'*70}")
    print("PHASE 2: Testing with Varying Real Data Percentages")
    print(f"{'*'*70}")
    
    all_results = []
    oracle_accuracy = None
    
    for train_ratio in config.TRAIN_RATIOS:
        ratio_results = run_experiment_for_ratio(
            config, dataset_info,
            dit_synthesizer, gan_synthesizer, krig_synthesizer,
            mlp_trainer, train_ratio, use_wandb
        )
        all_results.extend(ratio_results)
        
        # Get oracle accuracy (from 100% real experiment)
        if train_ratio == 1.0:
            oracle_result = [r for r in ratio_results if r['method'] == 'Real_Only'][0]
            oracle_accuracy = oracle_result['accuracy']
    
    # Add oracle reference for all ratios
    if oracle_accuracy is not None:
        for ratio in config.TRAIN_RATIOS:
            all_results.append({
                'dataset': dataset_info.name,
                'method': 'Oracle_100%',
                'train_ratio': ratio,
                'num_real': len(dataset_info.y_train),
                'num_synthetic': 0,
                'total_samples': len(dataset_info.y_train),
                'accuracy': oracle_accuracy
            })
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="DiT Data Fidelity Experiment"
    )
    parser.add_argument(
        '--datasets', 
        nargs='+', 
        default=['indoor', 'outdoor'],
        choices=['indoor', 'outdoor', 'uci', 'powder'],
        help='Datasets to run experiments on'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default=None,
        help='Wandb project name'
    )
    parser.add_argument(
        '--gen-epochs',
        type=int,
        default=None,
        help='Number of epochs for generative model training'
    )
    parser.add_argument(
        '--mlp-epochs',
        type=int,
        default=None,
        help='Number of epochs for MLP training'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Override config from args
    if args.gen_epochs is not None:
        config.GEN_EPOCHS = args.gen_epochs
    if args.mlp_epochs is not None:
        config.MLP_EPOCHS = args.mlp_epochs
    if args.wandb_project is not None:
        config.WANDB_PROJECT = args.wandb_project
    
    use_wandb = config.USE_WANDB and not args.no_wandb and WANDB_AVAILABLE
    
    # Setup
    set_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            config=vars(config),
            name=f"dit_experiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Run experiments
    print(f"\n{'='*70}")
    print("DiT DATA FIDELITY EXPERIMENT")
    print(f"{'='*70}")
    print(f"Training ratios: {config.TRAIN_RATIOS}")
    print(f"Generative model epochs: {config.GEN_EPOCHS}")
    print(f"MLP epochs: {config.MLP_EPOCHS}")
    print(f"Wandb logging: {use_wandb}")
    
    all_results = []
    
    # Normalize dataset names
    dataset_map = {
        'indoor': 'uci',
        'uci': 'uci',
        'outdoor': 'powder',
        'powder': 'powder'
    }
    
    datasets_to_run = [dataset_map[d.lower()] for d in args.datasets]
    datasets_to_run = list(set(datasets_to_run))  # Remove duplicates
    
    for dataset_name in datasets_to_run:
        results = run_dataset_experiment(dataset_name, config, device, use_wandb)
        all_results.extend(results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(config.WORKDIR, "dit_data_fidelity_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Log to wandb
    if use_wandb:
        wandb.save(results_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    
    # Visualize
    visualize_results(results_df, config)
    
    # Finish wandb
    if use_wandb:
        wandb.finish()
    
    print("\n✓ DiT data fidelity experiment complete!")


if __name__ == "__main__":
    main()
