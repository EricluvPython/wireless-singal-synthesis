# DiT Data Fidelity Experiment - Redesign Summary

## Methodology Change (Following GAN Paper)

### Previous Approach (INCORRECT)
- Train DiT separately on X% data for each experiment
- Generate synthetic to fill to 100%
- Problem: Generative model quality varies with data amount

### New Approach (CORRECT - Following GAN Paper)
1. **Phase 1**: Train generative models (DiT, GAN) ONCE on 100% training data
2. **Phase 2**: For each experiment (5%, 10%, 25%, 50%, 80%, 100%):
   - Take X% real samples
   - Use pre-trained generative model to fill gap to 100%
   - Train MLP classifier on combined data
   - Evaluate on test set

### Rationale
- Simulates realistic scenario: "Can I reduce data collection if I have pre-trained generative models from historical data?"
- Fair comparison: All generative models trained on same 100% data
- Matches published GAN paper methodology

## Comparison Methods

### 1. Real Only Baseline
- Train MLP on X% real data only
- No augmentation
- Shows performance degradation with limited data

### 2. DiT Augmented
- X% real + DiT synthetic (trained on 100%)
- Tests: Can diffusion models reduce data collection needs?

### 3. GAN Augmented (NEW)
- X% real + GAN synthetic (trained on 100%)
- Comparison to published GAN approach
- Conditional GAN with class embeddings

### 4. Kriging Augmented (NEW)
- X% real + Kriging interpolation
- Statistical baseline (RBF interpolation between real samples)
- Tests: Are deep generative models better than simple interpolation?

### 5. Oracle (100% Real)
- Upper bound: MLP trained on 100% real data
- Reference line in plots

## Key Changes to Code

### Added Components
1. **Generator/Discriminator classes**: Conditional GAN implementation
2. **train_gan()**: GAN training with adversarial loss
3. **synthesize_with_gan()**: Generate synthetic samples from trained GAN
4. **synthesize_with_kriging()**: Statistical interpolation baseline
5. **Redesigned main()**: Two-phase approach (train models once, then experiments)

### Removed Components
- `run_experiment()`: Old function that trained DiT per experiment
- `select_high_quality_synthetic()`: Removed filtering (use all synthetic)
- POWDER dataset loading: Commented out to focus on UCI Indoor first

### Model Architecture
- **DiT**: Same (IMG_SIZE=16, PATCH=4, WIDTH=256, DEPTH=4, 2000 epochs)
- **GAN**: Generator/Discriminator with 128 hidden units, 2000 epochs
- **MLP**: 6-layer network, 256 hidden units, BatchNorm, 500 epochs

## Expected Results

If DiT/GAN work well:
- At 5-10% real data: Augmented >> Real Only
- At 50%+ real data: Augmented ≈ Oracle (diminishing returns)
- DiT vs GAN: Should be competitive
- DiT/GAN vs Kriging: Deep models should outperform interpolation

If they don't work:
- Augmented ≈ Real Only (synthetic data not helpful)
- Kriging might outperform if data is simple/smooth

## Configuration
```python
GEN_EPOCHS: 2000  # Train DiT/GAN
MLP_EPOCHS: 500   # Train classifier
TRAIN_RATIOS: [0.05, 0.1, 0.25, 0.5, 0.8, 1.0]
```

## Next Steps
1. Run full experiment
2. Analyze results: Does DiT reduce data collection overhead?
3. If POWDER needed, uncomment and add to experiments
4. Consider adding SMOTE (another baseline) for completeness
