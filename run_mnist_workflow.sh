#!/bin/bash
# Complete MNIST experiments workflow

set -e  # Exit on error

echo "======================================================================"
echo "MNIST Physics-Guided Loss Experiments - Complete Workflow"
echo "======================================================================"
echo ""

# Step 1: Precompute priors
echo "Step 1/4: Precomputing MNIST priors..."
echo "----------------------------------------------------------------------"
python mnist_stats.py
echo ""

# Step 2: Test losses
echo "Step 2/4: Testing physics loss functions..."
echo "----------------------------------------------------------------------"
python physics_losses.py
echo ""

# Step 3: Quick demo
echo "Step 3/4: Running quick demo (1 epoch)..."
echo "----------------------------------------------------------------------"
python demo_mnist.py
echo ""

# Step 4: Full experiments (optional, commented out by default)
echo "Step 4/4: Full experiments (skipped - uncomment to run)"
echo "----------------------------------------------------------------------"
echo "To run full experiments (10 epochs each), execute:"
echo "  python train_mnist.py both --epochs 10"
echo "  python evaluate_mnist.py --n-samples 1000"
echo ""

echo "======================================================================"
echo "Workflow completed successfully!"
echo "======================================================================"
echo ""
echo "Summary:"
echo "  ✓ MNIST priors computed and saved to work_dir/mnist_priors.pt"
echo "  ✓ Physics loss functions validated"
echo "  ✓ Demo completed (baseline + physics-guided)"
echo ""
echo "Next steps:"
echo "  1. Run full experiments: python train_mnist.py both --epochs 10"
echo "  2. Evaluate models: python evaluate_mnist.py --n-samples 1000"
echo "  3. Check MNIST_EXPERIMENTS.md for detailed documentation"
echo ""
echo "======================================================================"
