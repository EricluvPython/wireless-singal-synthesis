#!/bin/bash
# Quick start script for DiT Data Fidelity Experiment

set -e  # Exit on error

echo "======================================================================"
echo "DiT Data Fidelity Experiment - Quick Start"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import torch, numpy, pandas, sklearn, matplotlib" 2>/dev/null || {
    echo "Missing dependencies. Installing from requirements.txt..."
    pip install -r requirements.txt
}

echo "✓ Dependencies OK"
echo ""

# Parse command line arguments
MODE=${1:-full}
DATASET=${2:-indoor}

case $MODE in
    full)
        echo "Running full experiment on both datasets..."
        echo "This will take several hours depending on your hardware."
        echo ""
        python3 experiment.py --datasets indoor outdoor
        ;;
    
    quick)
        echo "Running quick test (reduced epochs)..."
        echo "Dataset: $DATASET"
        echo ""
        python3 experiment.py \
            --datasets $DATASET \
            --gen-epochs 100 \
            --mlp-epochs 20
        ;;
    
    train-dit)
        echo "Training DiT model only..."
        echo "Dataset: $DATASET"
        echo ""
        python3 train_model.py dit --dataset $DATASET --epochs 2000
        ;;
    
    train-gan)
        echo "Training GAN model only..."
        echo "Dataset: $DATASET"
        echo ""
        python3 train_model.py gan --dataset $DATASET --epochs 2000
        ;;
    
    examples)
        echo "Running usage examples..."
        echo ""
        python3 examples.py
        ;;
    
    help|*)
        echo "Usage: $0 [mode] [dataset]"
        echo ""
        echo "Modes:"
        echo "  full         - Run complete experiment (default)"
        echo "  quick        - Quick test with reduced epochs"
        echo "  train-dit    - Train DiT model only"
        echo "  train-gan    - Train GAN model only"
        echo "  examples     - Run usage examples"
        echo "  help         - Show this help"
        echo ""
        echo "Datasets:"
        echo "  indoor       - UCI WiFi Indoor dataset (default)"
        echo "  outdoor      - POWDER Outdoor dataset"
        echo ""
        echo "Examples:"
        echo "  $0 full                    # Full experiment, both datasets"
        echo "  $0 quick indoor            # Quick test on indoor dataset"
        echo "  $0 train-dit outdoor       # Train DiT on outdoor data"
        echo "  $0 examples                # Run usage examples"
        echo ""
        exit 0
        ;;
esac

echo ""
echo "======================================================================"
echo "✓ Complete!"
echo "======================================================================"
echo ""
echo "Results saved in: ./work_dir/"
echo "Checkpoints saved in: ./checkpoints/"
echo ""
