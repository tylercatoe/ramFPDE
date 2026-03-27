#!/bin/bash
# run_cifar10_test.sh - CIFAR10 test run with minimal epochs for quick validation
# Based on run_cifar10.sh but optimized for fast testing
# Usage: chmod +x run_cifar10_test.sh && ./run_cifar10_test
set -euo pipefail

module load cuda/12.3.0
module load anaconda3/2023.09-0
eval "$(conda shell.bash hook)"

if ! conda run -n torch28 python -c "import torch, torchvision, pandas, matplotlib" >/dev/null 2>&1; then
  echo "Missing dependencies in env 'torch28'. Run:"
  echo "  conda run -n torch28 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
  echo "  conda run -n torch28 pip install pandas matplotlib"
  echo "  conda run -n torch28 pip install -e /home/tcatoe/home_FDNN/ramFPDE/ramFPDE"
  exit 1
fi

echo "Running CIFAR10 Test Experiments (shortened for quick validation)"
echo "============================================================="

# Test training arguments - kept minimal but enough for plotting
test_args=(
  --batch_size  4      # Keep small batch size for memory efficiency
  --nepochs   3         # Enough epochs to show convergence trend
  --lr 0.05
  --momentum 0.9
  --weight_decay 5e-4
  --test_freq 1         # Test every epoch to get 3 data points
  --width 64
)

# Use test results directory
results_dir="./raw_data"
echo "Results will be saved to: $results_dir"

# Seed
seed=26  # Different from production (25) to distinguish test runs

# Make log directory
mkdir -p slurm_logs
echo "CIFAR10 Test Configuration:"
echo "  -   Epochs: 3 (vs 100+ in production runs)"
echo "  -   Batch size: 4 (memory efficient for testing)"
echo "  -   Test frequency: every epoch (3 validation points)"
echo "  -   Seed: $seed (vs 25 in production)"
echo "  -   Results dir: $results_dir"
echo ""

# Test 1: High precision
echo "Test 1: High precision (float32) with no scaling"
fixed_args=(
    --precision "float32" 
    --seed "$seed"
    --results_dir "$results_dir"
)
echo "Submitting: rampde float32 no-scaling"
sbatch job_ode_cifar10_fractional.sbatch "${fixed_args[@]}" "${test_args[@]}"





echo ""
echo "CIFAR-10 High Precision Test Submitted: rampde float32 no-scaling"
echo "Expected runtime: ~15-20 min per run (depends on dataset download)"
echo "Expected output: 3 validation points per experiment"
echo ""
echo "To generate figures after completion:"
echo "  cd paper/cifar10_fractional"
echo "  python plot_cifar10_convergence.py"
echo "  python generate_cifar10_table.py"
echo ""
echo "Monitor progress with:"
echo "  watch -n 30 'squeue -u \$USER | grep cifar10'"
echo "  tail -f slurm_logs/ode_cifar10_*.out"
