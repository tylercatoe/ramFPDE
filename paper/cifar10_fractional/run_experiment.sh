#!/bin/bash
# run_cifar10.sh - CIFAR10 gradient scaling comparison experiments
# Usage: chmod +x run_cifar10.sh ; ./run_cifar10.sh
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

# Default training arguments
default_args=(
  --batch_size  16 
  --nepochs   160
  --lr 0.05
  --momentum 0.9
  --weight_decay 1e-4
  --test_freq 10
  --width 128
  --results_dir ./raw_data
)

# Use test results directory
results_dir="./raw_data"
echo "Results will be saved to: $results_dir"

# Seed
seed=25

# Make log directory
mkdir -p slurm_logs
echo "CIFAR10 Test Configuration:"
echo "  -   Epochs: 160"
echo "  -   Batch size: 16"
echo "  -   Test frequency: every 10 epochs"
echo "  -   Seed: $seed (vs 25 in production)"
echo "  -   Results dir: $results_dir"
echo ""

echo "Running CIFAR10 Experiment in High Precision"
echo "=========================================================="

# Test: rampde with no scaling in various precisions
echo "Test 1: No scaling comparison - float32, tfloat32"
for precision in "float32" "tfloat32"; do
    fixed_args=(
        --precision "$precision"
        --seed "$seed"
    )
    echo "Submitting: CIFAR-10 ${fixed_args[*]}"
    sbatch job_ode_cifar10_fractional.sbatch "${fixed_args[@]}" "${default_args[@]}"
done

# Remove wait commands since we're using sbatch instead of background jobs
echo "All experiments submitted!"