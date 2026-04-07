#!/bin/bash
# run_test.sh - Peaks short test run for quick validation
set -euo pipefail

module load cuda/12.3.0
module load anaconda3/2023.09-0
eval "$(conda shell.bash hook)"

if ! conda run -n torch28 python -c "import torch, pandas, matplotlib" >/dev/null 2>&1; then
  echo "Missing dependencies in env 'torch28'. Run:"
  echo "  conda run -n torch28 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
  echo "  conda run -n torch28 pip install pandas matplotlib"
  echo "  conda run -n torch28 pip install -e /home/tcatoe/home_FDNN/ramFPDE/ramFPDE"
  exit 1
fi

echo "Running MNIST test experiment (shortened)"
echo "========================================="

results_dir="./raw_data"
seed=26

test_args_adjoint=(
  --batch_size 64
  --test_batch_size 128
  --nepochs 3
  --lr 0.05
  --momentum 0.9
  --weight_decay 1e-4
  --test_freq 1
  --width 64
  --h 0.1
  --T 1.0
  --precision float32
  --method l1
  --odeint rampde
  --seed "$seed"
  --no_grad_scaler
  --no_dynamic_scaler
  --results_dir "$results_dir"
  --beta 0.6
  --adjoint
)

test_args_backprop=(
  --batch_size 64
  --test_batch_size 128
  --nepochs 3
  --lr 0.05
  --momentum 0.9
  --weight_decay 1e-4
  --test_freq 1
  --width 64
  --h 0.1
  --T 1.0
  --precision float32
  --method l1
  --odeint torchfde
  --seed "$seed"
  --no_grad_scaler
  --no_dynamic_scaler
  --results_dir "$results_dir"
  --beta 0.6
)

mkdir -p slurm_logs

echo "Test configuration:"
echo "  - Beta: 0.6"
echo "  - Epochs: 3"
echo "  - Batch size: 64"
echo "  - Precision: float32"
echo "  - ODE backend: rampde"
echo "  - Seed: $seed"
echo "  - Results dir: $results_dir"
echo ""

echo "Submitting: MNIST float32 adjoint test"
sbatch job_ode_mnist.sbatch "${test_args_adjoint[@]}"

echo "Submitting: MNIST float32 backprop test"
sbatch job_ode_mnist.sbatch "${test_args_backprop[@]}"
 
echo ""
echo "MNIST tests submitted."
echo "Monitor progress with:"
echo "  watch -n 30 'squeue -u \$USER | grep mnist'"
echo "  tail -f slurm_logs/ode_mnist_*.out"