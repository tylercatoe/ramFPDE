#!/bin/bash
# run_test_MP.sh - MNIST mixed-precision short test runs for quick validation
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

echo "Running MNIST MP test experiments (shortened)"
echo "============================================="

results_dir="./raw_data"
seed=26

common_args=(
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
  --method l1
  --odeint rampde
  --seed "$seed"
  --results_dir "$results_dir"
  --beta 0.6
)

mkdir -p slurm_logs

echo "Test configuration:"
echo "  - Epochs: 3"
echo "  - Batch size: 64"
echo "  - Width: 64"
echo "  - h: 0.1"
echo "  - T: 1.0"
echo "  - Beta: 0.6"
echo "  - Seed: $seed"
echo "  - Results dir: $results_dir"
echo ""

echo "Submitting: MNIST MP float32 (no scaling)"
sbatch job_ode_mnist_MP.sbatch \
  "${common_args[@]}" \
  --precision float32 \
  --no_grad_scaler \
  --no_dynamic_scaler

echo "Submitting: MNIST MP float16 (dynamic scaler only)"
sbatch job_ode_mnist_MP.sbatch \
  "${common_args[@]}" \
  --precision float16 \
  --no_grad_scaler

echo ""
echo "MNIST MP test jobs submitted."
echo "Monitor progress with:"
echo "  watch -n 30 'squeue -u \$USER | grep mnist_mp'"
echo "  tail -f slurm_logs/ode_mnist_mp_*.out"
