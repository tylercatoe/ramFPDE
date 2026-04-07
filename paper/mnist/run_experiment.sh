#!/bin/bash
# run_experiment.sh - MNIST experiment launcher
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

default_args=(
  --batch_size 128
  --test_batch_size 256
  --nepochs 160
  --lr 0.1
  --momentum 0.9
  --weight_decay 1e-4
  --test_freq 1
  --width 64
  --h 0.1a
  --T 1.0
  --beta 0.6
  --method l1
  --odeint rampde
  --results_dir ./raw_data
)

results_dir="./raw_data"
seed=25

mkdir -p slurm_logs

echo "MNIST Configuration:"
echo "  - Epochs: 160"
echo "  - Batch size: 128"
echo "  - Width: 64"
echo "  - h: 0.1"
echo "  - T: 1.0"
echo "  - beta: 0.6"
echo "  - Seed: $seed"
echo "  - Results dir: $results_dir"
echo ""

echo "Running MNIST experiments"
echo "========================="

# High-precision comparison (no scaling needed)
for precision in "float32" "tfloat32"; do
  fixed_args=(
    --precision "$precision"
    --seed "$seed"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  echo "Submitting: MNIST ${fixed_args[*]}"
  sbatch job_ode_mnist.sbatch "${fixed_args[@]}" "${default_args[@]}"
done

echo "All experiments submitted!"
