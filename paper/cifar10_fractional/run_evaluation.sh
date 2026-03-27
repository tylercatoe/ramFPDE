#!/bin/bash
# run_evaluation.sh - Evaluate CIFAR-10 fractional checkpoints on test set
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

mkdir -p slurm_logs
echo "Submitting CIFAR-10 fractional test-set evaluation job"
echo "======================================================="
sbatch job_evaluate_cifar10_fractional.sbatch --results-dir ./raw_data/ode_cifar10_fractional
echo "Evaluation job submitted!"
echo "Monitor progress with: tail -f slurm_logs/eval_cifar10_fractional_*.out"
