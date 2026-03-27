#!/bin/bash
# run_evaluation.sh - Evaluate STL10 trained model checkpoints on test set
# Usage: chmod +x run_evaluation.sh ; ./run_evaluation.sh
set -euo pipefail

module load anaconda3/2023.09-0
eval "$(conda shell.bash hook)"

if ! conda run -n torch28 python -c "import torch, torchvision, pandas, matplotlib" >/dev/null 2>&1; then
  echo "Missing dependencies in env 'torch28'. Run:"
  echo "  conda run -n torch28 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
  echo "  conda run -n torch28 pip install pandas matplotlib"
  echo "  conda run -n torch28 pip install -e /home/tcatoe/home_FDNN/ramFPDE/ramFPDE"
  exit 1
fi

# Make log directory
mkdir -p slurm_logs

echo "Submitting STL10 Test Set Evaluation Job"
echo "========================================="

# Submit evaluation job for all checkpoints in raw_data/ode_stl10_fractional
sbatch job_evaluate_stl10_fractional.sbatch --results-dir ./raw_data/ode_stl10_fractional

echo "Evaluation job submitted!"
echo "Monitor progress with: tail -f slurm_logs/eval_stl10_fractional_*.out"
