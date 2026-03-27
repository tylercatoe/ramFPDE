#!/bin/bash
#SBATCH --job-name=eval_cifar10_fractional
#SBATCH --output=slurm_logs/eval_cifar10_fractional_%j.out
#SBATCH --error=slurm_logs/eval_cifar10_fractional_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail

module load cuda/12.3.0
module load anaconda3/2023.09-0
eval "$(conda shell.bash hook)"

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"

conda run -n torch28 python -c "import torch, torchvision, pandas, matplotlib; print('env-ok', torch.__version__)"
conda run -n torch28 python -u evaluate_checkpoints.py "$@"
