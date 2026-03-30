#!/bin/bash
#SBATCH --job-name=ode_cifar10_fractional
#SBATCH --output=slurm_logs/ode_cifar10_fractional_%j.out
#SBATCH --error=slurm_logs/ode_cifar10_fractional_%j.err
#SBATCH --partition=work1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
set -euo pipefail

# Load modules and conda shell.
module load cuda/12.3.0
module load anaconda3/2023.09-0
eval "$(conda shell.bash hook)"

# Palmetto guide: cuDNN comes from the environment packages, not a module.
ENV_PATH="${HOME}/.conda/envs/torch28"
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
CUDNN_PATH="${ENV_PATH}/lib/python3.10/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
  export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi


echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the driver script and pass along all arguments.
conda run -n torch28 python -u ode_cifar10.py "$@"

# Wait a few seconds to ensure the driver has finished writing result_dir.txt.
sleep 10

RESULT_FILE="result_dir_${SLURM_JOB_ID}.txt"
if [ -f "${RESULT_FILE}" ]; then
  RESULT_DIR=$(cat "${RESULT_FILE}")
  echo "Result directory: ${RESULT_DIR}"
else
  echo "Warning: ${RESULT_FILE} not found"
fi

# Move the SLURM log files into the results directory.
# mv slurm_logs/ode_cifar10_fractional_${SLURM_JOB_ID}.out ${RESULT_DIR}/slurm_log.out
# mv slurm_logs/ode_cifar10_fractional_${SLURM_JOB_ID}.err ${RESULT_DIR}/slurm_log.err
