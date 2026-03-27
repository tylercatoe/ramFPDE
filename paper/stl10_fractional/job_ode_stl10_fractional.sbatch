#!/bin/bash
#SBATCH --job-name=ode_stl10_fractional
#SBATCH --output=slurm_logs/ode_stl10_fractional_%j.out
#SBATCH --error=slurm_logs/ode_stl10_fractional_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
set -euo pipefail

# Load conda from module system.
module load anaconda3/2023.09-0
eval "$(conda shell.bash hook)"

echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"

# Validate runtime environment once per job.
conda run -n torch28 python -c "import torch, torchvision, pandas, matplotlib; print('env-ok', torch.__version__)"


nvidia-smi
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

conda run -n torch28 python - << 'PY'
import torch
print("PyTorch version:", torch.__version__, "CUDA", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
torch.backends.cudnn.enabled = False
print("cudnn enabled (sanity check):", torch.backends.cudnn.enabled)
x = torch.randn(4,3,64,64, device='cuda')
w = torch.randn(8, 3, 3, 3, device='cuda')
y = torch.nn.functional.conv2d(x, w, padding=1)
print("conv ok:", y.shape)
PY



export TORCH_CUDNN_V8_API_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the driver script and pass along all arguments.
conda run -n torch28 python -u ode_stl10_fractional.py "$@"

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
# mv slurm_logs/ode_mnist_${SLURM_JOB_ID}.out ${RESULT_DIR}/slurm_log.out
# mv slurm_logs/ode_mnist_${SLURM_JOB_ID}.err ${RESULT_DIR}/slurm_log.err
