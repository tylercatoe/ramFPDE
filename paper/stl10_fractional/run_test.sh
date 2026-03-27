#!/bin/bash
# run_stl10_fractional_test.sh - quick STL10 fractional smoke runs
# Usage: chmod +x run_test.sh && ./run_test.sh
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

echo "Running STL10 fractional test experiments"
echo "========================================="

test_args=(
  --batch_size 4
  --nepochs 3
  --lr 0.05
  --momentum 0.9
  --weight_decay 5e-4
  --test_freq 1
  --width 64
  --results_dir ./raw_data
  --method l1
  --beta 0.6
  --odeint rampde
)

seed=26
mkdir -p slurm_logs

echo "Submitting fractional L1 rampde test matrix..."

for precision in "float32" "tfloat32" "bfloat16"; do
  fixed_args=(
    --precision "$precision"
    --seed "$seed"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  echo "Submitting: rampde $precision no-scaling"
  sbatch job_ode_stl10_fractional.sbatch "${fixed_args[@]}" "${test_args[@]}"
done

fixed_args=(
  --precision "float16"
  --seed "$seed"
  --no_grad_scaler
  --no_dynamic_scaler
)
echo "Submitting: rampde float16 no-scaling (safe uniform fallback)"
sbatch job_ode_stl10_fractional.sbatch "${fixed_args[@]}" "${test_args[@]}"

fixed_args=(
  --precision "float16"
  --seed "$seed"
  --no_dynamic_scaler
)
echo "Submitting: rampde float16 grad-scaler only"
sbatch job_ode_stl10_fractional.sbatch "${fixed_args[@]}" "${test_args[@]}"

fixed_args=(
  --precision "float16"
  --seed "$seed"
  --no_grad_scaler
)
echo "Submitting: rampde float16 dynamic-scaler only"
sbatch job_ode_stl10_fractional.sbatch "${fixed_args[@]}" "${test_args[@]}"

echo ""
echo "Done. To process results:"
echo "  cd paper/stl10_fractional"
echo "  python aggregate_stl10_fractional_results.py --raw-data-dir raw_data/ode_stl10_fractional"
