#!/bin/bash
# run_stl10_fractional.sh - STL10 fractional (L1) rampde experiments
# Usage: chmod +x run_experiment.sh ; ./run_experiment.sh

default_args=(
  --batch_size 16
  --nepochs 160
  --lr 0.05
  --momentum 0.9
  --weight_decay 1e-4
  --width 128
  --results_dir ./raw_data
  --method l1
  --beta 0.6
  --odeint rampde
)

seed=25
mkdir -p slurm_logs

echo "Running STL10 fractional rampde experiments"
echo "==========================================="

echo "Test 1: Non-fp16 precisions (unscaled uniform)"
for precision in "float32" "tfloat32" "bfloat16"; do
  fixed_args=(
    --precision "$precision"
    --seed "$seed"
    --no_grad_scaler
    --no_dynamic_scaler
  )
  echo "Submitting: rampde $precision no-scaling"
  sbatch job_ode_stl10_fractional.sbatch "${fixed_args[@]}" "${default_args[@]}"
done

echo "Test 2: fp16 scaling variants"
fixed_args=(
  --precision "float16"
  --seed "$seed"
  --no_grad_scaler
  --no_dynamic_scaler
)
echo "Submitting: rampde float16 no-scaling (safe uniform fallback)"
sbatch job_ode_stl10_fractional.sbatch "${fixed_args[@]}" "${default_args[@]}"

fixed_args=(
  --precision "float16"
  --seed "$seed"
  --no_dynamic_scaler
)
echo "Submitting: rampde float16 grad-scaler only"
sbatch job_ode_stl10_fractional.sbatch "${fixed_args[@]}" "${default_args[@]}"

fixed_args=(
  --precision "float16"
  --seed "$seed"
  --no_grad_scaler
)
echo "Submitting: rampde float16 dynamic-scaler only"
sbatch job_ode_stl10_fractional.sbatch "${fixed_args[@]}" "${default_args[@]}"

echo "All experiments submitted."
