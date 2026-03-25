#!/bin/bash
# run_evaluation.sh - Evaluate STL10 trained model checkpoints on test set
# Usage: chmod +x run_evaluation.sh ; ./run_evaluation.sh

# Make log directory
mkdir -p slurm_logs

echo "Submitting STL10 Test Set Evaluation Job"
echo "========================================="

# Submit evaluation job for all checkpoints in raw_data/ode_stl10_fractional
sbatch job_evaluate_stl10_fractional.sbatch --results-dir ./raw_data/ode_stl10_fractional

echo "Evaluation job submitted!"
echo "Monitor progress with: tail -f slurm_logs/eval_stl10_fractional_*.out"
