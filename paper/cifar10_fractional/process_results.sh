#!/bin/bash
#
# CIFAR-10 Fractional Results Processing Script
#
# Usage:
#   ./process_results.sh [SEED] [WIDTH]
#
set -euo pipefail

SEED=${1:-25}
WIDTH=${2:-128}

echo "=================================="
echo "CIFAR-10 Fractional Results Processing"
echo "=================================="
echo "Using seed: $SEED"
echo "Using width: $WIDTH"
echo ""

if [ ! -f "ode_cifar10.py" ]; then
  echo "Error: Must run from paper/cifar10_fractional/ directory"
  exit 1
fi

module load anaconda3/2023.09-0
eval "$(conda shell.bash hook)"

mkdir -p outputs/fig_cifar10_train_loss
mkdir -p outputs/tab_cifar10_results

echo "Step 1: Aggregating raw experiment data into summary CSV..."
conda run -n torch28 python aggregate_cifar10_fractional_results.py \
  --raw-data-dir raw_data/ode_cifar10_fractional \
  --output raw_data/ode_cifar10_fractional/summary_ode_cifar10_fractional.csv \
  --seed "$SEED" \
  --width "$WIDTH"

echo ""
echo "Step 2: Generating CIFAR-10 convergence plots..."
conda run -n torch28 python plot_cifar10_convergence.py --filter "*seed${SEED}*"

echo ""
echo "Step 3: Generating CIFAR-10 results table..."
conda run -n torch28 python generate_cifar10_table.py \
  --csv-file ./raw_data/ode_cifar10_fractional/summary_ode_cifar10_fractional.csv \
  --results-dir ./raw_data/ode_cifar10_fractional \
  --width "$WIDTH" \
  --seed "$SEED"

echo ""
echo "Step 4: Compiling PDFs (if pdflatex is available)..."
if command -v pdflatex >/dev/null 2>&1; then
  cd outputs/fig_cifar10_train_loss
  if [ -f "cifar10_train_loss_standalone.tex" ]; then
    pdflatex -interaction=batchmode cifar10_train_loss_standalone.tex >/dev/null 2>&1 || true
    pdflatex -interaction=batchmode cifar10_train_loss_standalone.tex >/dev/null 2>&1 || true
  fi
  cd ../..

  cd outputs/tab_cifar10_results
  if [ -f "cifar10_results_table_standalone.tex" ]; then
    pdflatex -interaction=batchmode cifar10_results_table_standalone.tex >/dev/null 2>&1 || true
  fi
  cd ../..
else
  echo "pdflatex not found; skipping PDF compilation."
fi

echo ""
echo "Step 5: Copying summary CSV..."
cp raw_data/ode_cifar10_fractional/summary_ode_cifar10_fractional.csv outputs/cifar10_results_seed${SEED}.csv

echo ""
echo "=================================="
echo "CIFAR-10 Fractional Processing Complete!"
echo "=================================="
echo "Outputs:"
echo "  - raw_data/ode_cifar10_fractional/summary_ode_cifar10_fractional.csv"
echo "  - outputs/fig_cifar10_train_loss/"
echo "  - outputs/tab_cifar10_results/"
echo "  - outputs/cifar10_results_seed${SEED}.csv"
