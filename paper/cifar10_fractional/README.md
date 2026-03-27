# CIFAR-10 Image Classification Experiment

Neural Fractional ODE image classification on CIFAR-10 dataset,
currently only in high precision

## Overview

This experiment evaluates:
-   `rampFDE` FODE L1 solver
-   float32 precision

Dataset: CIFAR-10:
    - 60,000 32x32 color images in 10 classes, with 6,000 images per class. 
        - Upscaled to 128x128 for better CUDA utilization
    - 50,000 training images, 10,000 test images

## Files

-   `ode_cifar10.py`: Neural FODE classifier training script
-   `run_experiment.sh`: Full experiment runner (submits SLURM jobs)
-   `run_test.sh`: Quick test runner (local execution, 3 epochs)
-   `job_ode_cifar10.sbatch`: SLURMM batch job template
-   `plot_cifar10_convergence.py`: Generate training convergence plots
-   `generate_cifar10_table.py`: Generate performance comparison table

## Quick Test

```bash 
./run_test.sh
```

**Expected runtime**: ?? 20-30 ?? minutes per job

**What it does**:
-   Trains for 3 epochs (vs 100+ in production)
-   Tests every epoch (3 validation points)
-   Uses smaller batch size (16) for faster execution
-   Generates data for convergence polts and tables

## Full Experiment

```bash
./run_experiment.sh
```

**Expected runtimes**: ?? 5 - 20 ?? hours per configuration depending on GPU
**What it does**: 
-   Trains neural FODE classifier for 160 epochs

## Evaluation (Computing Test Losses)

After training completes, evaluate the trained model checkpoints on the test set:

```bash
./run_evalution.sh
```

**What it does**: 
-   Loads trained model checkpoints from `raw_data/ode_cifar10/`
-   Evaluates each checkpoint on the CIFAR-10 test set
-   Computes and saves test losses and accuracies to `test_loss.txt` and `test_acc.txt`
-   Required before running processing scripts to generate tables

**Note**: This step is necessary to populate the test metrics that appera in the final results table.

## Expected Outputs

### Raw Data (`raw_data/`)
-   Experiment directories named by configuration (e.g., `ode_cifar10_rampde_float32_seed32_width128`)
-   Each contains:
    -   `summary_ode_cifar10.csv`: Metrics
    -   `train_loss.txt`, `val_loss.txt`, `val_acc.txt`: Training history
    -   `test_loss.txt`, `test_acc.txt`: Final test metrics
    -   `config.json`: Experiment configuration

### Processed Outputs (`outputs/`)

After running processing scripts:

```bash
python plot_cifar10_convergence.py
python generate_cifar10_table.py
```

**Figures** (`outputs/fig_cifar10_train_loss/`):
-   `cifar10_train_loss_convergence.tex`: TikZ convergence plot
-   `cifar10_train_loss_convergence.pdf`: Compiled PDF
-   Supporting CSV data files

**Tables** (`outputs/tab_cifar10_results/`):
- `cifar10_results_table.tex`: LaTeX performance table
- `cifar10_results_table_standalone.tex`: Standalone compilable version

## Configuration

Edit `run_experiment.sh` to modify:
-   Precision modes: uncomment/comment configurations
-   Network width: `--width` argument
-   Epochs: `--epochs` argument
-   Learning rate: `--lr` argument
-   Seed: `seed` variable

Default hyperparameters:
-   Epochs: 100
-   Batch size: 64
-   Learning rate: 0.05
-   Momentum: 0.9
-   Weight decay: 1e-4
-   Width: 128


## Notes

-   Test script uses `seed=26` and 3 epochs for quick validation
-   Production runs use `seed=25` and 160 epochs
-   SLURM account configured in `job_ode_cifar10.sbatch`
-   Results saved to `./raw_data` by default
-   Requires GPU, CUDA, and conda environment `torch28`
-   STL10 dataset will be downloaded automatically to `~/data/cifar10`