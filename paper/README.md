# Paper Experiments

This directory contains all code, scripts, and instructions on reproducing the various experiments. 

## Structure
Each experiment has its own subdirectory with:

-   **Experiment code**: Python files to run the experiment
-   **Run scripts**:
    -   `run_test.sh`: Quick validation test (2-3 ephocs/iterations)
    -   `run_experiment.sh`: Full experiment run
-   **SLURM job files**: `.sbatch` files for HPC
-   **Processing scripts**: Python files to generate figures and tables
-   **Data directories**: 
    -   `raw_data/`: Experiment outputs (created by run scripts)
    -   `outputs/`: LaTeX figures and tables created by processing scripts

## Experiments
### 1. CIFAR-10 Image Classification 
**Directory**: 
`cifar10_fractional/`

**Purpose**: 
Evaluate fractional neural ODE performace on CIFAR-10 image classification across precision configurations.

-   Currenlty only implemented for float32 and tfloat32.

**Outputs**:
-   `fig_cifar10_train_loss/`: Training convergence plots
-   `tab_cifar10_results/`: Performance comparison table

## Running Experiments

### Quick Test
```bash
cd paper/cifar10_fractional
./run_test.sh
```

**Key features of test runs**:

### Full Experiments 
```bash
cd paper/cifar10_fractional
./run_experiment.sh
```

### Generating Figures and Tables

After experiments complete, each experiment has a `process_results.sh` script that generates all figures and tables for that experiment:
```bash
# Process individual experiments
cd paper/cifar10_fractional
./process_results.sh 
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **SLURM**: HPC cluster with SLURM job scheduler
- **Conda**: Conda environment `torch28`

### Hardware Specifications

All experiments and runtime estimates in this documentation were performed on:

- **GPU**: ???
- **CUDA**: Compatible CUDA version with PyTorch 2.8

## Environment Setup

```bash
# Activate conda environment
conda activate torch28

........
```


## Dependencies
...


## Code Organization

### Utility Modules

- **`experiment_runtime.py`**: Runtime utilities for RUNNING experiments

  - Environment setup and ODE solver imports
  - Precision configuration (float32, tfloat32, float16, bfloat16)
  - Gradient scaler setup (GradScaler, DynamicScaler)
  - Experiment directory creation and logging
  - Training utility classes (RunningAverageMeter, etc.)
  - **Used by**: ode_cifar10.py

- **`analysis_utils.py`**: Analysis utilities for PROCESSING results
  - Parsing experiment directory names
  - Loading experiment results from CSV files
  - Creating legend labels for plots
  - **Used by**: Processing scripts that generate figures and tables

  ### File Structure

```
paper/
```

## Expected Runtimes

### Test Runs (Reduced Iterations)

### Full Production Runs

## Notes