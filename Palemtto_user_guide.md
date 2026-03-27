# Getting Started on Palmetto: Conda Environments & SLURM

**Clemson University HPC — Guide for New Users**

---

## Table of Contents

1. [Logging In](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#1-logging-in)
2. [Understanding the File System](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#2-understanding-the-file-system)
3. [Loading Modules](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#3-loading-modules)
4. [Creating a Conda Environment](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#4-creating-a-conda-environment)
5. [Installing Packages (PyTorch Example)](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#5-installing-packages-pytorch-example)
6. [Using SLURM — Submitting Jobs](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#6-using-slurm--submitting-jobs)
7. [Writing a SLURM Batch Script](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#7-writing-a-slurm-batch-script)
8. [Monitoring Your Jobs](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#8-monitoring-your-jobs)
9. [Common Errors &amp; Fixes](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#9-common-errors--fixes)
10. [Quick Reference Cheatsheet](https://claude.ai/chat/09bda46e-a68f-431c-86ca-750e444a788b#10-quick-reference-cheatsheet)

---

## 1. Logging In

Connect to Palmetto via SSH:

```bash
ssh <your-username>@login.palmetto.clemson.edu
```

You will land on a  **login node** . Do **not** run heavy computation here — always submit jobs through SLURM.

---

## 2. Understanding the File System

| Location        | Path                     | Purpose                           | Notes                             |
| --------------- | ------------------------ | --------------------------------- | --------------------------------- |
| Home directory  | `/home/<username>/`    | Scripts, code, small files        | Limited quota (~100 GB)           |
| Scratch storage | `/scratch/<username>/` | Large files, conda envs, datasets | No quota, but periodically purged |

> **Best practice:** Store your conda environment and large data in `/scratch/<username>/`.

---

## 3. Loading Modules

Palmetto uses the `module` system to manage software. Always load modules before activating your environment.

```bash
# See what's available
module avail

# Check available CUDA versions
module avail cuda

# Check available Anaconda versions
module avail anaconda3

# Load the modules you need
module load cuda/12.3.0
module load anaconda3/2023.09-0

# See what is currently loaded
module list

# Unload everything and start fresh
module purge
```

### Available CUDA versions on Palmetto

```
cuda/11.8.0
cuda/12.3.0   ← (default, recommended)
```

> **Note:** There is no `cudnn` module on Palmetto. If your framework needs cuDNN (e.g., PyTorch), it is bundled inside the pip-installed package — no separate module needed.

---

## 4. Creating a Conda Environment

It is recommended to create your environment in `/scratch` to avoid filling up your home directory.

```bash
# Load anaconda first
module load anaconda3/2023.09-0

# Create a new environment with Python 3.11 in scratch
conda create --prefix /scratch/<your-username>/my-env python=3.11 -y

# Activate it
source activate /scratch/<your-username>/my-env
```

> ⚠️ **Important:** On Palmetto, use `source activate <path>` — **not** `conda activate`. The `source activate` command is what works correctly with the Palmetto module system.

### Deactivating the environment

```bash
conda deactivate
```

---

## 5. Installing Packages (PyTorch Example)

After activating your environment:

```bash
# Activate the environment
source activate /scratch/<your-username>/my-env

# Install PyTorch with CUDA 12.4 support (works with the cuda/12.3.0 module on Palmetto)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify the installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Install other packages normally

```bash
pip install numpy scipy matplotlib pandas
```

---

## 6. Using SLURM — Submitting Jobs

Palmetto uses **SLURM** to schedule and run jobs on compute nodes.

### Key partitions available to general users

| Partition    | Description      | GPU         |
| ------------ | ---------------- | ----------- |
| `work1`    | General CPU jobs | None        |
| `gpu-v100` | V100 GPU nodes   | NVIDIA V100 |
| `gpu-a100` | A100 GPU nodes   | NVIDIA A100 |

### Check what partitions you can use

```bash
sinfo
```

### Run a quick interactive job (for testing)

```bash
# Interactive session on a CPU node (1 hour, 8GB RAM)
srun --partition=work1 --mem=8G --time=01:00:00 --pty bash

# Interactive session with a GPU
srun --partition=gpu-v100 --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
```

---

## 7. Writing a SLURM Batch Script

For longer jobs, write a batch script and submit it with `sbatch`.

### Example: CPU job

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=work1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/<username>/logs/%x-%j.out
#SBATCH --error=/home/<username>/logs/%x-%j.err

# Exit immediately if any command fails
set -eo pipefail

# Load modules
module load anaconda3/2023.09-0

# Set your environment path
ENV_PATH="/scratch/<username>/my-env"

# Activate conda environment (use source activate on Palmetto)
source activate "${ENV_PATH}"

# Run your script
python my_script.py
```

### Example: GPU job

```bash
#!/bin/bash
#SBATCH --job-name=gpu_train
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/<username>/logs/%x-%j.out
#SBATCH --error=/home/<username>/logs/%x-%j.err

set -eo pipefail

# Load modules
module load cuda/12.3.0
module load anaconda3/2023.09-0

ENV_PATH="/scratch/<username>/my-env"
REPO_DIR="/home/<username>/my-project"

# Activate conda environment
source activate "${ENV_PATH}"

# Set library paths (needed if your packages include nvidia/cudnn libs)
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
CUDNN_PATH="${ENV_PATH}/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
    export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi

# Useful environment settings
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg   # Disable display for matplotlib
export OMP_NUM_THREADS=4

# Create logs directory if it doesn't exist
mkdir -p /home/<username>/logs

# Change to project directory
cd "${REPO_DIR}"

# Print diagnostic info (helpful for debugging)
echo "Host: $(hostname)"
echo "GPU:"
nvidia-smi
echo "Modules loaded:"
module list 2>&1
echo "Python: $(python --version)"

# Run your script
python train.py
```

### Submitting the script

```bash
# Submit
sbatch my_job.sh

# You will see something like:
# Submitted batch job 1234567
```

---

## 8. Monitoring Your Jobs

```bash
# Check your currently running/queued jobs
squeue -u <your-username>

# Check detailed info about a specific job
scontrol show job <JOBID>

# Watch job status (refreshes every 5 seconds)
watch -n 5 squeue -u <your-username>

# View live output from a running job
tail -f /home/<username>/logs/my_job-<JOBID>.out

# Cancel a job
scancel <JOBID>

# View completed job info (efficiency, memory used, etc.)
seff <JOBID>

# Check your recent job history
sacct -u <your-username> --format=JobID,JobName,State,Elapsed,MaxRSS
```

---

## 9. Common Errors & Fixes

### ❌ `conda: error: argument COMMAND: invalid choice: 'activate'`

**Cause:** Using `conda activate` inside a SLURM script doesn't work — the shell isn't initialized for conda.

**Fix:** Always use `source activate` in your SLURM scripts on Palmetto:

```bash
# ❌ Wrong
conda activate /scratch/my-env

# ✅ Correct on Palmetto
source activate /scratch/my-env
```

---

### ❌ `ModuleNotFoundError: No module named 'torch'`

**Cause:** Your environment is not activated, or you're using the wrong Python.

**Fix:**

```bash
# Make sure you activate the env AFTER loading anaconda
module load anaconda3/2023.09-0
source activate /scratch/<username>/my-env

# Verify which python is being used
which python   # should point inside your env
```

---

### ❌ `OSError: libcudnn... cannot open shared object file`

**Cause:** The cuDNN shared library path isn't set correctly.

**Fix:** Add these lines to your SLURM script after activating the environment:

```bash
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
CUDNN_PATH="${ENV_PATH}/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ -d "${CUDNN_PATH}" ]; then
    export LD_LIBRARY_PATH="${CUDNN_PATH}:${LD_LIBRARY_PATH}"
fi
```

---

### ❌ `sbatch: error: Batch job submission failed: Invalid partition name`

**Cause:** You're requesting a partition you don't have access to, or the name is wrong.

**Fix:** Run `sinfo` to see available partitions, and use one you have access to (e.g., `work1`, `gpu-v100`).

---

### ❌ Job sits in PENDING state for a long time

**Cause:** The cluster is busy, or you've requested more resources than available.

**Fix:** Check the reason with:

```bash
squeue -u <username> -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"
```

Common reasons: `Resources` (waiting for resources), `Priority` (other jobs ahead), `QOSMaxJobsPerUserLimit` (you hit a job limit).

---

## 10. Quick Reference Cheatsheet

```bash
# ── Environment Setup ──────────────────────────────────────────────
module load cuda/12.3.0                          # Load CUDA
module load anaconda3/2023.09-0                  # Load Anaconda
conda create --prefix /scratch/<user>/env python=3.11 -y   # Create env
source activate /scratch/<user>/env              # Activate env (Palmetto syntax)
conda deactivate                                 # Deactivate env

# ── SLURM Job Control ──────────────────────────────────────────────
sbatch job.sh                                    # Submit a batch job
squeue -u <username>                             # View your jobs
scancel <JOBID>                                  # Cancel a job
seff <JOBID>                                     # Job efficiency report
sacct -u <username>                              # Job history

# ── Interactive Jobs ───────────────────────────────────────────────
srun --partition=work1 --mem=8G --time=01:00:00 --pty bash          # CPU node
srun --partition=gpu-v100 --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash  # GPU node

# ── Monitoring ─────────────────────────────────────────────────────
tail -f logs/myjob-<JOBID>.out                   # Live output
watch -n 5 squeue -u <username>                  # Refresh job status
sinfo                                            # See cluster/partition status
```

---

> **Questions?** Contact your advisor or the Palmetto help desk at hpc@clemson.edu
>
> **Palmetto Documentation:** https://docs.rcd.clemson.edu/palmetto/
>
