#!/usr/bin/env python3
"""
Evaluate trained CIFAR-10 fractional model checkpoints on the test set.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.amp import autocast

# Add parent directory to path for common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_runtime import setup_environment, get_precision_dtype

# Import training definitions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ode_cifar10 import MPNODE_CIFAR10, get_cifar10_loaders, accuracy


def evaluate_checkpoint(ckpt_path: str, device: str = "cuda:0"):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return None, None

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        args = checkpoint["args"]

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        odeint_func, DynamicScaler = setup_environment(args.odeint, base_dir)
        precision = get_precision_dtype(args.precision)

        grad_scaler_enabled = not args.no_grad_scaler
        dynamic_scaler_enabled = not args.no_dynamic_scaler

        model = MPNODE_CIFAR10(
            args.width,
            args,
            precision,
            odeint_func=odeint_func,
            ScalerClass=DynamicScaler,
            dynamic_scaler_enabled=dynamic_scaler_enabled,
            grad_scaler_enabled=grad_scaler_enabled,
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        _, test_loader, _ = get_cifar10_loaders(
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            seed=args.seed,
        )

        with torch.no_grad():
            amp_enabled = precision in (torch.float16, torch.bfloat16)
            with autocast(device_type="cuda", dtype=precision, enabled=amp_enabled):
                test_acc, test_loss = accuracy(model, test_loader, device)

        return test_acc, test_loss
    except Exception as exc:
        print(f"Error evaluating {ckpt_path}: {exc}")
        import traceback

        traceback.print_exc()
        return None, None


def process_experiment_directory(exp_dir: str, device: str = "cuda:0") -> bool:
    exp_path = Path(exp_dir)
    if not exp_path.is_dir():
        print(f"Not a directory: {exp_dir}")
        return False

    test_loss_file = exp_path / "test_loss.txt"
    ckpt_file = exp_path / "ckpt.pth"

    if test_loss_file.exists():
        print(f"✓ Test metrics already exist: {exp_path.name}")
        return True

    if not ckpt_file.exists():
        print(f"✗ No checkpoint found: {exp_path.name}")
        return False

    print(f"Evaluating: {exp_path.name}")
    test_acc, test_loss = evaluate_checkpoint(str(ckpt_file), device=device)
    if test_acc is None or test_loss is None:
        print(f"✗ Evaluation failed: {exp_path.name}")
        return False

    with open(test_loss_file, "w") as file_obj:
        file_obj.write(f"Test Loss: {test_loss:.6f}\n")
        file_obj.write(f"Test Accuracy: {test_acc:.6f}\n")

    print(f"✓ Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 fractional checkpoints")
    parser.add_argument(
        "--results-dir",
        default="./raw_data/ode_cifar10_fractional",
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument("--device", default="cuda:0", help="Device to use for evaluation")
    parser.add_argument("--exp-dir", default=None, help="Single experiment directory to evaluate")
    args = parser.parse_args()

    if args.exp_dir:
        return 0 if process_experiment_directory(args.exp_dir, args.device) else 1

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        return 1

    exp_dirs = [item for item in sorted(results_path.iterdir()) if item.is_dir() and (item / "ckpt.pth").exists()]
    if not exp_dirs:
        print(f"No experiment directories with checkpoints found in {results_path}")
        return 1

    print(f"Found {len(exp_dirs)} experiment directories with checkpoints\n")
    successes = 0
    failures = 0
    skipped = 0
    for exp_dir in exp_dirs:
        if (exp_dir / "test_loss.txt").exists():
            skipped += 1
            continue
        if process_experiment_directory(str(exp_dir), args.device):
            successes += 1
        else:
            failures += 1
        print()

    print("=" * 60)
    print("Evaluation complete:")
    print(f"  Successful: {successes}")
    print(f"  Failed: {failures}")
    print(f"  Skipped (already evaluated): {skipped}")
    print(f"  Total: {len(exp_dirs)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
