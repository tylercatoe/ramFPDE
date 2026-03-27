#!/usr/bin/env python3
"""
Aggregate CIFAR-10 fractional experiment results into a summary CSV.
"""

from pathlib import Path
import argparse

import pandas as pd


def parse_experiment_name(dirname: str):
    # Expected prefix from setup_experiment(data_name="cifar10", experiment="ode_cifar10_fractional")
    # Example:
    #   cifar10_float32_rampde_l1_stable_stable_lr_0.05_nepochs_3_batch_size_16_width_128_seed26_...
    if not dirname.startswith("cifar10_"):
        return None

    parts = dirname.split("_")
    if len(parts) < 4:
        return None

    precision = parts[1]
    if precision == "float16":
        if len(parts) < 5:
            return None
        scaler = parts[2]
        solver = parts[3]
    else:
        scaler = "none"
        solver = parts[2]

    seed = None
    width = None
    for i, part in enumerate(parts):
        if part.startswith("seed"):
            try:
                seed = int(part[4:])
            except ValueError:
                pass
        if part == "width" and i + 1 < len(parts):
            try:
                width = int(parts[i + 1])
            except ValueError:
                pass

    return {
        "directory": dirname,
        "data_name": "cifar10",
        "precision_str": precision,
        "odeint_type": solver,
        "scaler_name": scaler,
        "seed": seed,
        "width": width,
    }


def extract_metrics_from_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None

        final = df.iloc[-1]
        metrics = {
            "val_acc": final["val_acc"] if "val_acc" in final else None,
            "train_acc": final["train_acc"] if "train_acc" in final else None,
            "val_loss": final["val_loss"] if "val_loss" in final else None,
            "train_loss": final["train_loss"] if "train_loss" in final else None,
            "time_fwd": final["time_fwd"] if "time_fwd" in final else None,
            "time_bwd": final["time_bwd"] if "time_bwd" in final else None,
            "time_fwd_sum": final["time_fwd_sum"] if "time_fwd_sum" in final else None,
            "time_bwd_sum": final["time_bwd_sum"] if "time_bwd_sum" in final else None,
            "max_memory_mb": final["max_memory_mb"] if "max_memory_mb" in final else None,
            "epoch": int(final["epoch"]) if "epoch" in final else None,
            "iter": int(final["iter"]) if "iter" in final else None,
        }

        args_csv = csv_path.parent / "args.csv"
        if args_csv.exists():
            args_df = pd.read_csv(args_csv)
            if len(args_df) > 0:
                args_row = args_df.iloc[0]
                for col in [
                    "method",
                    "gpu",
                    "timestamp",
                    "job_id",
                    "tol",
                    "nepochs",
                    "lr",
                    "momentum",
                    "batch_size",
                    "test_batch_size",
                    "weight_decay",
                    "precision",
                    "odeint",
                    "unstable",
                    "no_grad_scaler",
                    "no_dynamic_scaler",
                    "results_dir",
                    "debug",
                    "test_freq",
                    "stable",
                ]:
                    if col in args_row:
                        metrics[col] = args_row[col]
        return metrics
    except Exception as exc:
        print(f"Error reading {csv_path}: {exc}")
        return None


def aggregate_cifar10_results(raw_data_dir: str, seed_filter=None, width_filter=None):
    raw_data_path = Path(raw_data_dir)
    if not raw_data_path.exists():
        print(f"Error: Directory {raw_data_dir} does not exist")
        return None

    results = []
    for exp_dir in sorted(raw_data_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        dirname = exp_dir.name
        if not dirname.startswith("cifar10_"):
            continue

        params = parse_experiment_name(dirname)
        if params is None:
            print(f"Skipping {dirname} - could not parse")
            continue

        if seed_filter is not None and params["seed"] != seed_filter:
            continue
        if width_filter is not None and params["width"] != width_filter:
            continue

        csv_files = [file_path for file_path in exp_dir.glob("*.csv") if file_path.name != "args.csv"]
        if len(csv_files) == 0:
            print(f"Warning: No data CSV file found in {dirname}")
            continue

        metrics = extract_metrics_from_csv(csv_files[0])
        if metrics is None:
            continue

        results.append({**params, **metrics})
        scaler_str = f" - {params['scaler_name']}" if params["scaler_name"] != "none" else ""
        print(
            f"Processed: {params['precision_str']} - {params['odeint_type']}{scaler_str} "
            f"(seed={params['seed']}, width={params['width']})"
        )

    if len(results) == 0:
        print("No results found!")
        return None

    df = pd.DataFrame(results)
    precision_order = ["float32", "tfloat32", "bfloat16", "float16"]
    scaler_order = ["none", "grad", "dynamic"]
    df["precision_order"] = df["precision_str"].map({name: i for i, name in enumerate(precision_order)})
    df["scaler_order"] = df["scaler_name"].map({name: i for i, name in enumerate(scaler_order)})
    df = df.sort_values(["precision_order", "odeint_type", "scaler_order", "seed", "width"])
    return df.drop(["precision_order", "scaler_order"], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Aggregate CIFAR-10 fractional experiment results")
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="raw_data/ode_cifar10_fractional",
        help="Directory containing raw experiment data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="raw_data/ode_cifar10_fractional/summary_ode_cifar10_fractional.csv",
        help="Output CSV file path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Filter results by random seed")
    parser.add_argument("--width", type=int, default=None, help="Filter results by network width")
    args = parser.parse_args()

    print("Aggregating CIFAR-10 fractional experiment results...")
    print(f"Raw data directory: {args.raw_data_dir}")
    if args.seed is not None:
        print(f"Filtering by seed: {args.seed}")
    if args.width is not None:
        print(f"Filtering by width: {args.width}")

    df = aggregate_cifar10_results(args.raw_data_dir, seed_filter=args.seed, width_filter=args.width)
    if df is None:
        print("Failed to aggregate results")
        return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved summary to: {args.output}")
    print(f"Total experiments: {len(df)}")
    print("\nSummary by configuration:")
    print(df.groupby(["precision_str", "odeint_type", "scaler_name"]).size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
