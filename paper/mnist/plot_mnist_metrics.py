#!/usr/bin/env python3
"""
Plot MNIST training metrics from experiment logs.

This script is designed for the current MNIST pipeline where periodic metrics are
logged as text lines (not per-iteration CSV).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_mpl_cache = Path(tempfile.gettempdir()) / "matplotlib-cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FLOAT_TOKEN = r"(?:[-+0-9.eE]+|nan|inf|-inf)"

EPOCH_LINE_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+"
    rf"Time\s+(?P<time_val>{FLOAT_TOKEN})\s+\((?P<time_avg>{FLOAT_TOKEN})\)\s+\|\s+"
    rf"NFE-F\s+(?P<nfe_f>{FLOAT_TOKEN})\s+\|\s+"
    rf"NFE-B\s+(?P<nfe_b>{FLOAT_TOKEN})"
    rf"(?:\s+\|\s+Loss\s+(?P<train_loss>{FLOAT_TOKEN}))?\s+\|\s+"
    rf"Train Acc\s+(?P<train_acc>{FLOAT_TOKEN})\s+\|\s+"
    rf"Test Acc\s+(?P<test_acc>{FLOAT_TOKEN})"
)

FINAL_METRICS_RE = re.compile(
    r"FINAL_METRICS\s+\|\s+Method\s+(?P<method>\w+)\s+\|\s+"
    rf"TestErrorPct\s+(?P<test_error_pct>{FLOAT_TOKEN})\s+\|\s+"
    rf"TrainGPUMemMB\s+(?P<train_gpu_mem_mb>{FLOAT_TOKEN})\s+\|\s+"
    rf"TrainTimeSec\s+(?P<train_time_sec>{FLOAT_TOKEN})\s+\|\s+"
    rf"InferenceGPUMemMB\s+(?P<inference_gpu_mem_mb>{FLOAT_TOKEN})\s+\|\s+"
    rf"InferenceTimeSec\s+(?P<inference_time_sec>{FLOAT_TOKEN})\s+\|\s+"
    rf"InferenceTestAcc\s+(?P<inference_test_acc>{FLOAT_TOKEN})"
)


@dataclass
class RunData:
    run_dir: Path
    args_row: pd.Series
    metrics: pd.DataFrame
    final_metrics: dict[str, float]


def _find_log_file(run_dir: Path) -> Optional[Path]:
    preferred = run_dir / f"{run_dir.name}.txt"
    if preferred.exists():
        return preferred
    txt_files = sorted(run_dir.glob("*.txt"))
    return txt_files[0] if txt_files else None


def _read_args_row(run_dir: Path) -> Optional[pd.Series]:
    args_path = run_dir / "args.csv"
    if not args_path.exists():
        return None
    args_df = pd.read_csv(args_path)
    if args_df.empty:
        return None
    return args_df.iloc[0]


def _to_float(value: str) -> float:
    low = value.strip().lower()
    if low == "nan":
        return float("nan")
    if low == "inf":
        return float("inf")
    if low == "-inf":
        return float("-inf")
    return float(value)


def _parse_metrics_from_log(log_file: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    final_metrics: dict[str, float] = {}
    for line in log_file.read_text(errors="ignore").splitlines():
        match = EPOCH_LINE_RE.search(line)
        if match:
            row = {}
            for k, v in match.groupdict().items():
                if k == "epoch":
                    continue
                row[k] = _to_float(v) if v is not None else float("nan")
            row["epoch"] = int(match.group("epoch"))
            rows.append(row)

        final_match = FINAL_METRICS_RE.search(line)
        if final_match:
            parsed = {k: _to_float(v) for k, v in final_match.groupdict().items() if k != "method"}
            parsed["method"] = final_match.group("method")
            final_metrics = parsed
    if not rows:
        return pd.DataFrame(), final_metrics
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    # Helpful timing transforms for presentation.
    df["time_delta_sec"] = df["time_val"].diff().fillna(df["time_val"]).clip(lower=0.0)
    return df, final_metrics


def load_runs(
    raw_data_dir: Path,
    precision_filter: Optional[str],
    seed_filter: Optional[int],
) -> list[RunData]:
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {raw_data_dir}")

    runs: list[RunData] = []
    for run_dir in sorted(raw_data_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        args_row = _read_args_row(run_dir)
        if args_row is None:
            continue

        precision = str(args_row.get("precision_str", args_row.get("precision", "")))
        seed = args_row.get("seed", None)
        if precision_filter and precision != precision_filter:
            continue
        if seed_filter is not None and (pd.isna(seed) or int(seed) != seed_filter):
            continue

        log_file = _find_log_file(run_dir)
        if log_file is None:
            continue

        metrics, final_metrics = _parse_metrics_from_log(log_file)
        if metrics.empty:
            continue

        runs.append(RunData(run_dir=run_dir, args_row=args_row, metrics=metrics, final_metrics=final_metrics))
    return runs


def to_epoch_dataframe(runs: list[RunData]) -> pd.DataFrame:
    frames = []
    for run in runs:
        df = run.metrics.copy()
        df["run_name"] = run.run_dir.name
        df["train_method"] = get_method_label(run.args_row)
        for key in ["seed", "width", "h", "T", "beta", "precision_str", "odeint_type", "method"]:
            if key in run.args_row:
                df[key] = run.args_row[key]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def get_method_label(args_row: pd.Series) -> str:
    if _as_bool(args_row.get("adjoint", False)):
        return "Adjoint"
    return "Backprop"


def to_summary_dataframe(epoch_df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for run_name, run_df in epoch_df.groupby("run_name"):
        run_df = run_df.sort_values("epoch").reset_index(drop=True)
        best_idx = run_df["test_acc"].idxmax()
        best_row = run_df.loc[best_idx]
        final_row = run_df.iloc[-1]

        reached = run_df[run_df["test_acc"] >= 0.90]
        epoch_to_90 = int(reached.iloc[0]["epoch"]) if not reached.empty else np.nan

        summaries.append(
            {
                "run_name": run_name,
                "seed": final_row.get("seed", np.nan),
                "width": final_row.get("width", np.nan),
                "h": final_row.get("h", np.nan),
                "T": final_row.get("T", np.nan),
                "beta": final_row.get("beta", np.nan),
                "precision": final_row.get("precision_str", np.nan),
                "odeint": final_row.get("odeint_type", np.nan),
                "best_test_acc": float(best_row["test_acc"]),
                "epoch_of_best_test_acc": int(best_row["epoch"]),
                "final_test_acc": float(final_row["test_acc"]),
                "final_train_acc": float(final_row["train_acc"]),
                "final_time_reported_sec": float(final_row["time_val"]),
                "avg_reported_time_sec_ema": float(final_row["time_avg"]),
                "avg_epoch_delta_time_sec": float(run_df["time_delta_sec"].mean()),
                "mean_nfe_f": float(run_df["nfe_f"].mean()),
                "mean_nfe_b": float(run_df["nfe_b"].mean()),
                "epoch_to_90_test_acc": epoch_to_90,
            }
        )
    return pd.DataFrame(summaries).sort_values("run_name").reset_index(drop=True)


def _run_label(run_name: str, max_len: int = 28) -> str:
    return run_name if len(run_name) <= max_len else run_name[: max_len - 3] + "..."


def plot_accuracy_curves(epoch_df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, run_df in epoch_df.groupby("run_name"):
        run_df = run_df.sort_values("epoch")
        label = _run_label(run_name)
        ax.plot(run_df["epoch"], run_df["test_acc"], linewidth=2.2, label=f"test")
        ax.plot(run_df["epoch"], run_df["train_acc"], linestyle="--", alpha=0.8, linewidth=1.6, label=f"train")
    ax.set_title("MNIST Accuracy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "mnist_accuracy_curves.png", dpi=220)
    plt.close(fig)


def plot_runtime_nfe(epoch_df: pd.DataFrame, outdir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for run_name, run_df in epoch_df.groupby("run_name"):
        run_df = run_df.sort_values("epoch")
        label = _run_label(run_name, max_len=22)
        ax1.plot(run_df["epoch"], run_df["time_delta_sec"], marker="o", markersize=3, linewidth=1.8, label=label)
        ax2.plot(run_df["epoch"], run_df["nfe_f"], marker="s", markersize=3, linewidth=1.8, label=f"{label} NFE-F")
        ax2.plot(run_df["epoch"], run_df["nfe_b"], marker="^", markersize=3, linewidth=1.8, linestyle="--", label=f"{label} NFE-B")

    ax1.set_title("Per-Epoch Reported Time Delta")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Seconds")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("NFE vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("NFE")
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    if len(handles) <= 12:
        ax2.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "mnist_runtime_nfe.png", dpi=220)
    plt.close(fig)


def plot_loss_curves_by_method(epoch_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if "train_loss" not in epoch_df.columns:
        return None

    loss_df = epoch_df.dropna(subset=["train_loss"]).copy()
    if loss_df.empty:
        return None

    agg = (
        loss_df.groupby(["train_method", "epoch"], as_index=False)["train_loss"]
        .mean()
        .sort_values(["train_method", "epoch"])
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_name, run_df in agg.groupby("train_method"):
        ax.plot(
            run_df["epoch"],
            run_df["train_loss"],
            marker="o",
            markersize=4,
            linewidth=2.2,
            label=method_name,
        )
    ax.set_title("MNIST Training Loss vs Epoch (Adjoint vs Backprop)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = outdir / "mnist_loss_by_method.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_summary_bars(summary_df: pd.DataFrame, outdir: Path) -> None:
    names = [_run_label(name, max_len=20) for name in summary_df["run_name"]]
    x = np.arange(len(summary_df))
    width = 0.37

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(summary_df) + 4), 5.5))
    ax.bar(x - width / 2, summary_df["final_test_acc"], width=width, label="Final Test Acc")
    ax.bar(x + width / 2, summary_df["best_test_acc"], width=width, label="Best Test Acc")

    ax.set_title("MNIST Test Accuracy Summary (Float32)")
    ax.set_xlabel("Run")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, min(1.0, max(0.97, float(summary_df["best_test_acc"].max()) + 0.02)))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "mnist_test_acc_summary.png", dpi=220)
    plt.close(fig)


def build_method_comparison_tables(
    runs: list[RunData],
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_by_run = {row["run_name"]: row for _, row in summary_df.iterrows()}
    rows = []
    for run in runs:
        run_name = run.run_dir.name
        summary = summary_by_run[run_name]
        method_label = get_method_label(run.args_row)
        final = run.final_metrics

        final_test_acc = float(summary["final_test_acc"])
        fallback_test_error_pct = 100.0 * (1.0 - final_test_acc)
        fallback_train_time_sec = float(summary["final_time_reported_sec"])

        rows.append(
            {
                "run_name": run_name,
                "method": method_label,
                "test_error_pct": float(final.get("test_error_pct", fallback_test_error_pct)),
                "train_gpu_mem_mb": float(final.get("train_gpu_mem_mb", np.nan)),
                "train_time_sec": float(final.get("train_time_sec", fallback_train_time_sec)),
                "inference_gpu_mem_mb": float(final.get("inference_gpu_mem_mb", np.nan)),
                "inference_time_sec": float(final.get("inference_time_sec", np.nan)),
            }
        )

    per_run_df = pd.DataFrame(rows).sort_values(["method", "run_name"]).reset_index(drop=True)
    if per_run_df.empty:
        return per_run_df, per_run_df

    agg_df = (
        per_run_df.groupby("method", as_index=False)
        .agg(
            n_runs=("run_name", "count"),
            test_error_pct=("test_error_pct", "mean"),
            train_gpu_mem_mb=("train_gpu_mem_mb", "mean"),
            train_time_sec=("train_time_sec", "mean"),
            inference_gpu_mem_mb=("inference_gpu_mem_mb", "mean"),
            inference_time_sec=("inference_time_sec", "mean"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    return per_run_df, agg_df


def plot_method_comparison_table(method_df: pd.DataFrame, outdir: Path) -> Optional[Path]:
    if method_df.empty:
        return None

    display_df = method_df.copy()
    for col in [
        "test_error_pct",
        "train_gpu_mem_mb",
        "train_time_sec",
        "inference_gpu_mem_mb",
        "inference_time_sec",
    ]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

    display_df = display_df.rename(
        columns={
            "method": "Method",
            "n_runs": "Runs",
            "test_error_pct": "Test Error (%)",
            "train_gpu_mem_mb": "Training GPU Mem (MB)",
            "train_time_sec": "Training Time (s)",
            "inference_gpu_mem_mb": "Inference GPU Mem (MB)",
            "inference_time_sec": "Inference Time (s)",
        }
    )

    fig_h = max(2.8, 1.1 + 0.5 * len(display_df))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)
    ax.set_title("MNIST Method Comparison Metrics", pad=12)
    fig.tight_layout()
    out_path = outdir / "mnist_method_comparison_table.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _as_namespace(args_obj):
    if isinstance(args_obj, argparse.Namespace):
        return args_obj
    if isinstance(args_obj, dict):
        return argparse.Namespace(**args_obj)
    return args_obj


def _ensure_eval_args(ckpt_args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "width": 64,
        "odeint": "rampde",
        "precision": "float32",
        "h": 0.1,
        "T": 1.0,
        "gpu": 0,
        "no_dynamic_scaler": True,
        "no_grad_scaler": True,
    }
    for key, value in defaults.items():
        if not hasattr(ckpt_args, key):
            setattr(ckpt_args, key, value)
    return ckpt_args


def _resolve_checkpoint_path(
    runs: list[RunData],
    summary_df: pd.DataFrame,
    checkpoint_arg: Optional[Path],
    confusion_run_name: Optional[str],
) -> Path:
    if checkpoint_arg is not None:
        ckpt = Path(checkpoint_arg)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt}")
        return ckpt

    run_map = {r.run_dir.name: r for r in runs}
    if confusion_run_name:
        if confusion_run_name not in run_map:
            available = ", ".join(sorted(run_map.keys()))
            raise ValueError(f"Run '{confusion_run_name}' not found. Available: {available}")
        target_run = run_map[confusion_run_name]
    else:
        best_run_name = summary_df.sort_values("best_test_acc", ascending=False).iloc[0]["run_name"]
        target_run = run_map[best_run_name]

    candidates = [
        target_run.run_dir / "ckpt.pth",
        target_run.run_dir / "model.pth",
        Path("model.pth"),
        Path(__file__).resolve().parent / "model.pth",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find a checkpoint. Tried: " + ", ".join(str(p) for p in candidates)
    )


def _compute_confusion_matrix(model, dataloader, device: torch.device, num_classes: int = 10) -> np.ndarray:
    import torch

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            for true_label, pred_label in zip(y.view(-1), pred.view(-1)):
                cm[int(true_label.item()), int(pred_label.item())] += 1
    return cm


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path, normalize: bool, title: str) -> None:
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            display = np.divide(cm, row_sums, where=row_sums != 0)
            display = np.nan_to_num(display)
    else:
        display = cm

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(display, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(10),
        yticks=np.arange(10),
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    thresh = display.max() * 0.6 if display.size else 0.0
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def generate_confusion_matrix_outputs(
    runs: list[RunData],
    summary_df: pd.DataFrame,
    outdir: Path,
    checkpoint_arg: Optional[Path],
    confusion_run_name: Optional[str],
) -> tuple[Path, Path, float]:
    import torch

    ckpt_path = _resolve_checkpoint_path(runs, summary_df, checkpoint_arg, confusion_run_name)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = _ensure_eval_args(_as_namespace(checkpoint.get("args", argparse.Namespace())))

    # Import MNIST model helpers from the local training script.
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    import ode_mnist as mnist_mod  # local module import

    base_dir = os.path.abspath(os.path.join(this_dir, "../.."))
    odeint_func, DynamicScaler = mnist_mod.setup_environment(ckpt_args.odeint, base_dir)
    precision = mnist_mod.get_precision_dtype(ckpt_args.precision)
    device = torch.device(f"cuda:{ckpt_args.gpu}" if torch.cuda.is_available() else "cpu")

    model = mnist_mod.MPNFODE_MNIST(
        ckpt_args.width,
        ckpt_args,
        precision,
        odeint_func,
        DynamicScaler,
        dynamic_scaler_enabled=not ckpt_args.no_dynamic_scaler,
        grad_scaler_enabled=not ckpt_args.no_grad_scaler,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    _, test_loader, _ = mnist_mod.get_mnist_loaders(
        batch_size=getattr(ckpt_args, "batch_size", 128),
        test_batch_size=getattr(ckpt_args, "test_batch_size", 256),
        seed=getattr(ckpt_args, "seed", None),
    )

    cm = _compute_confusion_matrix(model, test_loader, device=device, num_classes=10)
    test_acc = float(np.trace(cm) / max(cm.sum(), 1))

    counts_path = outdir / "mnist_confusion_matrix_counts.png"
    norm_path = outdir / "mnist_confusion_matrix_normalized.png"
    _plot_confusion_matrix(cm, counts_path, normalize=False, title="MNIST Confusion Matrix (Counts)")
    _plot_confusion_matrix(cm, norm_path, normalize=True, title="MNIST Confusion Matrix (Row-Normalized)")
    return counts_path, norm_path, test_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MNIST metrics for presentation.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("./raw_data/ode_mnist"),
        help="Directory containing MNIST experiment folders.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        help="Precision filter (default: float32). Use '' to disable filtering.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed filter.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/fig_mnist_metrics"),
        help="Output directory for plots and CSV summaries.",
    )
    parser.add_argument(
        "--confusion-matrix",
        default=True,
        action="store_true",
        help="Also generate confusion matrix plots from a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path (.pth) for confusion matrix.",
    )
    parser.add_argument(
        "--confusion-run-name",
        type=str,
        default=None,
        help="Run directory name to use for confusion matrix (default: best test-acc run).",
    )
    parser.add_argument(
        "--backprop",
        action="store_true",
        help="Whether to include runs using the backpropagation method (if precision filter allows).",
    )
    args = parser.parse_args()

    precision_filter = args.precision if args.precision else None
    runs = load_runs(args.raw_data_dir, precision_filter=precision_filter, seed_filter=args.seed)
    if not runs:
        raise RuntimeError(
            f"No parseable runs found in {args.raw_data_dir} "
            f"(precision={precision_filter}, seed={args.seed})."
        )

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    epoch_df = to_epoch_dataframe(runs)
    summary_df = to_summary_dataframe(epoch_df)

    epoch_csv = outdir / "mnist_epoch_metrics.csv"
    summary_csv = outdir / "mnist_summary_metrics.csv"
    per_run_method_csv = outdir / "mnist_method_comparison_per_run.csv"
    method_table_csv = outdir / "mnist_method_comparison_table.csv"
    epoch_df.to_csv(epoch_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plot_accuracy_curves(epoch_df, outdir)
    plot_runtime_nfe(epoch_df, outdir)
    plot_summary_bars(summary_df, outdir)
    loss_plot = plot_loss_curves_by_method(epoch_df, outdir)
    per_run_method_df, method_table_df = build_method_comparison_tables(runs, summary_df)
    per_run_method_df.to_csv(per_run_method_csv, index=False)
    method_table_df.to_csv(method_table_csv, index=False)
    table_plot = plot_method_comparison_table(method_table_df, outdir)

    print(f"Processed runs: {len(summary_df)}")
    print(f"Epoch metrics CSV: {epoch_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Per-run method metrics CSV: {per_run_method_csv}")
    print(f"Method comparison table CSV: {method_table_csv}")
    if loss_plot is not None:
        print(f"Loss-by-method plot: {loss_plot}")
    else:
        print("Loss-by-method plot: skipped (no train loss found in logs).")
    if table_plot is not None:
        print(f"Method comparison table image: {table_plot}")
    print(f"Plots written to: {outdir}")
    print("\nTop runs by best_test_acc:")
    top = summary_df.sort_values("best_test_acc", ascending=False).head(5)
    print(top[["run_name", "best_test_acc", "final_test_acc", "epoch_of_best_test_acc", "mean_nfe_f", "mean_nfe_b"]].to_string(index=False))





if __name__ == "__main__":
    main()
