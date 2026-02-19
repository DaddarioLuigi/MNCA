#!/usr/bin/env python3
"""
Analyze results from curriculum learning experiments (tissue_simulation_extended with --curriculum).

Expected layout:
  results_dir/
    tissue_simulation_extended/
      NB_2/, NB_3/, NB_5/, ...
        biological_metrics.csv
        mixture_nca_curriculum_loss_curve.npy
        stochastic_mix_nca_curriculum_loss_curve.npy
      all_neighborhood_sizes_metrics.csv  (optional)

Usage:
  cd experiments
  python analyze_curriculum.py --results_dir results_extended
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLS = [
    "KL Divergence",
    "Chi-Square",
    "Categorical MMD",
    "Tumor Size Diff",
    "Border Size Diff",
    "Spatial Variance Diff",
]


def _parse_sd(val):
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    s = str(val).strip().replace("±", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_metrics(base_dir: Path, experiment_subdir: str = "tissue_simulation_extended") -> pd.DataFrame:
    """Load all biological_metrics.csv from NB_* and optionally all_neighborhood_sizes_metrics.csv."""
    exp_dir = base_dir / experiment_subdir
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    agg_path = exp_dir / "all_neighborhood_sizes_metrics.csv"
    if agg_path.exists():
        df = pd.read_csv(agg_path)
    else:
        frames = []
        for nb_dir in sorted(exp_dir.glob("NB_*")):
            if not nb_dir.is_dir():
                continue
            csv_path = nb_dir / "biological_metrics.csv"
            if not csv_path.exists():
                continue
            nb = int(nb_dir.name.split("_")[1])
            part = pd.read_csv(csv_path)
            part["Neighborhood Size"] = nb
            frames.append(part)
        if not frames:
            raise FileNotFoundError(f"No biological_metrics.csv found under {exp_dir}")
        df = pd.concat(frames, ignore_index=True)

    if "Step Length" in df.columns:
        df["Step Length"] = df["Step Length"].astype(int)
    return df


def prepare_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed SD for error bars if needed."""
    out = df.copy()
    for col in METRIC_COLS:
        if col not in out.columns:
            continue
        sd_col = f"{col} SD"
        if sd_col in out.columns:
            out[f"{sd_col}_num"] = out[sd_col].map(_parse_sd)
    return out


def plot_metrics_by_nb_and_step(df: pd.DataFrame, out_dir: Path, metrics=None, models=None) -> None:
    """Line plots: metric vs Neighborhood Size, one panel per step length and model."""
    df = prepare_plot_df(df)
    metrics = metrics or METRIC_COLS

    model_col = "Model Type"
    if model_col not in df.columns:
        return
    if models is None:
        models = df[model_col].unique().tolist()

    for metric in metrics:
        if metric not in df.columns:
            continue

        sd_col = f"{metric} SD"
        sd_num_col = f"{sd_col}_num"
        has_sd = sd_col in df.columns and sd_num_col in df.columns

        step_lengths = sorted(df["Step Length"].unique())
        n_steps = len(step_lengths)
        n_models = len(models)
        fig, axes = plt.subplots(
            n_models,
            n_steps,
            figsize=(4 * n_steps, 4 * n_models),
            squeeze=False,
        )

        for mi, model in enumerate(models):
            sub = df[df[model_col] == model]
            if sub.empty:
                continue
            for si, sl in enumerate(step_lengths):
                ax = axes[mi, si]
                s = sub[sub["Step Length"] == sl].sort_values("Neighborhood Size")
                nbs = s["Neighborhood Size"].values
                vals = s[metric].values
                ax.plot(nbs, vals, "o-", label=model)
                if has_sd:
                    sd_vals = s[sd_num_col].values
                    ax.fill_between(nbs, vals - sd_vals, vals + sd_vals, alpha=0.2)
                ax.set_xlabel("Neighborhood Size")
                ax.set_ylabel(metric)
                ax.set_title(f"Step length = {sl}")
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"Curriculum learning: {metric} vs NB", fontsize=12)
        plt.tight_layout()
        out_path = out_dir / f"curriculum_metric_vs_nb_{metric.replace(' ', '_').lower()}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_metrics_dashboard(df: pd.DataFrame, out_dir: Path) -> None:
    """One figure with 6 metric panels (mean across step lengths per NB and model)."""
    df = prepare_plot_df(df)
    model_col = "Model Type"
    if model_col not in df.columns:
        return
    models = df[model_col].unique().tolist()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(METRIC_COLS):
        if idx >= len(axes) or metric not in df.columns:
            continue
        ax = axes[idx]
        for model in models:
            sub = df[df[model_col] == model]
            means = sub.groupby("Neighborhood Size")[metric].mean()
            ax.plot(means.index, means.values, "o-", label=model)
        ax.set_xlabel("Neighborhood Size")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Curriculum learning: metrics vs NB (mean over step lengths)", fontsize=12)
    plt.tight_layout()
    out_path = out_dir / "curriculum_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def load_loss_curves(exp_dir: Path, nb_sizes: list[int], suffix: str = "curriculum") -> dict:
    """Load loss curves from *_{suffix}_loss_curve.npy per NB and model."""
    curves: dict[tuple[int, str], np.ndarray] = {}
    for nb in nb_sizes:
        nb_dir = exp_dir / f"NB_{nb}"
        if not nb_dir.exists():
            continue
        for name, pattern in [
            ("Mixture NCA", f"mixture_nca_{suffix}_loss_curve.npy"),
            ("Stochastic Mixture NCA", f"stochastic_mix_nca_{suffix}_loss_curve.npy"),
        ]:
            path = nb_dir / pattern
            if path.exists():
                curves[(nb, name)] = np.load(path)
    return curves


def plot_loss_curves(exp_dir: Path, out_dir: Path, nb_sizes: list[int], suffix: str = "curriculum") -> None:
    """Plot curriculum loss curves per NB and model."""
    curves = load_loss_curves(exp_dir, nb_sizes, suffix=suffix)
    if not curves:
        curves = load_loss_curves(exp_dir, nb_sizes, suffix="")
    if not curves:
        print("No loss curve files found; skipping loss plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for (nb, model_name), loss in curves.items():
        if "Mixture" in model_name and "Stochastic" not in model_name:
            ax = axes[0]
        else:
            ax = axes[1]
        ax.plot(loss, alpha=0.8, label=f"NB={nb}")

    axes[0].set_title("Mixture NCA – curriculum loss")
    axes[0].set_xlabel("Step (concatenated phases)")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Stochastic Mixture NCA – curriculum loss")
    axes[1].set_xlabel("Step (concatenated phases)")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "curriculum_loss_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Same data with log-scale y-axis so initial drop and small differences are visible
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for (nb, model_name), loss in curves.items():
        loss_safe = np.maximum(loss, 1e-8)
        if "Mixture" in model_name and "Stochastic" not in model_name:
            ax = axes2[0]
        else:
            ax = axes2[1]
        ax.semilogy(loss_safe, alpha=0.8, label=f"NB={nb}")
    axes2[0].set_title("Mixture NCA – curriculum loss (log scale)")
    axes2[0].set_xlabel("Step (concatenated phases)")
    axes2[0].set_ylabel("Loss (log scale)")
    axes2[0].legend(loc="best", fontsize=8)
    axes2[0].grid(True, alpha=0.3)
    axes2[1].set_title("Stochastic Mixture NCA – curriculum loss (log scale)")
    axes2[1].set_xlabel("Step (concatenated phases)")
    axes2[1].set_ylabel("Loss (log scale)")
    axes2[1].legend(loc="best", fontsize=8)
    axes2[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out_path_log = out_dir / "curriculum_loss_curves_log.png"
    fig2.savefig(out_path_log, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {out_path_log}")


def summary_table(df: pd.DataFrame, out_dir: Path) -> None:
    """Write a short summary: best NB per (model, step length, metric)."""
    df = prepare_plot_df(df)
    model_col = "Model Type"
    if model_col not in df.columns or "Neighborhood Size" not in df.columns:
        return

    rows: list[dict[str, object]] = []
    for model in df[model_col].unique():
        for step in df["Step Length"].unique():
            sub = df[(df[model_col] == model) & (df["Step Length"] == step)]
            if sub.empty:
                continue
            for metric in METRIC_COLS:
                if metric not in sub.columns:
                    continue
                best_idx = sub[metric].idxmin()
                best_row = sub.loc[best_idx]
                rows.append(
                    {
                        "Model": model,
                        "Step Length": int(step),
                        "Metric": metric,
                        "Best NB": int(best_row["Neighborhood Size"]),
                        "Value": float(best_row[metric]),
                    }
                )

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "curriculum_summary_best_nb.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze curriculum learning results (metrics + loss curves).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_extended",
        help="Base results directory (default: results_extended)",
    )
    parser.add_argument(
        "--experiment_subdir",
        type=str,
        default="tissue_simulation_extended",
        help="Subfolder under results_dir containing NB_* (default: tissue_simulation_extended)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots. Default: results_dir/experiment_subdir/curriculum_analysis",
    )
    parser.add_argument(
        "--no_loss",
        action="store_true",
        help="Skip loading/plotting loss curves",
    )
    parser.add_argument(
        "--loss_suffix",
        type=str,
        default="curriculum",
        help="Suffix for loss files, e.g. mixture_nca_{suffix}_loss_curve.npy (default: curriculum)",
    )
    args = parser.parse_args()

    base = Path(args.results_dir)
    exp_dir = base / args.experiment_subdir
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "curriculum_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results base: {base}")
    print(f"Experiment dir: {exp_dir}")
    print(f"Output dir: {out_dir}")

    df = load_metrics(base, args.experiment_subdir)
    nb_sizes = sorted(df["Neighborhood Size"].unique().tolist())
    print(f"Loaded metrics for NB: {nb_sizes}, step lengths: {sorted(df['Step Length'].unique().tolist())}")

    plot_metrics_dashboard(df, out_dir)
    plot_metrics_by_nb_and_step(df, out_dir)
    summary_table(df, out_dir)

    if not args.no_loss:
        plot_loss_curves(exp_dir, out_dir, nb_sizes, suffix=args.loss_suffix)

    print("Done.")


if __name__ == "__main__":
    main()

