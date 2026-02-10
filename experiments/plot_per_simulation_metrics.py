"""
Plotly analysis for per-simulation metrics across neighborhood sizes.

Input:
  Folder containing CSV files named like:
    raw_metrics_per_simulation_nb1.csv ... raw_metrics_per_simulation_nb7.csv

Outputs:
  - HTML plots under <input_folder>/plots/
  - A small index.html linking to all plots
  - summary CSVs (evaluation-level + aggregated)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRICS: list[str] = [
    "KL Divergence",
    "Chi-Square",
    "Categorical MMD",
    "Tumor Size Diff",
    "Border Size Diff",
    "Spatial Variance Diff",
]


@dataclass(frozen=True)
class PlotPaths:
    out_dir: Path
    index_html: Path


def _infer_nb_from_filename(p: Path) -> int | None:
    m = re.search(r"nb(\d+)", p.stem.lower())
    return int(m.group(1)) if m else None


def load_per_simulation_folder(folder: Path) -> pd.DataFrame:
    csvs = sorted(folder.glob("raw_metrics_per_simulation_nb*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No per-simulation CSVs found in: {folder}")

    frames: list[pd.DataFrame] = []
    for p in csvs:
        df = pd.read_csv(p)
        nb = _infer_nb_from_filename(p)
        if "Neighborhood Size" not in df.columns and nb is not None:
            df["Neighborhood Size"] = nb
        if nb is not None and "Neighborhood Size" in df.columns:
            # Ensure consistency with filename
            df["Neighborhood Size"] = df["Neighborhood Size"].astype(int)
            df.loc[:, "Neighborhood Size"] = nb
        df["__source_file__"] = p.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Normalize dtypes
    for col in ["Neighborhood Size", "Step Length", "Evaluation", "Simulation"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    # Ensure metrics are numeric
    for m in METRICS:
        if m in out.columns:
            out[m] = pd.to_numeric(out[m], errors="coerce")

    # Drop rows missing core columns
    required = {"Model Type", "Neighborhood Size", "Step Length", "Evaluation", "Simulation"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing required columns in per-simulation CSVs: {sorted(missing)}")

    out = out.dropna(subset=["Model Type", "Neighborhood Size", "Step Length", "Evaluation", "Simulation"])
    out["Neighborhood Size"] = out["Neighborhood Size"].astype(int)
    out["Step Length"] = out["Step Length"].astype(int)
    out["Evaluation"] = out["Evaluation"].astype(int)
    out["Simulation"] = out["Simulation"].astype(int)

    # Keep only known metrics + core columns
    keep_cols = ["Model Type", "Neighborhood Size", "Step Length", "Evaluation", "Simulation"] + [
        c for c in METRICS if c in out.columns
    ]
    return out[keep_cols]


def compute_eval_level_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse per-simulation rows to evaluation-level means.
    This is the right unit if you want uncertainty across 'Evaluation' seeds.
    """
    group_cols = ["Model Type", "Neighborhood Size", "Step Length", "Evaluation"]
    agg = {m: "mean" for m in METRICS if m in df.columns}
    eval_df = df.groupby(group_cols, as_index=False).agg(agg)
    return eval_df


def compute_across_eval_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, nb, step), compute mean/std across evaluations.
    """
    group_cols = ["Model Type", "Neighborhood Size", "Step Length"]
    rows = []
    for (model, nb, step), g in eval_df.groupby(group_cols):
        row = {"Model Type": model, "Neighborhood Size": nb, "Step Length": step, "n_evaluations": int(g["Evaluation"].nunique())}
        for m in METRICS:
            if m not in g.columns:
                continue
            row[f"{m} mean_across_eval"] = float(g[m].mean())
            row[f"{m} std_across_eval"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def compute_distribution_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostics that mirror what you 'see' in violin plots.

    Returns per (model, step, nb, metric):
      - median, q1, q3, p5, p95
      - iqr
      - tukey_outlier_rate: fraction outside [q1-1.5*IQR, q3+1.5*IQR]
    """
    rows = []
    for (model, step, nb), g in df.groupby(["Model Type", "Step Length", "Neighborhood Size"]):
        for metric in [m for m in METRICS if m in df.columns]:
            x = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if x.size == 0:
                continue
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            p5, p95 = np.quantile(x, [0.05, 0.95])
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            tuk = float(((x < lo) | (x > hi)).mean())
            rows.append(
                {
                    "Model Type": model,
                    "Step Length": int(step),
                    "Neighborhood Size": int(nb),
                    "Metric": metric,
                    "n": int(x.size),
                    "median": float(med),
                    "q1": float(q1),
                    "q3": float(q3),
                    "p5": float(p5),
                    "p95": float(p95),
                    "iqr": float(iqr),
                    "tukey_outlier_rate": tuk,
                }
            )
    return pd.DataFrame(rows)


def compute_bad_rate_vs_best(df: pd.DataFrame, best_quantile: float = 0.95) -> pd.DataFrame:
    """
    Bad-rate defined as: P(metric > threshold), where threshold is taken from the 'best NB'
    distribution for that (model, step, metric).

    Best NB is chosen as the one with the smallest median for that (model, step, metric).
    Threshold = Q_best_quantile of the best-NB per-simulation distribution.

    This makes 'failure probability' comparable across NB: how often does NB produce a result
    worse than the tail of the best-performing NB?
    """
    if not (0.5 < best_quantile < 1.0):
        raise ValueError("best_quantile should be in (0.5, 1.0)")

    rows = []
    for (model, step), g_ms in df.groupby(["Model Type", "Step Length"]):
        for metric in [m for m in METRICS if m in df.columns]:
            # Find best NB by median
            medians = (
                g_ms.groupby("Neighborhood Size")[metric]
                .median(numeric_only=False)
                .sort_values(ascending=True)
            )
            if medians.empty:
                continue
            best_nb = int(medians.index[0])
            x_best = pd.to_numeric(
                g_ms[g_ms["Neighborhood Size"] == best_nb][metric], errors="coerce"
            ).dropna()
            if x_best.empty:
                continue
            thr = float(np.quantile(x_best.to_numpy(dtype=float), best_quantile))

            for nb, g_nb in g_ms.groupby("Neighborhood Size"):
                x = pd.to_numeric(g_nb[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if x.size == 0:
                    continue
                bad_rate = float((x > thr).mean())
                rows.append(
                    {
                        "Model Type": model,
                        "Step Length": int(step),
                        "Metric": metric,
                        "Best NB (by median)": best_nb,
                        "Best threshold quantile": best_quantile,
                        "Threshold (from best NB)": thr,
                        "Neighborhood Size": int(nb),
                        "bad_rate": bad_rate,
                        "n": int(x.size),
                    }
                )
    return pd.DataFrame(rows)


def _ensure_out_dir(folder: Path) -> PlotPaths:
    out_dir = folder / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return PlotPaths(out_dir=out_dir, index_html=out_dir / "index.html")


def _write_index(paths: PlotPaths, html_files: Iterable[Path]) -> None:
    links = "\n".join(
        f'<li><a href="{p.name}">{p.stem}</a></li>' for p in sorted(html_files, key=lambda x: x.name)
    )
    content = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Per-simulation metrics (Plotly)</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; padding: 24px; }}
      h1 {{ margin: 0 0 12px 0; }}
      ul {{ line-height: 1.8; }}
      .meta {{ color: #555; margin-bottom: 16px; }}
      code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 4px; }}
    </style>
  </head>
  <body>
    <h1>Per-simulation metrics (Plotly)</h1>
    <div class="meta">
      Generated from CSVs in <code>per_simulation_metrics/</code>.
      Each plot is a self-contained HTML file.
    </div>
    <ul>
      {links}
    </ul>
  </body>
</html>
"""
    paths.index_html.write_text(content, encoding="utf-8")


def plot_violin_per_simulation(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.violin(
        df,
        x="Neighborhood Size",
        y=metric,
        color="Model Type",
        facet_col="Step Length",
        box=True,
        points="outliers",
        category_orders={"Neighborhood Size": sorted(df["Neighborhood Size"].unique())},
        title=f"{metric} — per-simulation distributions (faceted by Step Length)",
        template="plotly_white",
    )
    fig.update_layout(width=1400, height=600, legend_title_text="Model Type")
    fig.update_xaxes(title_text="Neighborhood Size")
    return fig


def plot_eval_means_scatter(eval_df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Each dot is an evaluation-level mean (aggregated across simulations).
    We also overlay a line with error bars: mean ± std across evaluations.
    """
    # Dots: per evaluation
    fig = px.strip(
        eval_df,
        x="Neighborhood Size",
        y=metric,
        color="Model Type",
        facet_col="Step Length",
        hover_data=["Evaluation"],
        category_orders={"Neighborhood Size": sorted(eval_df["Neighborhood Size"].unique())},
        title=f"{metric} — evaluation-level means (dot = one Evaluation seed)",
        template="plotly_white",
    )

    # Overlay mean±std across evals
    grouped = eval_df.groupby(["Model Type", "Step Length", "Neighborhood Size"], as_index=False).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        n=("Evaluation", "nunique"),
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    # Add as separate traces so they sit on top of strip plot
    for (model, step), g in grouped.groupby(["Model Type", "Step Length"]):
        fig.add_trace(
            go.Scatter(
                x=g["Neighborhood Size"],
                y=g["mean"],
                mode="lines+markers",
                name=f"{model} (mean across eval)",
                legendgroup=f"{model}",
                showlegend=True,
                error_y=dict(type="data", array=g["std"], visible=True),
            ),
            row=1,
            col=list(sorted(eval_df["Step Length"].unique())).index(step) + 1,
        )

    fig.update_layout(width=1400, height=600, legend_title_text="Model Type")
    fig.update_xaxes(title_text="Neighborhood Size")
    return fig


def plot_heatmap_across_eval(summary_df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Heatmap of mean across evaluations for each (NB, Step Length), one panel per Model Type.
    """
    mean_col = f"{metric} mean_across_eval"
    if mean_col not in summary_df.columns:
        raise ValueError(f"Missing column in summary_df: {mean_col}")

    models = list(summary_df["Model Type"].unique())
    steps = sorted(summary_df["Step Length"].unique())
    nbs = sorted(summary_df["Neighborhood Size"].unique())

    fig = make_subplots(
        rows=1,
        cols=len(models),
        subplot_titles=[f"{m}" for m in models],
        horizontal_spacing=0.10,
    )

    for col_i, model in enumerate(models, start=1):
        d = summary_df[summary_df["Model Type"] == model]
        pivot = d.pivot_table(index="Step Length", columns="Neighborhood Size", values=mean_col, aggfunc="mean")
        pivot = pivot.reindex(index=steps, columns=nbs)
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[f"NB={nb}" for nb in nbs],
                y=[f"steps={s}" for s in steps],
                coloraxis="coloraxis",
                hovertemplate="NB=%{x}<br>%{y}<br>mean=%{z:.4f}<extra></extra>",
            ),
            row=1,
            col=col_i,
        )

    fig.update_layout(
        title=f"{metric} — mean across evaluations (heatmap)",
        width=1400,
        height=520,
        template="plotly_white",
        coloraxis=dict(colorscale="Viridis"),
    )
    return fig


def plot_bad_rate_heatmap(bad_df: pd.DataFrame, metric: str) -> go.Figure:
    d = bad_df[bad_df["Metric"] == metric].copy()
    models = list(d["Model Type"].unique())
    steps = sorted(d["Step Length"].unique())
    nbs = sorted(d["Neighborhood Size"].unique())

    fig = make_subplots(
        rows=1,
        cols=len(models),
        subplot_titles=[f"{m}" for m in models],
        horizontal_spacing=0.10,
    )

    for col_i, model in enumerate(models, start=1):
        dm = d[d["Model Type"] == model]
        pivot = dm.pivot_table(index="Step Length", columns="Neighborhood Size", values="bad_rate", aggfunc="mean")
        pivot = pivot.reindex(index=steps, columns=nbs)
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[f"NB={nb}" for nb in nbs],
                y=[f"steps={s}" for s in steps],
                coloraxis="coloraxis",
                hovertemplate="NB=%{x}<br>%{y}<br>bad_rate=%{z:.1%}<extra></extra>",
            ),
            row=1,
            col=col_i,
        )

    # Pull meta from any row (threshold is per model+step but we want a readable title)
    q = float(d["Best threshold quantile"].iloc[0]) if len(d) else 0.95
    fig.update_layout(
        title=f"{metric} — bad-rate vs best-NB threshold (q={q:.2f})",
        width=1400,
        height=520,
        template="plotly_white",
        coloraxis=dict(colorscale="Reds", cmin=0.0, cmax=1.0),
    )
    return fig


def plot_bad_rate_bars(bad_df: pd.DataFrame, metric: str) -> go.Figure:
    d = bad_df[bad_df["Metric"] == metric].copy()
    q = float(d["Best threshold quantile"].iloc[0]) if len(d) else 0.95
    fig = px.bar(
        d,
        x="Neighborhood Size",
        y="bad_rate",
        color="Model Type",
        facet_col="Step Length",
        barmode="group",
        category_orders={"Neighborhood Size": sorted(d["Neighborhood Size"].unique())},
        title=f"{metric} — bad-rate P(metric > Q{int(q*100)}(best NB))",
        template="plotly_white",
    )
    fig.update_layout(width=1400, height=600)
    fig.update_yaxes(tickformat=".0%")
    return fig


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot Plotly analysis from per-simulation metrics CSVs.")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Folder containing raw_metrics_per_simulation_nb*.csv files",
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    paths = _ensure_out_dir(folder)

    df = load_per_simulation_folder(folder)
    eval_df = compute_eval_level_means(df)
    summary_df = compute_across_eval_summary(eval_df)
    diag_df = compute_distribution_diagnostics(df)
    bad_df = compute_bad_rate_vs_best(df, best_quantile=0.95)

    # Save summaries for reuse
    eval_df.to_csv(paths.out_dir / "evaluation_level_means.csv", index=False)
    summary_df.to_csv(paths.out_dir / "summary_across_evaluations.csv", index=False)
    diag_df.to_csv(paths.out_dir / "distribution_diagnostics.csv", index=False)
    bad_df.to_csv(paths.out_dir / "bad_rate_vs_best_q95.csv", index=False)

    html_files: list[Path] = []

    for metric in [m for m in METRICS if m in df.columns]:
        fig1 = plot_violin_per_simulation(df, metric)
        out1 = paths.out_dir / f"{metric.replace(' ', '_').replace('-', '').lower()}__violin_per_sim.html"
        fig1.write_html(str(out1), include_plotlyjs="cdn")
        html_files.append(out1)

        fig2 = plot_eval_means_scatter(eval_df, metric)
        out2 = paths.out_dir / f"{metric.replace(' ', '_').replace('-', '').lower()}__eval_means.html"
        fig2.write_html(str(out2), include_plotlyjs="cdn")
        html_files.append(out2)

        # Heatmap (mean across eval) per model
        try:
            fig3 = plot_heatmap_across_eval(summary_df, metric)
            out3 = paths.out_dir / f"{metric.replace(' ', '_').replace('-', '').lower()}__heatmap_across_eval.html"
            fig3.write_html(str(out3), include_plotlyjs="cdn")
            html_files.append(out3)
        except Exception:
            # Keep robust: if anything fails, skip heatmap for this metric
            pass

        # Bad-rate plots (failure probability proxy)
        try:
            fig4 = plot_bad_rate_bars(bad_df, metric)
            out4 = paths.out_dir / f"{metric.replace(' ', '_').replace('-', '').lower()}__bad_rate_bars.html"
            fig4.write_html(str(out4), include_plotlyjs="cdn")
            html_files.append(out4)

            fig5 = plot_bad_rate_heatmap(bad_df, metric)
            out5 = paths.out_dir / f"{metric.replace(' ', '_').replace('-', '').lower()}__bad_rate_heatmap.html"
            fig5.write_html(str(out5), include_plotlyjs="cdn")
            html_files.append(out5)
        except Exception:
            pass

    _write_index(paths, html_files)
    print(f"✅ Wrote {len(html_files)} plots to: {paths.out_dir}")
    print(f"🔗 Index: {paths.index_html}")


if __name__ == "__main__":
    main()

