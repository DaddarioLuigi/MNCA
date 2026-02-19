"""
Dashboard risultati esperimenti: aggregazione metriche e visualizzazioni.
"""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def discover_experiments(results_base: Path):
    """
    Prende tutte le sottocartelle dirette NB_* sotto results_base (es. tissue_simulation_extended/NB_2, NB_3, NB_5).
    Per ogni NB_* legge summary.json e biological_metrics.csv se presenti.
    Ritorna: list di dict con keys: path, nb_size, summary, metrics_path, has_summary, has_metrics
    """
    results_base = Path(results_base)
    if not results_base.exists():
        return []
    experiments = []
    for exp_dir in sorted(results_base.iterdir(), key=lambda p: (_nb_from_dir_name(p), str(p))):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("NB_"):
            continue
        nb_size = _nb_from_dir_name(exp_dir)
        summary_path = exp_dir / "summary.json"
        metrics_path = exp_dir / "biological_metrics.csv"
        summary = {}
        if summary_path.exists():
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except Exception:
                pass
        experiments.append({
            "path": exp_dir,
            "nb_size": nb_size,
            "summary": summary,
            "metrics_path": metrics_path,
            "has_summary": summary_path.exists(),
            "has_metrics": metrics_path.exists(),
            "checkpoints": list_final_checkpoints(exp_dir),
        })
    return experiments


def _nb_from_dir_name(exp_dir: Path) -> int:
    name = exp_dir.name
    if name.startswith("NB_"):
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            pass
    return -1


def list_final_checkpoints(exp_dir: Path) -> list:
    """Elenco dei file .pt nella cartella che sono risultati completi (no _phase)."""
    exp_dir = Path(exp_dir)
    out = []
    for f in exp_dir.glob("*.pt"):
        if "_phase" not in f.name.lower():
            out.append(f.name)
    return sorted(out)


def load_all_metrics(experiments: list) -> pd.DataFrame:
    """Carica tutti i biological_metrics.csv in un unico DataFrame."""
    rows = []
    for exp in experiments:
        if not exp.get("has_metrics"):
            continue
        try:
            df = pd.read_csv(exp["metrics_path"])
            if "exp_dir" in df.columns:
                df = df.drop(columns=["exp_dir"])
            if "nb_size" in df.columns:
                df = df.drop(columns=["nb_size"])
            df["exp_dir"] = str(exp["path"])
            df["nb_size"] = exp["nb_size"]
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_metrics_dashboard(df: pd.DataFrame, metric_cols: list = None):
    """
    Crea grafici Plotly per le metriche principali.
    df deve contenere colonne come 'Model Type', 'Neighborhood Size', 'Step Length'
    e le colonne numeriche delle metriche.
    """
    if df.empty:
        return go.Figure(layout=dict(title="No data available"))
    default_metrics = [
        "KL Divergence", "Chi-Square", "Categorical MMD",
        "Tumor Size Diff", "Border Size Diff", "Spatial Variance Diff",
    ]
    available = [c for c in default_metrics if c in df.columns]
    if not available:
        available = [c for c in df.select_dtypes(include=["number"]).columns if "SD" not in c][:6]
    if metric_cols:
        available = [c for c in metric_cols if c in df.columns] or available
    n_metrics = len(available)
    if n_metrics == 0:
        return go.Figure(layout=dict(title="No data available"))
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=available,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    for i, col in enumerate(available):
        row, col_idx = i // n_cols + 1, i % n_cols + 1
        if "Model Type" in df.columns and "Neighborhood Size" in df.columns:
            for model_type in df["Model Type"].dropna().unique():
                sub = df[(df["Model Type"] == model_type)]
                fig.add_trace(
                    go.Scatter(
                        x=sub["Neighborhood Size"],
                        y=sub[col],
                        name=model_type,
                        mode="lines+markers",
                        legendgroup=model_type,
                    ),
                    row=row, col=col_idx,
                )
        else:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], name=col, mode="lines+markers"),
                row=row, col=col_idx,
            )
    fig.update_layout(
        height=280 * n_rows,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_metrics_single_nb(df_nb: pd.DataFrame, nb_size: int):
    """Grafico metriche per una singola cartella NB_ (Step Length vs metriche, per Model Type)."""
    if df_nb.empty or "Step Length" not in df_nb.columns:
        return None
    metric_cols = [
        c for c in ["KL Divergence", "Chi-Square", "Categorical MMD",
                    "Tumor Size Diff", "Border Size Diff", "Spatial Variance Diff"]
        if c in df_nb.columns
    ]
    if not metric_cols:
        return None
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=metric_cols[:6],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for i, col in enumerate(metric_cols[:6]):
        row, col_idx = i // 3 + 1, i % 3 + 1
        if "Model Type" in df_nb.columns:
            for model_type in df_nb["Model Type"].dropna().unique():
                sub = df_nb[df_nb["Model Type"] == model_type]
                fig.add_trace(
                    go.Scatter(
                        x=sub["Step Length"], y=sub[col],
                        name=model_type, mode="lines+markers", legendgroup=model_type,
                    ),
                    row=row, col=col_idx,
                )
        else:
            fig.add_trace(
                go.Scatter(x=df_nb["Step Length"], y=df_nb[col], name=col, mode="lines+markers"),
                row=row, col=col_idx,
            )
    fig.update_layout(
        title=f"NB_{nb_size} — Metrics by Step Length",
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_metrics_by_step_length(df: pd.DataFrame):
    """Un grafico che mostra l'andamento delle metriche per Step Length (facet per metrica)."""
    if df.empty or "Step Length" not in df.columns:
        return go.Figure(layout=dict(title="No Step Length data"))
    num_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64") and "SD" not in c][:6]
    if not num_cols:
        return go.Figure(layout=dict(title="No numeric data"))
    fig = px.line(
        df,
        x="Step Length",
        y=num_cols[0],
        color="Model Type" if "Model Type" in df.columns else None,
        facet_row=None,
        markers=True,
    )
    if len(num_cols) > 1:
        fig = make_subplots(rows=len(num_cols), cols=1, subplot_titles=num_cols, shared_xaxes=True)
        for i, col in enumerate(num_cols):
            for model_type in (df["Model Type"].unique() if "Model Type" in df.columns else [None]):
                sub = df[df["Model Type"] == model_type] if model_type else df
                fig.add_trace(
                    go.Scatter(x=sub["Step Length"], y=sub[col], name=model_type or col, mode="lines+markers"),
                    row=i + 1, col=1,
                )
        fig.update_layout(height=200 * len(num_cols))
    return fig
