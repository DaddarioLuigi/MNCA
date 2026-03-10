"""
Analysis of MNCA training outputs on BBBC031 (notebook BBBC031_Colab.ipynb).

Reads the folder results_extended/MNCA_bbbc031_outputs (or path from command line):
- models/*.pth: model checkpoint
- models/*_loss.npy: loss history per training run (one scalar per optimization step)
- figs/bbbc031/*.png: GT vs prediction figures and video

What "steps" mean:
- 4000 steps (training): optimization iterations (gradient updates). Each step updates
  the network weights. During training, each episode uses a random number of NCA
  simulation steps (min_steps=10, max_steps=20); loss is computed on the output after
  that number of steps.
- 20 steps (in figures): cellular automaton simulation steps used at inference. The
  model is run for 20 steps from seed → final mask; "MNCA (after 20 steps)" = grid
  state after 20 applications of the NCA rule.

So: 4000 = number of weight updates; 20 = number of simulation steps to generate
the figure image.

Usage (from repo root, with venv active or on Colab):
  python experiments/analyze_bbbc031_outputs.py
  python experiments/analyze_bbbc031_outputs.py --output_dir /path/to/MNCA_bbbc031_outputs
  python experiments/analyze_bbbc031_outputs.py --inspect_checkpoint results_extended/.../bbbc031_stochastic_NB3_img0.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Plot export and readability (thesis-quality figures)
PNG_SCALE = 4
FONT_SIZE_TITLE = 18
FONT_SIZE_SUBPLOT = 14
FONT_SIZE_AXES = 13
FONT_SIZE_TICKS = 11


def load_all_losses(
    models_dir: Path,
    nb7_retrain_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load all *_loss.npy from models_dir. If nb7_retrain_dir is set, use it for NB=7 stems."""
    out = {}
    for f in sorted(models_dir.glob("*_loss.npy")):
        stem = f.stem.removesuffix("_loss")
        out[stem] = np.load(f)
    if nb7_retrain_dir is not None and nb7_retrain_dir.exists():
        for f in sorted(nb7_retrain_dir.glob("*_loss.npy")):
            stem = f.stem.removesuffix("_loss")
            parsed = parse_stem(stem)
            if parsed and parsed[0] == 7:
                out[stem] = np.load(f)
                print(f"  (NB=7 retrain) {stem}: {len(out[stem])} steps")
    return out


def load_nb7_original_losses(models_dir: Path) -> dict[str, np.ndarray]:
    """Load only NB=7 *_loss.npy from models_dir (original run, before retrain)."""
    out = {}
    for f in sorted(models_dir.glob("*_loss.npy")):
        stem = f.stem.removesuffix("_loss")
        parsed = parse_stem(stem)
        if parsed and parsed[0] == 7:
            out[stem] = np.load(f)
    return out


def parse_stem(stem: str) -> tuple[int, int] | None:
    """From 'bbbc031_stochastic_NB3_img0' extract (nb=3, img_idx=0). Returns None if format unknown."""
    try:
        nb_part = stem.split("NB")[1].split("_")[0]
        img_part = stem.split("img")[1].split("_")[0]
        return int(nb_part), int(img_part)
    except (IndexError, ValueError):
        return None


def _sorted_stems_by_nb_img(losses: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """Return (stem, arr) list sorted by (NB, img_idx) for consistent grid order."""
    items = list(losses.items())
    def key(item):
        stem, _ = item
        p = parse_stem(stem)
        return (p if p else (0, 0), stem)
    return sorted(items, key=key)


def plot_loss_curves(
    losses: dict[str, np.ndarray],
    out_path: Path,
    smooth_window: int = 50,
) -> None:
    """Plot loss curves in a 2x5 grid: row1 NB=3 (img0..img4), row2 NB=7 (img0..img4). Shared Y per row."""
    ordered = _sorted_stems_by_nb_img(losses)
    if not ordered:
        print("No loss files found, skipping plot.")
        return
    # Build NB=3 and NB=7 groups (each up to 5 images)
    by_nb: dict[int, list[tuple[str, np.ndarray]]] = {}
    for stem, arr in ordered:
        p = parse_stem(stem)
        if p is None:
            continue
        nb, _ = p
        by_nb.setdefault(nb, []).append((stem, arr))
    nb3 = sorted(by_nb.get(3, []), key=lambda x: parse_stem(x[0])[1])
    nb7 = sorted(by_nb.get(7, []), key=lambda x: parse_stem(x[0])[1])
    if not nb3 and not nb7:
        print("No parseable NB=3/NB=7 stems, skipping plot.")
        return
    n_cols = 5
    n_rows = (2 if (nb3 and nb7) else 1)
    titles = []
    if nb3:
        titles.extend([f"NB=3, img{parse_stem(s)[1]}" for s, _ in nb3])
    while len(titles) < n_cols:
        titles.append("")
    if nb7:
        titles.extend([f"NB=7, img{parse_stem(s)[1]}" for s, _ in nb7])
    while len(titles) < n_rows * n_cols:
        titles.append("")
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles[: n_rows * n_cols],
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
        shared_yaxes="rows",
    )
    row_offset = 0
    for row_idx, group in enumerate([nb3, nb7]):
        if not group:
            continue
        row = row_offset + 1
        for col_idx, (stem, arr) in enumerate(group):
            if col_idx >= n_cols:
                break
            col = col_idx + 1
            steps = np.arange(1, len(arr) + 1, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=arr,
                    name="raw",
                    line=dict(color="rgba(31,119,180,0.35)", width=2),
                    legendgroup=stem,
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            if len(arr) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(arr, kernel, mode="valid")
                steps_s = np.arange(smooth_window, len(arr) + 1, dtype=float)
                fig.add_trace(
                    go.Scatter(
                        x=steps_s,
                        y=smoothed,
                        name="smoothed",
                        line=dict(color="rgb(31,119,180)", width=3.5),
                        legendgroup=stem,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
        row_offset += 1
    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text="Training step",
            row=n_rows,
            col=c,
            title_font_size=FONT_SIZE_AXES,
            tickfont_size=FONT_SIZE_TICKS,
        )
    fig.update_yaxes(title_text="Loss", row=1, col=1, title_font_size=FONT_SIZE_AXES, tickfont_size=FONT_SIZE_TICKS)
    if n_rows > 1:
        fig.update_yaxes(title_text="Loss", row=2, col=1, title_font_size=FONT_SIZE_AXES, tickfont_size=FONT_SIZE_TICKS)
    fig.update_layout(
        title_text="BBBC031 MNCA — Loss per run (row 1: NB=3, row 2: NB=7)",
        title_font_size=FONT_SIZE_TITLE,
        font_size=FONT_SIZE_TICKS,
        showlegend=False,
        height=360 * n_rows,
        template="plotly_white",
    )
    fig.update_annotations(font_size=FONT_SIZE_SUBPLOT)
    html_path = out_path if out_path.suffix.lower() == ".html" else out_path.parent / (out_path.stem + ".html")
    fig.write_html(str(html_path))
    print("Saved:", html_path)
    png_path = html_path.with_suffix(".png")
    try:
        fig.write_image(str(png_path), scale=PNG_SCALE)
        print("Saved:", png_path)
    except Exception:
        pass
    if out_path.suffix.lower() == ".png":
        try:
            fig.write_image(str(out_path), scale=PNG_SCALE)
            print("Saved:", out_path)
        except Exception as e:
            print("PNG not saved (kaleido required):", e)


def plot_nb7_original_vs_retrain(
    original: dict[str, np.ndarray],
    retrain: dict[str, np.ndarray],
    out_path: Path,
    smooth_window: int = 50,
) -> None:
    """One subplot per image: overlay NB=7 original (4000 steps) vs retrain (6000 steps)."""
    img_indices = sorted(set(parse_stem(s)[1] for s in original if parse_stem(s)) & set(parse_stem(s)[1] for s in retrain if parse_stem(s)))
    if not img_indices:
        print("No common NB=7 images for original vs retrain, skipping.")
        return
    n = len(img_indices)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[f"img{i}" for i in img_indices],
        horizontal_spacing=0.07,
        shared_yaxes=True,
    )
    for col_idx, img_i in enumerate(img_indices, start=1):
        stem_orig = next((s for s in original if parse_stem(s) and parse_stem(s)[1] == img_i), None)
        stem_ret = next((s for s in retrain if parse_stem(s) and parse_stem(s)[1] == img_i), None)
        if stem_orig is None or stem_ret is None:
            continue
        arr_orig = original[stem_orig]
        arr_ret = retrain[stem_ret]
        for arr, label, color in [
            (arr_orig, "NB=7 original (4000 steps)", "rgb(214,39,40)"),
            (arr_ret, "NB=7 retrain (6000 steps)", "rgb(31,119,180)"),
        ]:
            steps = np.arange(1, len(arr) + 1, dtype=float)
            if len(arr) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(arr, kernel, mode="valid")
                steps = np.arange(smooth_window, len(arr) + 1, dtype=float)
                y = smoothed
            else:
                y = arr
            # Show legend only in first subplot to avoid duplicate entries
            show_in_legend = col_idx == 1
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=y,
                    name=label,
                    line=dict(color=color, width=4),
                    legendgroup=label,
                    showlegend=show_in_legend,
                ),
                row=1,
                col=col_idx,
            )
    for c in range(1, n + 1):
        fig.update_xaxes(
            title_text="Training step",
            row=1,
            col=c,
            title_font_size=FONT_SIZE_AXES,
            tickfont_size=FONT_SIZE_TICKS,
        )
    fig.update_yaxes(
        title_text="Loss (smoothed)",
        row=1,
        col=1,
        title_font_size=FONT_SIZE_AXES,
        tickfont_size=FONT_SIZE_TICKS,
    )
    fig.update_layout(
        title_text="BBBC031 — NB=7: original run vs re-trained run (per image)",
        title_font_size=FONT_SIZE_TITLE,
        font_size=FONT_SIZE_TICKS,
        showlegend=True,
        height=450,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font_size=FONT_SIZE_AXES,
        ),
    )
    fig.update_annotations(font_size=FONT_SIZE_SUBPLOT)
    html_path = out_path if out_path.suffix.lower() == ".html" else out_path.parent / (out_path.stem + ".html")
    fig.write_html(str(html_path))
    print("Saved:", html_path)
    try:
        fig.write_image(str(html_path.with_suffix(".png")), scale=PNG_SCALE)
        print("Saved:", html_path.with_suffix(".png"))
    except Exception:
        pass
    if out_path.suffix.lower() == ".png":
        try:
            fig.write_image(str(out_path), scale=PNG_SCALE)
            print("Saved:", out_path)
        except Exception as e:
            print("PNG not saved (kaleido required):", e)


def plot_nb7_6k_vs_12k(
    losses_6k: dict[str, np.ndarray],
    losses_12k: dict[str, np.ndarray],
    out_path: Path,
    smooth_window: int = 50,
) -> None:
    """Overlay NB=7 re-trained 6000 steps vs 12000 steps (per image)."""
    common = sorted(
        set(parse_stem(s)[1] for s in losses_6k if parse_stem(s))
        & set(parse_stem(s)[1] for s in losses_12k if parse_stem(s))
    )
    if not common:
        print("No common images for 6k vs 12k, skipping.")
        return
    n = len(common)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[f"img{i}" for i in common],
        horizontal_spacing=0.07,
        shared_yaxes=True,
    )
    for col_idx, img_i in enumerate(common, start=1):
        stem_6k = next((s for s in losses_6k if parse_stem(s) and parse_stem(s)[1] == img_i), None)
        stem_12k = next((s for s in losses_12k if parse_stem(s) and parse_stem(s)[1] == img_i), None)
        if stem_6k is None or stem_12k is None:
            continue
        for arr, label, color in [
            (losses_6k[stem_6k], "NB=7 retrain (6000 steps)", "rgb(31,119,180)"),
            (losses_12k[stem_12k], "NB=7 retrain (12000 steps)", "rgb(44,160,44)"),
        ]:
            steps = np.arange(1, len(arr) + 1, dtype=float)
            if len(arr) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(arr, kernel, mode="valid")
                steps = np.arange(smooth_window, len(arr) + 1, dtype=float)
                y = smoothed
            else:
                y = arr
            show_in_legend = col_idx == 1
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=y,
                    name=label,
                    line=dict(color=color, width=4),
                    legendgroup=label,
                    showlegend=show_in_legend,
                ),
                row=1,
                col=col_idx,
            )
    for c in range(1, n + 1):
        fig.update_xaxes(
            title_text="Training step",
            row=1,
            col=c,
            title_font_size=FONT_SIZE_AXES,
            tickfont_size=FONT_SIZE_TICKS,
        )
    fig.update_yaxes(
        title_text="Loss (smoothed)",
        row=1,
        col=1,
        title_font_size=FONT_SIZE_AXES,
        tickfont_size=FONT_SIZE_TICKS,
    )
    fig.update_layout(
        title_text="BBBC031 — NB=7 re-trained: 6000 steps vs 12000 steps (per image)",
        title_font_size=FONT_SIZE_TITLE,
        font_size=FONT_SIZE_TICKS,
        showlegend=True,
        height=450,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font_size=FONT_SIZE_AXES,
        ),
    )
    fig.update_annotations(font_size=FONT_SIZE_SUBPLOT)
    html_path = out_path if out_path.suffix.lower() == ".html" else out_path.parent / (out_path.stem + ".html")
    fig.write_html(str(html_path))
    print("Saved:", html_path)
    try:
        fig.write_image(str(html_path.with_suffix(".png")), scale=PNG_SCALE)
        print("Saved:", html_path.with_suffix(".png"))
    except Exception:
        pass


def plot_loss_by_nb(
    losses: dict[str, np.ndarray],
    out_path: Path,
    smooth_window: int = 50,
    nb7_original: dict[str, np.ndarray] | None = None,
) -> None:
    """One plot per NB: comparison across images (smoothed). If nb7_original given, 3 cols: NB=3, NB=7 original, NB=7 retrain."""
    by_nb: dict[int, list[tuple[int, np.ndarray]]] = {}
    for stem, arr in losses.items():
        parsed = parse_stem(stem)
        if parsed is None:
            continue
        nb, img_idx = parsed
        by_nb.setdefault(nb, []).append((img_idx, arr))
    if not by_nb:
        print("No loss parseable by NB, skipping plot by NB.")
        return
    # Build column list: [NB=3], then optionally [NB=7 original], then [NB=7 from losses]
    columns_data: list[tuple[str, list[tuple[int, np.ndarray]]]] = []
    if 3 in by_nb:
        columns_data.append(("NB = 3", sorted(by_nb[3])))
    if nb7_original and 7 in by_nb:
        orig_list = [(parse_stem(s)[1], arr) for s, arr in nb7_original.items() if parse_stem(s)]
        if orig_list:
            columns_data.append(("NB = 7 (original, 4000 steps)", sorted(orig_list)))
        columns_data.append(("NB = 7 (retrain, 6000 steps)", sorted(by_nb[7])))
    elif 7 in by_nb:
        columns_data.append(("NB = 7", sorted(by_nb[7])))
    if not columns_data:
        return
    n_cols = len(columns_data)
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[title for title, _ in columns_data],
        horizontal_spacing=0.08,
    )
    colors = [
        "rgb(31,119,180)",
        "rgb(255,127,14)",
        "rgb(44,160,44)",
        "rgb(214,39,40)",
        "rgb(148,103,189)",
    ]
    for col_idx, (title, items) in enumerate(columns_data, start=1):
        for img_idx, arr in items:
            steps = np.arange(1, len(arr) + 1, dtype=float)
            if len(arr) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(arr, kernel, mode="valid")
                steps = np.arange(smooth_window, len(arr) + 1, dtype=float)
                y = smoothed
            else:
                y = arr
            color = colors[img_idx % len(colors)]
            # Show legend only from first column to avoid img0..img4 repeated 3 times
            show_in_legend = col_idx == 1
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=y,
                    name=f"img{img_idx}",
                    line=dict(color=color, width=3.5),
                    legendgroup=f"img{img_idx}",
                    showlegend=show_in_legend,
                ),
                row=1,
                col=col_idx,
            )
    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text="Training step",
            row=1,
            col=c,
            title_font_size=FONT_SIZE_AXES,
            tickfont_size=FONT_SIZE_TICKS,
        )
    fig.update_yaxes(
        title_text="Loss (smoothed)",
        row=1,
        col=1,
        title_font_size=FONT_SIZE_AXES,
        tickfont_size=FONT_SIZE_TICKS,
    )
    fig.update_layout(
        title_text="BBBC031 MNCA — Loss by neighborhood size (all images)",
        title_font_size=FONT_SIZE_TITLE,
        font_size=FONT_SIZE_TICKS,
        showlegend=True,
        height=480,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font_size=FONT_SIZE_AXES,
        ),
    )
    fig.update_annotations(font_size=FONT_SIZE_SUBPLOT)
    html_path = out_path if out_path.suffix.lower() == ".html" else out_path.parent / (out_path.stem + ".html")
    fig.write_html(str(html_path))
    print("Saved:", html_path)
    try:
        fig.write_image(str(html_path.with_suffix(".png")), scale=PNG_SCALE)
        print("Saved:", html_path.with_suffix(".png"))
    except Exception:
        pass
    if out_path.suffix.lower() == ".png":
        try:
            fig.write_image(str(out_path), scale=PNG_SCALE)
            print("Saved:", out_path)
        except Exception as e:
            print("PNG not saved (kaleido required):", e)


def analyze_nb7_trend(losses: dict[str, np.ndarray], last_n: int = 500) -> None:
    """
    Analyze loss trend over the last `last_n` steps for NB=7.
    Used to decide whether extending training might lower the loss.
    """
    nb7 = {k: v for k, v in losses.items() if parse_stem(k) and parse_stem(k)[0] == 7}
    if not nb7:
        return
    print("\n--- NB=7 loss trend (last {} steps) ---".format(last_n))
    for stem in sorted(nb7.keys()):
        arr = nb7[stem]
        parsed = parse_stem(stem)
        if parsed is None or len(arr) < last_n:
            continue
        tail = arr[-last_n:]
        x = np.arange(len(tail), dtype=float)
        # linear regression: loss = a + b*step
        x_mean = x.mean()
        y_mean = tail.mean()
        b = np.dot(x - x_mean, tail - y_mean) / (np.dot(x - x_mean, x - x_mean) + 1e-12)
        a = y_mean - b * x_mean
        trend = "decreasing" if b < -1e-6 else ("increasing" if b > 1e-6 else "flat")
        print(f"  NB=7, img{parsed[1]}: slope={b:.2e} ({trend}), last={arr[-1]:.4e}")
    print("  → Exploded loss (img0, img1 ~1e3): even with negative slope, extending is of little use;")
    print("    try lower LR or curriculum. Not worth only increasing steps.")
    print("  → High but not exploded (img2–4): slightly negative slope; you can try")
    print("    8k–10k steps or slightly lower LR to see if loss drops further.")


def print_summary(losses: dict[str, np.ndarray], models_dir: Path, figs_dir: Path) -> None:
    """Print a text summary: final loss, models, figures."""
    print("\n" + "=" * 60)
    print("BBBC031 MNCA OUTPUT SUMMARY")
    print("=" * 60)
    print("\n--- Loss (last value and mean of last 200 steps) ---")
    for stem in sorted(losses.keys()):
        arr = losses[stem]
        last = float(arr[-1]) if len(arr) else float("nan")
        tail = arr[-200:] if len(arr) >= 200 else arr
        mean_tail = float(np.mean(tail)) if len(tail) else float("nan")
        parsed = parse_stem(stem)
        nb_img = f"NB={parsed[0]}, img{parsed[1]}" if parsed else stem
        print(f"  {nb_img}: last={last:.4e}, mean_last200={mean_tail:.4e} (n_steps={len(arr)})")
    print("\n--- Checkpoint (.pth) ---")
    for f in sorted(models_dir.glob("*.pth")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.2f} MB)")
    print("\n--- Figure (figs/bbbc031) ---")
    if figs_dir.exists():
        for f in sorted(figs_dir.glob("*.png")):
            print(f"  {f.name}")
        for f in sorted(figs_dir.glob("*.mp4")):
            print(f"  {f.name}")
    else:
        print("  (figures folder not found)")
    print("=" * 60)


def inspect_model(checkpoint_path: Path) -> None:
    """Load a checkpoint and print parameter count (requires torch and mix_NCA)."""
    try:
        import torch
        from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
        from mix_NCA.utils_images import standard_update_net
    except ImportError as e:
        print("Model inspect skipped (import failed):", e)
        return
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    n_params = sum(p.numel() for p in state.values() if isinstance(p, torch.Tensor))
    print(f"\nCheckpoint: {checkpoint_path.name}")
    print(f"  state_dict keys: {list(state.keys())[:8]}...")
    print(f"  Total parameters: {n_params:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze loss, models and results for BBBC031 from MNCA_bbbc031_outputs."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "results_extended" / "MNCA_bbbc031_outputs",
        help="Root output folder (contains models/ and figs/).",
    )
    parser.add_argument(
        "--out_plots",
        type=Path,
        default=None,
        help="Folder to save plots (default: output_dir/analysis).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="Window for loss smoothing in plots.",
    )
    parser.add_argument(
        "--inspect_checkpoint",
        type=Path,
        default=None,
        help="Path to a .pth to print model info.",
    )
    parser.add_argument(
        "--nb7_retrain_dir",
        type=Path,
        default=None,
        help="If set, load NB=7 loss and use for plots (e.g. output_dir/models/nb7_retrain).",
    )
    parser.add_argument(
        "--nb7_retrain_12k_dir",
        type=Path,
        default=None,
        help="If set, load NB=7 12k-step loss and plot vs 6k (e.g. output_dir/models/nb7_retrain_12k).",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    models_dir = output_dir / "models"
    nb7_retrain_dir = args.nb7_retrain_dir or (models_dir / "nb7_retrain")
    if not nb7_retrain_dir.exists():
        nb7_retrain_dir = None
    nb7_retrain_12k_dir = args.nb7_retrain_12k_dir or (models_dir / "nb7_retrain_12k")
    if not nb7_retrain_12k_dir.exists():
        nb7_retrain_12k_dir = None
    figs_dir = output_dir / "figs" / "bbbc031"
    out_plots = args.out_plots or (output_dir / "analysis")
    out_plots.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        print(f"Models folder not found: {models_dir}")
        sys.exit(1)

    losses = load_all_losses(models_dir, nb7_retrain_dir=nb7_retrain_dir)
    print(f"Loaded {len(losses)} loss curves from {models_dir}" + (
        f" (NB=7 from {nb7_retrain_dir})" if nb7_retrain_dir else ""
    ))

    nb7_original = None
    if nb7_retrain_dir is not None:
        nb7_original = load_nb7_original_losses(models_dir)
        if nb7_original:
            print(f"Loaded {len(nb7_original)} original NB=7 curves (for comparison)")

    plot_loss_curves(losses, out_plots / "bbbc031_loss_curves.html", smooth_window=args.smooth)
    plot_loss_by_nb(
        losses,
        out_plots / "bbbc031_loss_by_NB.html",
        smooth_window=args.smooth,
        nb7_original=nb7_original,
    )
    if nb7_original and nb7_retrain_dir is not None:
        nb7_retrain = {k: v for k, v in losses.items() if parse_stem(k) and parse_stem(k)[0] == 7}
        if nb7_retrain:
            plot_nb7_original_vs_retrain(
                nb7_original,
                nb7_retrain,
                out_plots / "bbbc031_NB7_original_vs_retrain.html",
                smooth_window=args.smooth,
            )
    if nb7_retrain_12k_dir is not None:
        losses_12k = {}
        for f in sorted(nb7_retrain_12k_dir.glob("*_loss.npy")):
            stem = f.stem.removesuffix("_loss")
            if parse_stem(stem) and parse_stem(stem)[0] == 7:
                losses_12k[stem] = np.load(f)
                print(f"  (NB=7 retrain 12k) {stem}: {len(losses_12k[stem])} steps")
        if losses_12k:
            nb7_retrain_6k = {k: v for k, v in losses.items() if parse_stem(k) and parse_stem(k)[0] == 7}
            if nb7_retrain_6k:
                plot_nb7_6k_vs_12k(
                    nb7_retrain_6k,
                    losses_12k,
                    out_plots / "bbbc031_NB7_6k_vs_12k.html",
                    smooth_window=args.smooth,
                )
    print_summary(losses, models_dir, figs_dir)
    analyze_nb7_trend(losses, last_n=500)

    if args.inspect_checkpoint and args.inspect_checkpoint.exists():
        inspect_model(args.inspect_checkpoint)
    elif (models_dir / "bbbc031_stochastic_NB3_img0.pth").exists():
        inspect_model(models_dir / "bbbc031_stochastic_NB3_img0.pth")


if __name__ == "__main__":
    main()
