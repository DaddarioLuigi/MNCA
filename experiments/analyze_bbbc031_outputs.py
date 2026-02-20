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


def load_all_losses(models_dir: Path) -> dict[str, np.ndarray]:
    """Load all *_loss.npy files in models_dir. Key: stem (e.g. bbbc031_stochastic_NB3_img0)."""
    out = {}
    for f in sorted(models_dir.glob("*_loss.npy")):
        stem = f.stem.removesuffix("_loss")
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


def plot_loss_curves(
    losses: dict[str, np.ndarray],
    out_path: Path,
    smooth_window: int = 50,
) -> None:
    """Plot all loss curves (raw + smoothed) with Plotly and save HTML (and optional PNG)."""
    n = len(losses)
    if n == 0:
        print("No loss files found, skipping plot.")
        return
    cols = 2
    rows = (n + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[
            f"NB={parsed[0]}, img{parsed[1]}" if (parsed := parse_stem(s)) else s
            for s, _ in sorted(losses.items())
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    for idx, (stem, arr) in enumerate(sorted(losses.items())):
        row, col = idx // cols + 1, idx % cols + 1
        steps = np.arange(1, len(arr) + 1, dtype=float)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=arr,
                name="loss raw",
                line=dict(color="rgba(31,119,180,0.4)", width=1),
                legendgroup=stem,
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
                    name=f"mean {smooth_window}",
                    line=dict(color="rgb(31,119,180)", width=2),
                    legendgroup=stem,
                ),
                row=row,
                col=col,
            )
    fig.update_xaxes(title_text="Training step", row=rows, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_layout(
        title_text="BBBC031 MNCA — Loss per training run (4000 steps = gradient updates)",
        title_font_size=14,
        showlegend=True,
        height=300 * rows,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    html_path = out_path.with_suffix(".html") if out_path.suffix.lower() == ".png" else out_path
    if html_path.suffix.lower() != ".html":
        html_path = out_path.parent / (out_path.stem + ".html")
    fig.write_html(str(html_path))
    print("Saved:", html_path)
    if out_path.suffix.lower() == ".png":
        try:
            fig.write_image(str(out_path), scale=2)
            print("Saved:", out_path)
        except Exception as e:
            print("PNG not saved (kaleido required):", e)


def plot_loss_by_nb(
    losses: dict[str, np.ndarray],
    out_path: Path,
    smooth_window: int = 50,
) -> None:
    """One plot per NB: comparison across the 5 images (smoothed curves) with Plotly."""
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
    n_nb = len(by_nb)
    fig = make_subplots(
        rows=1,
        cols=n_nb,
        subplot_titles=[f"Neighborhood size = {nb}" for nb in sorted(by_nb.keys())],
        horizontal_spacing=0.08,
    )
    colors = [
        "rgb(31,119,180)",
        "rgb(255,127,14)",
        "rgb(44,160,44)",
        "rgb(214,39,40)",
        "rgb(148,103,189)",
    ]
    for col_idx, (nb, items) in enumerate(sorted(by_nb.items()), start=1):
        for i, (img_idx, arr) in enumerate(sorted(items)):
            steps = np.arange(1, len(arr) + 1, dtype=float)
            if len(arr) >= smooth_window:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(arr, kernel, mode="valid")
                steps = np.arange(smooth_window, len(arr) + 1, dtype=float)
                y = smoothed
            else:
                y = arr
            color = colors[img_idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=y,
                    name=f"img{img_idx}",
                    line=dict(color=color, width=1.5),
                    legendgroup=f"nb{nb}",
                ),
                row=1,
                col=col_idx,
            )
    fig.update_xaxes(title_text="Training step", row=1, col=1)
    fig.update_yaxes(title_text="Loss (smoothed)", row=1, col=1)
    fig.update_layout(
        title_text="BBBC031 MNCA — Loss comparison by neighborhood size",
        title_font_size=14,
        showlegend=True,
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    html_path = out_path.with_suffix(".html") if out_path.suffix.lower() == ".png" else out_path
    if html_path.suffix.lower() != ".html":
        html_path = out_path.parent / (out_path.stem + ".html")
    fig.write_html(str(html_path))
    print("Saved:", html_path)
    if out_path.suffix.lower() == ".png":
        try:
            fig.write_image(str(out_path), scale=2)
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
    args = parser.parse_args()
    output_dir = args.output_dir
    models_dir = output_dir / "models"
    figs_dir = output_dir / "figs" / "bbbc031"
    out_plots = args.out_plots or (output_dir / "analysis")
    out_plots.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        print(f"Models folder not found: {models_dir}")
        sys.exit(1)

    losses = load_all_losses(models_dir)
    print(f"Loaded {len(losses)} loss curves from {models_dir}")

    plot_loss_curves(losses, out_plots / "bbbc031_loss_curves.html", smooth_window=args.smooth)
    plot_loss_by_nb(losses, out_plots / "bbbc031_loss_by_NB.html", smooth_window=args.smooth)
    print_summary(losses, models_dir, figs_dir)
    analyze_nb7_trend(losses, last_n=500)

    if args.inspect_checkpoint and args.inspect_checkpoint.exists():
        inspect_model(args.inspect_checkpoint)
    elif (models_dir / "bbbc031_stochastic_NB3_img0.pth").exists():
        inspect_model(models_dir / "bbbc031_stochastic_NB3_img0.pth")


if __name__ == "__main__":
    main()
