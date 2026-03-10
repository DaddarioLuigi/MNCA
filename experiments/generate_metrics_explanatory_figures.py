#!/usr/bin/env python3
"""
Generate explanatory figures for each of the six biological metrics.
Output: thesis-latex/figs/metrics_explanatory/*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from mix_NCA.TissueModel import ComplexCellType

N_CELL_TYPES = len(ComplexCellType)
CELL_NAMES = ["EMPTY", "STEM", "INT1", "INT2", "DIFF1", "DIFF2"]
CELL_NAMES_NO_EMPTY = ["STEM", "INT1", "INT2", "DIFF1", "DIFF2"]


def load_sample_grids():
    """Load a few grids from Step 3 histories for illustration."""
    hist_path = repo_root / "histories_300_100.npy"
    if not hist_path.exists():
        return None
    histories = np.load(hist_path, allow_pickle=True)
    grids = []
    for i in [0, 50, 149, 200, 299]:
        h = histories[i]
        if isinstance(h, np.ndarray) and h.ndim == 3:
            grids.append(h[-1])  # final state
        else:
            grids.append(np.array(h[-1]))
    return np.array(grids)


def fig_kl_divergence(out_path):
    """KL: pool cells → count by type → normalize → KL(p||q)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    types = np.arange(5)
    p_true = np.array([0.15, 0.25, 0.30, 0.18, 0.12])
    p_gen = np.array([0.12, 0.28, 0.32, 0.20, 0.08])
    width = 0.35
    axes[0].bar(types - width/2, p_true, width, label="True", color="#3366CC", alpha=0.8)
    axes[0].bar(types + width/2, p_gen, width, label="Generated", color="#CC6633", alpha=0.8)
    axes[0].set_xticks(types)
    axes[0].set_xticklabels(CELL_NAMES_NO_EMPTY)
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Pooled cell-type distributions")
    axes[0].legend()
    axes[0].set_ylim(0, 0.4)
    kl = np.sum(p_true * np.log((p_true + 1e-10) / (p_gen + 1e-10)))
    axes[1].axis("off")
    axes[1].text(0.5, 0.7, "KL divergence", fontsize=14, ha="center", fontweight="bold")
    axes[1].text(0.5, 0.5, r"$\sum p \log(p/q)$", fontsize=16, ha="center")
    axes[1].text(0.5, 0.3, f"Example: KL ≈ {kl:.3f}", fontsize=12, ha="center")
    axes[1].text(0.5, 0.1, "Compares global mix of cell types\n(no spatial info)", fontsize=10, ha="center", style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_chi_square(out_path):
    """Chi-square: same data as KL, different formula."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    types = np.arange(5)
    p_true = np.array([0.15, 0.25, 0.30, 0.18, 0.12])
    p_gen = np.array([0.12, 0.28, 0.32, 0.20, 0.08])
    width = 0.35
    axes[0].bar(types - width/2, p_true, width, label="True", color="#3366CC", alpha=0.8)
    axes[0].bar(types + width/2, p_gen, width, label="Generated", color="#CC6633", alpha=0.8)
    axes[0].set_xticks(types)
    axes[0].set_xticklabels(CELL_NAMES_NO_EMPTY)
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Pooled cell-type distributions (same as KL)")
    axes[0].legend()
    axes[0].set_ylim(0, 0.4)
    chi = np.sum((p_true - p_gen)**2 / (p_true + p_gen + 1e-10))
    axes[1].axis("off")
    axes[1].text(0.5, 0.7, "Chi-square distance", fontsize=14, ha="center", fontweight="bold")
    axes[1].text(0.5, 0.5, r"$\sum \frac{(p-q)^2}{p+q}$", fontsize=16, ha="center")
    axes[1].text(0.5, 0.3, f"Example: χ² ≈ {chi:.3f}", fontsize=12, ha="center")
    axes[1].text(0.5, 0.1, "Same data as KL:\nglobal cell-type proportions", fontsize=10, ha="center", style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_categorical_mmd(out_path):
    """MMD: compare grids position-by-position with kernel."""
    grids = load_sample_grids()
    if grids is None:
        # Fallback: create toy grids
        g1 = np.zeros((20, 20), dtype=int)
        g1[5:15, 5:15] = 1
        g1[8:12, 8:12] = 2
        g2 = g1.copy()
        g2[6:14, 6:14] = 1  # slightly different
        g2[9:11, 9:11] = 3
        grids = np.array([g1, g2])
    g_true = grids[0]
    g_gen = grids[1] if len(grids) > 1 else grids[0]
    # Crop for clarity
    sz = min(24, g_true.shape[0])
    g_true = g_true[:sz, :sz]
    g_gen = g_gen[:sz, :sz]
    match = (g_true == g_gen) & (g_true > 0)
    colors = plt.cm.tab10(np.linspace(0, 1, N_CELL_TYPES))
    cmap = plt.cm.colors.ListedColormap(colors)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(g_true, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1)
    axes[0].set_title("True grid")
    axes[0].axis("off")
    axes[1].imshow(g_gen, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1)
    axes[1].set_title("Generated grid")
    axes[1].axis("off")
    overlay = np.ma.masked_where(~match, np.ones_like(g_true))
    axes[2].imshow(g_gen, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1)
    axes[2].imshow(overlay, cmap="Greens", alpha=0.4, vmin=0, vmax=1)
    axes[2].set_title("Kernel: match at each position")
    axes[2].axis("off")
    fig.suptitle("Categorical MMD: spatial layout similarity", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_tumor_size(out_path):
    """Tumor size: count non-empty cells per grid → Wasserstein."""
    grids = load_sample_grids()
    if grids is None:
        true_sizes = np.array([120, 180, 95, 210, 150])
        gen_sizes = np.array([115, 175, 100, 205, 155])
    else:
        true_sizes = np.array([(g > 0).sum() for g in grids])
        gen_sizes = true_sizes + np.random.randint(-15, 16, size=len(grids))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(true_sizes, bins=8, alpha=0.7, label="True", color="#3366CC", edgecolor="black")
    axes[0].hist(gen_sizes, bins=8, alpha=0.7, label="Generated", color="#CC6633", edgecolor="black")
    axes[0].set_xlabel("Tumor size (non-empty cell count)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution over N simulations")
    axes[0].legend()
    axes[1].axis("off")
    axes[1].text(0.5, 0.7, "Tumor size difference", fontsize=14, ha="center", fontweight="bold")
    axes[1].text(0.5, 0.5, "Wasserstein distance\n(normalized)", fontsize=12, ha="center")
    axes[1].text(0.5, 0.3, "One scalar per grid:\n# non-empty cells", fontsize=10, ha="center")
    axes[1].text(0.5, 0.1, "Compares spread of total\ncell counts over simulations", fontsize=10, ha="center", style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_border_size(out_path):
    """Border size: binary mask → Laplacian edge detection → count edges."""
    grids = load_sample_grids()
    if grids is None:
        g = np.zeros((30, 30), dtype=int)
        g[8:22, 8:22] = 1
        g[10:20, 10:20] = 2
    else:
        g = grids[0][:30, :30]
    mask = (g > 0).astype(float)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8
    from scipy.ndimage import convolve
    edges = np.abs(convolve(mask, kernel, mode="constant", cval=0)) > 0.25
    border_pixels = np.sum(edges)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, N_CELL_TYPES))
    cmap = plt.cm.colors.ListedColormap(colors)
    axes[0].imshow(g, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1)
    axes[0].set_title("Grid (cell types)")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="binary", vmin=0, vmax=1)
    axes[1].set_title("Binary mask\n(1 = any cell)")
    axes[1].axis("off")
    overlay = np.ma.masked_where(~edges, np.ones_like(edges))
    axes[2].imshow(mask, cmap="Greys", vmin=0, vmax=1)
    axes[2].imshow(overlay, cmap="Reds", alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title(f"Edge detection (3×3 Laplacian)\nBorder size = {border_pixels} px")
    axes[2].axis("off")
    fig.suptitle("Border size: tissue–empty interface length", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def fig_spatial_variance(out_path):
    """Spatial variance: center of mass → mean squared distance."""
    grids = load_sample_grids()
    if grids is None:
        g = np.zeros((25, 25), dtype=int)
        g[5:20, 5:20] = 1
    else:
        g = grids[0][:25, :25]
    mask = (g > 0)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        xs, ys = np.array([12]), np.array([12])
    cx, cy = xs.mean(), ys.mean()
    dist_sq = (xs - cx)**2 + (ys - cy)**2
    spvar = dist_sq.mean()
    colors = plt.cm.tab10(np.linspace(0, 1, N_CELL_TYPES))
    cmap = plt.cm.colors.ListedColormap(colors)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.imshow(g, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1)
    ax.plot(cx, cy, "r*", markersize=14, label="Center of mass")
    for i in range(0, len(xs), max(1, len(xs)//30)):
        ax.plot([cx, xs[i]], [cy, ys[i]], "r-", alpha=0.3, linewidth=0.8)
    ax.set_title(f"Spatial variance: mean dist² from center ≈ {spvar:.0f}")
    ax.axis("off")
    ax.legend(loc="upper right")
    fig.suptitle("Spatial variance: spread around center of mass", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    out_dir = repo_root / "thesis-latex" / "figs" / "metrics_explanatory"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_kl_divergence(out_dir / "metric_kl_divergence.png")
    fig_chi_square(out_dir / "metric_chi_square.png")
    fig_categorical_mmd(out_dir / "metric_categorical_mmd.png")
    fig_tumor_size(out_dir / "metric_tumor_size.png")
    fig_border_size(out_dir / "metric_border_size.png")
    fig_spatial_variance(out_dir / "metric_spatial_variance.png")
    print(f"Saved 6 figures to {out_dir}")


if __name__ == "__main__":
    main()
