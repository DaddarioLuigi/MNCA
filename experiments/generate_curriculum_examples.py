#!/usr/bin/env python3
"""
Generate one figure per neighborhood size (NB=2, 3, 5) for the curriculum experiment.
Each figure: 3 rows (GT, Mixture, Stochastic) x 12 columns (3 simulations x 4 time points).
Panels are large enough to be readable; one figure per NB so the thesis can show them
one per page or in sequence.

Allineato all'app Streamlit: stesso file histories, stessi checkpoint, Stochastic con
sample_non_differentiable=True e seed fisso (42) per riproducibilità. In Streamlit
lo stochastic non usa seed quindi la stessa simulazione può dare rollout diversi a ogni run.

Saves to thesis-latex/figs/curriculum/curriculum_examples_NB2.png, NB3.png, NB5.png.

Usage (from repo root):
  python experiments/generate_curriculum_examples.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from mix_NCA.TissueModel import ComplexCellType
from mix_NCA.utils_simulations import grid_to_channels_batch, classification_update_net
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise

N_CELL_TYPES = len(ComplexCellType)
HIDDEN_DIM = 128
STATE_DIM = 6
N_RULES = 5
TIME_POINTS = [10, 50, 100, 500]
SAMPLE_INDICES = [0, 149, 299]  # 3 simulations
NB_SIZES = [2, 3, 5]

# One figure per NB: 3 rows x 12 cols, large panels for readability
FIG_SIZE = (16, 6)
DPI = 180
FONTSIZE = 10


def _make_update_net_fn(device):
    def fn(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
        return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)
    return fn


def _get_final_grid(out, n_cell_types):
    """Extract final grid from model output (single rollout)."""
    if isinstance(out, tuple):
        out = out[1]
    if out.dim() == 5:
        final_state = out[-1]
    else:
        final_state = out[-1] if out.dim() == 4 else out
    return final_state[0, :n_cell_types].argmax(dim=0).cpu().numpy()


def _get_grid_at_step(out, step_index: int, n_cell_types: int):
    """Extract grid at a given step from full history (for single long rollout)."""
    if isinstance(out, tuple):
        out = out[1]
    # out: (T+1, B, C, H, W) or (T+1, C, H, W)
    if out.dim() == 5:
        frame = out[step_index, 0, :n_cell_types]
    else:
        frame = out[step_index, :n_cell_types]
    return frame.argmax(dim=0).cpu().numpy()


def main():
    device = os.environ.get("DEVICE", "cpu")
    histories_path = repo_root / "notebooks" / "histories_300x500.npy"
    if not histories_path.exists():
        histories_path = (repo_root / "experiments" / ".." / "notebooks" / "histories_300x500.npy").resolve()
    if not histories_path.exists():
        print("Error: histories_300x500.npy not found.")
        sys.exit(1)
    results_base = repo_root / "experiments" / "results_extended" / "tissue_simulation_extended"
    out_dir = repo_root / "thesis-latex" / "figs" / "curriculum"
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = np.load(histories_path, allow_pickle=True)
    colors = plt.cm.tab10(np.linspace(0, 1, N_CELL_TYPES))
    cmap = plt.cm.colors.ListedColormap(colors)

    for nb in NB_SIZES:
        mix_ckpt = results_base / f"NB_{nb}" / "mixture_nca_curriculum.pt"
        stoch_ckpt = results_base / f"NB_{nb}" / "stochastic_mix_nca_curriculum.pt"
        if not mix_ckpt.exists() or not stoch_ckpt.exists():
            print(f"Error: checkpoints not found for NB={nb}.")
            continue
        update_net_fn = _make_update_net_fn(device)
        mix_model = ExtendedMixtureNCA(
            update_nets=update_net_fn, hidden_dim=HIDDEN_DIM, maintain_seed=False, use_alive_mask=False,
            state_dim=STATE_DIM, num_rules=N_RULES, residual=False, temperature=3,
            neighborhood_size=nb, device=device,
        )
        mix_model.load_state_dict(torch.load(mix_ckpt, map_location=device, weights_only=True))
        mix_model = mix_model.to(device).eval()
        stoch_model = ExtendedMixtureNCANoise(
            update_nets=update_net_fn, hidden_dim=HIDDEN_DIM, maintain_seed=False, use_alive_mask=False,
            state_dim=STATE_DIM, num_rules=N_RULES, residual=False, temperature=3,
            neighborhood_size=nb, device=device,
        )
        stoch_model.load_state_dict(torch.load(stoch_ckpt, map_location=device, weights_only=True))
        stoch_model = stoch_model.to(device).eval()

        # One figure per NB: 3 rows x 12 columns
        fig, axes = plt.subplots(3, 12, figsize=FIG_SIZE, dpi=DPI)
        for sim_idx, sample_idx in enumerate(SAMPLE_INDICES):
            if sample_idx >= len(histories):
                continue
            hist = histories[sample_idx]
            if isinstance(hist, np.ndarray) and hist.ndim == 3:
                frames = [np.asarray(hist[t]) for t in range(hist.shape[0])]
            else:
                frames = [np.asarray(f) for f in hist]
            x0 = grid_to_channels_batch([frames[0]], n_cell_types=N_CELL_TYPES, device=device)
            col_offset = sim_idx * 4

            # Row 0: Ground truth
            for c, t in enumerate(TIME_POINTS):
                t_gt = min(t, len(frames) - 1)
                ax = axes[0, col_offset + c]
                ax.imshow(frames[t_gt], cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1, interpolation="nearest")
                ax.set_title(f"t={t}", fontsize=FONTSIZE)
                ax.set_xticks([])
                ax.set_yticks([])

            # Row 1: Mixture NCA — un solo rollout lungo (come Streamlit), estraiamo frame a t=10,50,100,500.
            with torch.no_grad():
                max_steps = max(TIME_POINTS)
                out_mix = mix_model(x0, max_steps, return_history=True)
                for c, t in enumerate(TIME_POINTS):
                    idx = t - 1  # stato a t=10 è all'indice 9
                    grid = _get_grid_at_step(out_mix, idx, N_CELL_TYPES)
                    ax = axes[1, col_offset + c]
                    ax.imshow(grid, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1, interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Stochastic: un solo rollout lungo (come Streamlit), poi estraiamo i frame a t=10,50,100,500.
                # Così la traiettoria è la stessa che si vede in Streamlit con stesso seed e stesso n_steps.
                torch.manual_seed(42)
                if hasattr(np.random, "default_rng"):
                    np.random.default_rng(42)
                else:
                    np.random.seed(42)
                # La history del modello ha length = num_steps: frame i = stato dopo (i+1) step; non c'è t=0.
                max_steps = max(TIME_POINTS)
                out = stoch_model(x0, max_steps, return_history=True, sample_non_differentiable=True)
                for c, t in enumerate(TIME_POINTS):
                    idx = t - 1  # stato a t=10 è all'indice 9
                    grid = _get_grid_at_step(out, idx, N_CELL_TYPES)
                    ax = axes[2, col_offset + c]
                    ax.imshow(grid, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1, interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])

        axes[0, 0].set_ylabel("Ground truth", fontsize=FONTSIZE)
        axes[1, 0].set_ylabel("Mixture NCA", fontsize=FONTSIZE)
        axes[2, 0].set_ylabel("Stochastic Mixture NCA", fontsize=FONTSIZE)

        # Titles: first col of each block "Sim X / t=10", others t=50,100,500
        for sim_idx in range(3):
            col_offset = sim_idx * 4
            axes[0, col_offset].set_title(f"Sim {SAMPLE_INDICES[sim_idx]}\nt=10", fontsize=FONTSIZE)
            for k in range(1, 4):
                axes[0, col_offset + k].set_title(f"t={TIME_POINTS[k]}", fontsize=FONTSIZE)

        plt.suptitle(f"Curriculum: NB={nb} — 3 simulations × (GT, Mixture, Stochastic) at t=10, 50, 100, 500", fontsize=12)
        plt.tight_layout(rect=[0.02, 0, 1, 0.96])
        out_path = out_dir / f"curriculum_examples_NB{nb}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
