#!/usr/bin/env python3
"""
Generate a side-by-side figure: ground truth (simulator) final state vs
curriculum-trained Mixture NCA final state, same initial condition.
Saves to thesis-latex/figs/curriculum/curriculum_gt_vs_prediction.png.

Usage (from repo root or experiments/):
  python experiments/generate_curriculum_gt_vs_prediction.py
  # Optional: SAMPLE_IDX=0 N_STEPS=100 NB=3
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

N_CELL_TYPES = len(ComplexCellType)
HIDDEN_DIM = 128
STATE_DIM = 6
N_RULES = 5


def _make_update_net_fn(device):
    def fn(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
        return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)
    return fn


def main():
    sample_idx = int(os.environ.get("SAMPLE_IDX", "0"))
    n_steps = int(os.environ.get("N_STEPS", "100"))
    nb_size = int(os.environ.get("NB", "3"))
    device = os.environ.get("DEVICE", "cpu")

    histories_path = repo_root / "notebooks" / "histories_300x500.npy"
    if not histories_path.exists():
        histories_path = repo_root / "experiments" / ".." / "notebooks" / "histories_300x500.npy"
        histories_path = histories_path.resolve()
    if not histories_path.exists():
        print("Error: histories_300x500.npy not found. Expected notebooks/histories_300x500.npy")
        sys.exit(1)

    results_base = repo_root / "experiments" / "results_extended" / "tissue_simulation_extended"
    ckpt_path = results_base / f"NB_{nb_size}" / "mixture_nca_curriculum.pt"
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out_path = repo_root / "thesis-latex" / "figs" / "curriculum" / "curriculum_gt_vs_prediction.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading histories: {histories_path}")
    histories = np.load(histories_path, allow_pickle=True)
    if sample_idx >= len(histories):
        sample_idx = 0
    hist = histories[sample_idx]
    if isinstance(hist, np.ndarray) and hist.ndim == 3:
        frames = [hist[t] for t in range(hist.shape[0])]
    else:
        frames = list(hist)

    t_final = min(n_steps, len(frames) - 1)
    gt_final = np.asarray(frames[t_final])
    x0_grid = np.asarray(frames[0])

    x0 = grid_to_channels_batch([x0_grid], n_cell_types=N_CELL_TYPES, device=device)
    update_net_fn = _make_update_net_fn(device)
    model = ExtendedMixtureNCA(
        update_nets=update_net_fn,
        hidden_dim=HIDDEN_DIM,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=STATE_DIM,
        num_rules=N_RULES,
        residual=False,
        temperature=3,
        neighborhood_size=nb_size,
        device=device,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        out = model(x0, n_steps, return_history=True)
    # Model returns stacked frames (T+1, B, C, H, W) when return_history=True
    if out.dim() == 5:
        final_state = out[-1]
    else:
        final_state = out
    # final_state: (B, C, H, W) -> take first sample, argmax over channels -> (H, W)
    pred_grid = final_state[0, :N_CELL_TYPES].argmax(dim=0).cpu().numpy()

    colors = plt.cm.tab10(np.linspace(0, 1, N_CELL_TYPES))
    cmap = plt.cm.colors.ListedColormap(colors)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_final, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1)
    axes[0].set_title("Ground truth (simulator), $t=" + str(t_final) + "$")
    axes[0].axis("off")
    axes[1].imshow(pred_grid, cmap=cmap, vmin=0, vmax=N_CELL_TYPES - 1)
    axes[1].set_title("Mixture NCA (curriculum, $NB=" + str(nb_size) + "$), $t=" + str(n_steps) + "$")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
