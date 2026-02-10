import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from mix_NCA.TissueModel import ComplexCellType
from mix_NCA.utils_simulations import classification_update_net, grid_to_channels_batch
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise


def _make_update_net_fn(device: str):
    def update_net_wrapper(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
        return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)

    return update_net_wrapper


def _as_label_grid(frame_like, n_cell_types: int) -> np.ndarray:
    """
    Convert a frame-like tensor/ndarray to a 2D numpy array of integer cell labels.
    Accepts:
      - torch.Tensor: (C,H,W) or (B,C,H,W)
      - np.ndarray:   (C,H,W) or (B,C,H,W) or already (H,W)
    """
    if isinstance(frame_like, np.ndarray):
        arr = frame_like
        if arr.ndim == 2:
            return arr
        t = torch.from_numpy(arr)
    elif isinstance(frame_like, torch.Tensor):
        t = frame_like
    else:
        raise TypeError(f"Unsupported frame type: {type(frame_like)}")

    t = t.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3:
        t = t[:n_cell_types].argmax(dim=0)
    if t.ndim != 2:
        raise ValueError(f"Unexpected frame shape after conversion: {tuple(t.shape)}")
    return t.numpy()


def _flatten_history(history_like, n_cell_types: int) -> List[np.ndarray]:
    """
    Flatten model return_history output to a list of label grids.
    Handles:
      - list/tuple of frames
      - torch.Tensor (T,B,C,H,W) or (T,C,H,W) or (B,C,H,W)
    """
    frames: List[object] = []
    if isinstance(history_like, (list, tuple)):
        for item in history_like:
            frames.extend(_flatten_history(item, n_cell_types))
        return frames

    if isinstance(history_like, torch.Tensor):
        t = history_like
        if t.ndim == 5:  # (T,B,C,H,W)
            return [_as_label_grid(t[i], n_cell_types) for i in range(t.shape[0])]
        if t.ndim == 4:
            # Heuristic: if first dim looks like time (T,C,H,W)
            if t.shape[0] > 1 and t.shape[1] >= n_cell_types:
                return [_as_label_grid(t[i].unsqueeze(0), n_cell_types) for i in range(t.shape[0])]
            return [_as_label_grid(t, n_cell_types)]
        if t.ndim == 3:
            return [_as_label_grid(t.unsqueeze(0), n_cell_types)]

    if isinstance(history_like, np.ndarray):
        arr = history_like
        if arr.ndim >= 4 and arr.shape[0] > 1:
            return [_as_label_grid(arr[i], n_cell_types) for i in range(arr.shape[0])]
        return [_as_label_grid(arr, n_cell_types)]

    raise TypeError(f"Unsupported history type: {type(history_like)}")


def _save_montage(
    frames: List[np.ndarray],
    times: List[int],
    out_path: Path,
    title: str,
    n_cell_types: int,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Color map consistent with existing videos
    colors = plt.cm.tab10(np.linspace(0, 1, n_cell_types))
    cmap = plt.cm.colors.ListedColormap(colors)

    fig, axes = plt.subplots(1, len(times), figsize=(12, 2.8), dpi=160)
    fig.suptitle(title, fontsize=10)

    for ax, t in zip(axes, times):
        idx = min(max(int(t), 0), len(frames) - 1)
        ax.imshow(frames[idx], cmap=cmap, vmin=0, vmax=n_cell_types - 1)
        ax.set_title(f"t={idx}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    histories_path = repo_root / "histories_300_100.npy"
    base_dir = repo_root / "experiments" / "results_extended" / "tissue_simulation_nb_sweep_300x100"

    sample_idx = int(os.environ.get("SAMPLE_IDX", "149"))
    n_steps = int(os.environ.get("N_STEPS", "100"))
    device = os.environ.get("DEVICE", "cpu")

    # Time points for montage (must exist in both GT and model rollout)
    times = [0, 10, 25, 50, 99]

    print(f"Loading histories: {histories_path}")
    histories = np.load(histories_path, allow_pickle=True)
    n_cell_types = len(ComplexCellType)

    gt_history = histories[sample_idx]
    # Ensure list of label grids
    if isinstance(gt_history, np.ndarray) and gt_history.ndim == 3:
        gt_frames = [gt_history[t] for t in range(gt_history.shape[0])]
    else:
        gt_frames = list(gt_history)

    out_dir = repo_root / "presentation" / "assets" / "same_init" / f"sample_{sample_idx}"
    _save_montage(
        frames=gt_frames,
        times=times,
        out_path=out_dir / "ground_truth.png",
        title="Ground truth (simulator) — multiple time steps",
        n_cell_types=n_cell_types,
    )

    # Initial state for all models (same x0)
    x0_grid = gt_frames[0]
    x0 = grid_to_channels_batch([x0_grid], n_cell_types, device)

    update_net_fn = _make_update_net_fn(device)

    # Model configs and checkpoints
    configs: List[Tuple[str, str, object]] = [
        ("mixture", "mixture_nca_300x100_TL35.pt", ExtendedMixtureNCA),
        ("stochastic", "stochastic_mix_nca_300x100_TL35.pt", ExtendedMixtureNCANoise),
    ]

    for nb in [3, 4, 5, 6, 7]:
        exp_dir = base_dir / f"NB_{nb}"
        for model_key, ckpt_name, cls in configs:
            ckpt_path = exp_dir / ckpt_name
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            print(f"Loading {model_key} NB={nb}: {ckpt_path}")
            model = cls(
                update_nets=update_net_fn,
                hidden_dim=128,
                maintain_seed=False,
                use_alive_mask=False,
                state_dim=6,
                num_rules=5,
                residual=False,
                temperature=3,
                neighborhood_size=nb,
                device=device,
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                result = model(x0, n_steps, return_history=True)
                history_like = result[1] if isinstance(result, tuple) and len(result) > 1 else result
                model_frames = _flatten_history(history_like, n_cell_types)

            out_path = out_dir / f"{model_key}_nb_{nb}.png"
            _save_montage(
                frames=model_frames,
                times=times,
                out_path=out_path,
                title=f"{model_key.title()} MNCA — NB={nb} — rollout from same x0",
                n_cell_types=n_cell_types,
            )

    print(f"Saved comparison montages to: {out_dir}")


if __name__ == "__main__":
    main()

