"""
Generate GT vs prediction figures for the NB=7 re-trained checkpoints (BBBC031).

Saves side-by-side Ground truth / MNCA prediction for a few images so the thesis
can show example outputs from the new training. Uses checkpoints from
output_dir/models/nb7_retrain.

Usage (from repo root, with venv active):
  python experiments/generate_bbbc031_retrain_figures.py \\
      --dataset_dir /path/to/BBBC031_v1_dataset \\
      --csv_path /path/to/BBBC031_v1_DatasetGroundTruth.csv \\
      --output_dir results_extended/MNCA_bbbc031_outputs
  # Optional: --indices 0 2 4  (default: 0 2 4)
  # Optional: --out_dir thesis-latex/figs/bbbc031  (default: output_dir/figs/bbbc031)
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load bbc031_mnca_demo from same directory (avoid import path issues)
_experiments = Path(__file__).resolve().parent
_spec_demo = importlib.util.spec_from_file_location(
    "bbbc031_mnca_demo",
    _experiments / "bbbc031_mnca_demo.py",
)
_demo = importlib.util.module_from_spec(_spec_demo)
sys.modules["bbbc031_mnca_demo"] = _demo
_spec_demo.loader.exec_module(_demo)

_spec_metrics = importlib.util.spec_from_file_location(
    "bbbc031_metrics_thesis",
    _experiments / "bbbc031_metrics_thesis.py",
)
_metrics = importlib.util.module_from_spec(_spec_metrics)
sys.modules["bbbc031_metrics_thesis"] = _metrics
_spec_metrics.loader.exec_module(_metrics)

from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.utils_images import standard_update_net

ALIVE_CHANNEL = _demo.ALIVE_CHANNEL
N_CHANNELS = _demo.N_CHANNELS
build_init_state = _demo.build_init_state
get_seed_locations_from_csv = _demo.get_seed_locations_from_csv
load_microscopy_image = _demo.load_microscopy_image
rgba_to_display = _demo.rgba_to_display
get_example_images = _metrics.get_example_images

N_RULES = 5
HIDDEN_DIM = 128
TARGET_SIZE = 96
NUM_STEPS = 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GT vs prediction figures for NB=7 retrain checkpoints."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="BBBC031 folder (contains Images/).",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        required=True,
        help="Ground truth CSV (sep=;).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "results_extended" / "MNCA_bbbc031_outputs",
        help="Output folder (contains models/).",
    )
    parser.add_argument(
        "--checkpoint_subdir",
        type=str,
        default="nb7_retrain",
        help="Subdir under output_dir/models/ (e.g. nb7_retrain or nb7_retrain_12k).",
    )
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="",
        help="Suffix for output filenames (e.g. _12k for bbbc031_gt_vs_mnca_NB7_retrain_12k_img0.png).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 2, 4],
        help="Image indices to plot (default: 0 2 4).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Where to save figures (default: output_dir/figs/bbbc031).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=NUM_STEPS,
        help="NCA simulation steps at inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    retrain_dir = args.output_dir / "models" / args.checkpoint_subdir
    if not retrain_dir.exists():
        print(f"Checkpoint folder not found: {retrain_dir}")
        sys.exit(1)

    out_dir = args.out_dir or (args.output_dir / "figs" / "bbbc031")
    out_dir.mkdir(parents=True, exist_ok=True)

    example_images = get_example_images(args.dataset_dir, args.csv_path)
    if len(example_images) < 5:
        print(f"Need at least 5 images; found {len(example_images)}.")
        sys.exit(1)

    df = pd.read_csv(args.csv_path, sep=";")
    images_dir = args.dataset_dir / "Images"

    model = ExtendedMixtureNCANoise(
        update_nets=standard_update_net,
        num_rules=N_RULES,
        state_dim=N_CHANNELS,
        hidden_dim=HIDDEN_DIM,
        dropout=0.0,
        temperature=1.0,
        device=device,
        num_latent_dims=1,
        use_alive_mask=True,
        alive_threshold=0.1,
        alive_channel=ALIVE_CHANNEL,
        maintain_seed=True,
        residual=True,
        grid_type="square",
        modality="image",
        filter_type="sobel",
        seed_value=1.0,
        neighborhood_size=7,
    )

    for img_idx in args.indices:
        if img_idx >= len(example_images):
            continue
        image_name = example_images[img_idx]
        img_path = images_dir / f"{image_name}_CELLMASK.png"
        if not img_path.exists():
            print(f"Skip img{img_idx}: {img_path} not found.")
            continue
        ckpt_path = retrain_dir / f"bbbc031_stochastic_NB7_img{img_idx}.pth"
        if not ckpt_path.exists():
            print(f"Skip img{img_idx}: {ckpt_path} not found.")
            continue

        target = load_microscopy_image(img_path, TARGET_SIZE, padding=0).to(device)
        seed_y, seed_x = get_seed_locations_from_csv(df, image_name, TARGET_SIZE)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        init_state = build_init_state(
            device, N_CHANNELS, TARGET_SIZE, TARGET_SIZE, seed_y, seed_x
        )
        with torch.no_grad():
            out = model(
                init_state,
                args.num_steps,
                seed_loc=(seed_y, seed_x),
                return_history=False,
                sample_non_differentiable=True,
                straight_through=True,
            )
        pred_rgba = out[:, :4].clamp(0, 1)
        target_np = rgba_to_display(target)
        pred_np = rgba_to_display(pred_rgba)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(target_np)
        axes[0].set_title("Ground truth (CELLMASK)", fontsize=12)
        axes[0].axis("off")
        axes[1].imshow(pred_np)
        axes[1].set_title(f"MNCA NB=7 retrain{args.filename_suffix} (after {args.num_steps} steps)", fontsize=12)
        axes[1].axis("off")
        fig.suptitle(f"BBBC031: img{img_idx} — GT vs re-trained NB=7{args.filename_suffix}", fontsize=13)
        fig.tight_layout()
        out_path = out_dir / f"bbbc031_gt_vs_mnca_NB7_retrain{args.filename_suffix}_img{img_idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)

    print("Done.")


if __name__ == "__main__":
    main()
