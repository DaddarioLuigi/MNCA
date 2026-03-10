"""
Generate comparison figures: NB=7 original vs NB=7 re-trained (BBBC031).

For each image, saves one figure with three panels: Ground truth | Prediction
(original NB=7) | Prediction (re-trained NB=7). Uses checkpoints from
output_dir/models/ (original) and output_dir/models/nb7_retrain/ (retrain).

Usage (from repo root, with venv active):
  python experiments/generate_bbbc031_nb7_comparison.py \\
      --dataset_dir /path/to/BBBC031_v1_dataset \\
      --csv_path /path/to/BBBC031_v1_DatasetGroundTruth.csv \\
      --output_dir results_extended/MNCA_bbbc031_outputs
  # Optional: --out_dir thesis-latex/figs/bbbc031
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def run_inference(model, init_state, seed_loc, num_steps, device):
    with torch.no_grad():
        out = model(
            init_state,
            num_steps,
            seed_loc=seed_loc,
            return_history=False,
            sample_non_differentiable=True,
            straight_through=True,
        )
    return out[:, :4].clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NB=7 original vs retrain comparison figures."
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "results_extended" / "MNCA_bbbc031_outputs",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    models_dir = args.output_dir / "models"
    retrain_dir = models_dir / "nb7_retrain"
    if not retrain_dir.exists():
        print(f"Retrain folder not found: {retrain_dir}")
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

    for img_idx in range(5):
        image_name = example_images[img_idx]
        img_path = images_dir / f"{image_name}_CELLMASK.png"
        if not img_path.exists():
            print(f"Skip img{img_idx}: {img_path} not found.")
            continue
        ckpt_orig = models_dir / f"bbbc031_stochastic_NB7_img{img_idx}.pth"
        ckpt_retrain = retrain_dir / f"bbbc031_stochastic_NB7_img{img_idx}.pth"
        if not ckpt_orig.exists():
            print(f"Skip img{img_idx}: original checkpoint not found.")
            continue
        if not ckpt_retrain.exists():
            print(f"Skip img{img_idx}: retrain checkpoint not found.")
            continue

        target = load_microscopy_image(img_path, TARGET_SIZE, padding=0).to(device)
        seed_y, seed_x = get_seed_locations_from_csv(df, image_name, TARGET_SIZE)
        init_state = build_init_state(
            device, N_CHANNELS, TARGET_SIZE, TARGET_SIZE, seed_y, seed_x
        )
        seed_loc = (seed_y, seed_x)

        # Original NB=7
        model.load_state_dict(
            torch.load(ckpt_orig, map_location=device, weights_only=False)
        )
        model.eval()
        pred_orig = run_inference(
            model, init_state, seed_loc, args.num_steps, device
        )

        # Retrained NB=7
        model.load_state_dict(
            torch.load(ckpt_retrain, map_location=device, weights_only=False)
        )
        model.eval()
        pred_retrain = run_inference(
            model, init_state, seed_loc, args.num_steps, device
        )

        gt_np = rgba_to_display(target)
        orig_np = rgba_to_display(pred_orig)
        retrain_np = rgba_to_display(pred_retrain)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gt_np)
        axes[0].set_title("Ground truth", fontsize=11)
        axes[0].axis("off")
        axes[1].imshow(orig_np)
        axes[1].set_title("NB=7 original (4000 steps)", fontsize=11)
        axes[1].axis("off")
        axes[2].imshow(retrain_np)
        axes[2].set_title("NB=7 re-trained (6000 steps)", fontsize=11)
        axes[2].axis("off")
        fig.suptitle(f"BBBC031 img{img_idx}: NB=7 original vs re-trained", fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"bbbc031_NB7_original_vs_retrain_img{img_idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)

    print("Done.")


if __name__ == "__main__":
    main()
