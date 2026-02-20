"""
Quantitative metrics for BBBC031 (IoU, Dice, MSE) for use in the thesis.

Runs inference with each checkpoint in output_dir/models, compares prediction
to ground truth (CELLMASK) and computes:
- IoU (Intersection over Union) on binarized mask (alpha channel > 0.5)
- Dice (F1 on mask)
- MSE on the 4 RGBA channels (same as training loss)

Output: CSV with all metrics and LaTeX table for the thesis.

Usage (from repo root):
  python experiments/bbbc031_metrics_thesis.py \\
      --dataset_dir /path/to/BBBC031_v1_dataset \\
      --csv_path /path/to/BBBC031_v1_DatasetGroundTruth.csv \\
      --output_dir results_extended/MNCA_bbbc031_outputs
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bbbc031_mnca_demo import (
    ALIVE_CHANNEL,
    N_CHANNELS,
    ORIGINAL_SIZE,
    build_init_state,
    get_seed_locations_from_csv,
    load_microscopy_image,
)
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.utils_images import standard_update_net

N_RULES = 5
HIDDEN_DIM = 128
THRESHOLD = 0.5  # binarizzazione maschera (alpha)


def compute_iou(gt_mask: np.ndarray, pred_mask: np.ndarray, eps: float = 1e-8) -> float:
    """IoU = |A ∩ B| / |A ∪ B| on binary masks."""
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / (union + eps))


def compute_dice(gt_mask: np.ndarray, pred_mask: np.ndarray, eps: float = 1e-8) -> float:
    """Dice = 2|A ∩ B| / (|A| + |B|)."""
    inter = np.logical_and(gt_mask, pred_mask).sum()
    a_sum = gt_mask.sum()
    b_sum = pred_mask.sum()
    if a_sum == 0 and b_sum == 0:
        return 1.0
    return float(2 * inter / (a_sum + b_sum + eps))


def compute_mse_rgba(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """MSE on the 4 RGBA channels, as in training."""
    return float(((gt - pred) ** 2).mean().item())


def get_mask_from_rgba(rgba: torch.Tensor, channel: int = 3) -> np.ndarray:
    """(1,4,H,W) or (4,H,W) -> binary mask (H,W) with alpha > threshold."""
    if rgba.dim() == 4:
        rgba = rgba[0]
    alpha = rgba[channel].cpu().numpy()
    return (alpha > THRESHOLD).astype(np.float64)


def get_example_images(dataset_dir: Path, csv_path: Path) -> list[str]:
    """First 5 images with CELLMASK file present (same order as Colab notebook)."""
    df = pd.read_csv(csv_path, sep=";")
    images_dir = dataset_dir / "Images"
    all_names = df["ImageName"].unique()
    available = [
        n
        for n in all_names
        if (images_dir / f"{n}_CELLMASK.png").exists()
    ]
    return available[:5]


def parse_checkpoint_name(name: str) -> tuple[int, int] | None:
    """From 'bbbc031_stochastic_NB3_img0.pth' extract (nb=3, img_idx=0)."""
    m = re.match(r"bbbc031_stochastic_NB(\d+)_img(\d+)\.pth", name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute IoU, Dice, MSE for BBBC031 checkpoints and generate thesis table."
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
        help="BBBC031 ground truth CSV (sep=;).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REPO_ROOT / "results_extended" / "MNCA_bbbc031_outputs",
        help="Output folder (contains models/ with .pth).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="NCA simulation steps at inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=96,
        help="Grid resolution (same as training).",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Output CSV path (default: output_dir/analysis/bbbc031_metrics.csv).",
    )
    parser.add_argument(
        "--out_tex",
        type=Path,
        default=None,
        help="LaTeX table path (default: thesis-latex/tables/bbbc031_metrics.tex).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    models_dir = args.output_dir / "models"
    if not models_dir.exists():
        print(f"Models folder not found: {models_dir}")
        sys.exit(1)

    example_images = get_example_images(args.dataset_dir, args.csv_path)
    if len(example_images) < 5:
        print(f"At least 5 images with CELLMASK required; found {len(example_images)}.")
        sys.exit(1)
    print("Images:", example_images)

    df_gt = pd.read_csv(args.csv_path, sep=";")
    images_dir = args.dataset_dir / "Images"
    checkpoints = sorted(models_dir.glob("bbbc031_stochastic_NB*_img*.pth"))

    rows = []
    for ckpt in checkpoints:
        parsed = parse_checkpoint_name(ckpt.name)
        if parsed is None:
            continue
        nb, img_idx = parsed
        if img_idx >= len(example_images):
            continue
        image_name = example_images[img_idx]
        img_path = images_dir / f"{image_name}_CELLMASK.png"
        if not img_path.exists():
            print(f"Skip {ckpt.name}: {img_path} not found.")
            continue

        # Load GT
        target = load_microscopy_image(img_path, args.target_size, padding=0).to(device)
        seed_y, seed_x = get_seed_locations_from_csv(
            df_gt, image_name, args.target_size
        )
        seed_loc = (seed_y, seed_x)

        # Model and inference
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
            neighborhood_size=nb,
        )
        state_dict = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        init_state = build_init_state(
            device, N_CHANNELS, args.target_size, args.target_size, seed_y, seed_x
        )
        with torch.no_grad():
            out = model(
                init_state,
                args.num_steps,
                seed_loc=seed_loc,
                return_history=False,
                sample_non_differentiable=True,
                straight_through=True,
            )
        pred_rgba = out[:, :4].clamp(0, 1)

        gt_mask = get_mask_from_rgba(target)
        pred_mask = get_mask_from_rgba(pred_rgba)
        iou = compute_iou(gt_mask, pred_mask)
        dice = compute_dice(gt_mask, pred_mask)
        mse = compute_mse_rgba(target, pred_rgba)

        rows.append({
            "image_name": image_name,
            "img_idx": img_idx,
            "neighborhood_size": nb,
            "IoU": iou,
            "Dice": dice,
            "MSE_RGBA": mse,
        })
        print(f"  {ckpt.name}: IoU={iou:.4f}, Dice={dice:.4f}, MSE={mse:.4e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics computed.")
        sys.exit(1)

    # Save CSV
    out_csv = args.out_csv or (args.output_dir / "analysis" / "bbbc031_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"CSV saved: {out_csv}")

    # Pivot by (img_idx, nb) and LaTeX table
    out_tex = args.out_tex or (REPO_ROOT / "thesis-latex" / "tables" / "bbbc031_metrics.tex")
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # Table: rows = img_idx, columns = NB3 / NB7 with IoU, Dice, MSE
    pivot_iou = df.pivot_table(
        index="img_idx", columns="neighborhood_size", values="IoU", aggfunc="first"
    )
    pivot_dice = df.pivot_table(
        index="img_idx", columns="neighborhood_size", values="Dice", aggfunc="first"
    )
    pivot_mse = df.pivot_table(
        index="img_idx", columns="neighborhood_size", values="MSE_RGBA", aggfunc="first"
    )

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% BBBC031 MNCA metrics table (auto-generated by bbbc031_metrics_thesis.py)\n")
        f.write("% IoU, Dice, MSE on Stochastic checkpoints NB=3 and NB=7, 20 simulation steps.\n")
        f.write("% Requires \\usepackage{booktabs} in the preamble.\n")
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{BBBC031 segmentation metrics: IoU, Dice and MSE (RGBA) per image and neighborhood size (NB).}\n")
        f.write("\\label{tab:bbbc031_metrics}\n")
        n_cols = 7  # img_idx + NB3 IoU,Dice,MSE + NB7 IoU,Dice,MSE
        f.write("\\begin{tabular}{l|ccc|ccc}\n")
        f.write("\\toprule\n")
        f.write("Image & \\multicolumn{3}{c}{NB=3} & \\multicolumn{3}{c}{NB=7} \\\\\n")
        f.write("(idx) & IoU & Dice & MSE & IoU & Dice & MSE \\\\\n")
        f.write("\\midrule\n")
        for img_idx in sorted(pivot_iou.index):
            iou3 = pivot_iou.loc[img_idx, 3] if 3 in pivot_iou.columns else float("nan")
            iou7 = pivot_iou.loc[img_idx, 7] if 7 in pivot_iou.columns else float("nan")
            d3 = pivot_dice.loc[img_idx, 3] if 3 in pivot_dice.columns else float("nan")
            d7 = pivot_dice.loc[img_idx, 7] if 7 in pivot_dice.columns else float("nan")
            m3 = pivot_mse.loc[img_idx, 3] if 3 in pivot_mse.columns else float("nan")
            m7 = pivot_mse.loc[img_idx, 7] if 7 in pivot_mse.columns else float("nan")
            f.write(
                f"{img_idx} & {iou3:.3f} & {d3:.3f} & {m3:.4e} & {iou7:.3f} & {d7:.3f} & {m7:.4e} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"LaTeX table saved: {out_tex}")


if __name__ == "__main__":
    main()
