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


def compute_mse_rgb(gt: torch.Tensor, pred: torch.Tensor) -> float:
    """MSE on the first 3 (RGB) channels only. Useful to separate colour fidelity from alpha."""
    return float(((gt[:, :3] - pred[:, :3]) ** 2).mean().item())


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
    parser.add_argument(
        "--nb7_retrain_dir",
        type=Path,
        default=None,
        help="If set, use NB=7 checkpoints from this dir (e.g. output_dir/models/nb7_retrain).",
    )
    parser.add_argument(
        "--nb7_retrain_12k_dir",
        type=Path,
        default=None,
        help="If set, compute metrics for NB=7 12k run and write 6k vs 12k comparison table.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    models_dir = args.output_dir / "models"
    if not models_dir.exists():
        print(f"Models folder not found: {models_dir}")
        sys.exit(1)
    nb7_retrain_dir = args.nb7_retrain_dir or (models_dir / "nb7_retrain")
    if not nb7_retrain_dir.exists():
        nb7_retrain_dir = None
    nb7_retrain_12k_dir = args.nb7_retrain_12k_dir or (models_dir / "nb7_retrain_12k")
    if not nb7_retrain_12k_dir.exists():
        nb7_retrain_12k_dir = None

    example_images = get_example_images(args.dataset_dir, args.csv_path)
    if len(example_images) < 5:
        print(f"At least 5 images with CELLMASK required; found {len(example_images)}.")
        sys.exit(1)
    print("Images:", example_images)

    df_gt = pd.read_csv(args.csv_path, sep=";")
    images_dir = args.dataset_dir / "Images"
    # NB=3 from models_dir; NB=7 from nb7_retrain_dir if present, else models_dir
    all_ckpts = list(models_dir.glob("bbbc031_stochastic_NB*_img*.pth"))
    checkpoints = []
    for ckpt in sorted(all_ckpts):
        parsed = parse_checkpoint_name(ckpt.name)
        if parsed and parsed[0] == 7 and nb7_retrain_dir is not None:
            alt = nb7_retrain_dir / ckpt.name
            if alt.exists():
                checkpoints.append((alt, "retrain"))
                continue
        checkpoints.append((ckpt, "original"))

    rows = []
    for ckpt, _ in checkpoints:
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
        f.write("\\caption{BBBC031 segmentation metrics: IoU, Dice and MSE (RGBA) per image and neighborhood size (NB). NB=7 results are from the re-trained run (learning rate $3\\times10^{-4}$, 6000 steps).}\n")
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

    # If we have retrain dir, compute NB=7 original vs retrain metrics and write comparison table
    if nb7_retrain_dir is not None:
        out_tex_comp = out_tex.parent / "bbbc031_metrics_NB7_comparison.tex"
        comp_rows = []
        model_nb7 = ExtendedMixtureNCANoise(
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
        for img_idx in range(min(5, len(example_images))):
            image_name = example_images[img_idx]
            img_path = images_dir / f"{image_name}_CELLMASK.png"
            if not img_path.exists():
                continue
            ckpt_orig = models_dir / f"bbbc031_stochastic_NB7_img{img_idx}.pth"
            ckpt_retrain = nb7_retrain_dir / f"bbbc031_stochastic_NB7_img{img_idx}.pth"
            if not ckpt_orig.exists() or not ckpt_retrain.exists():
                continue
            target = load_microscopy_image(img_path, args.target_size, padding=0).to(device)
            seed_y, seed_x = get_seed_locations_from_csv(df_gt, image_name, args.target_size)
            init_state = build_init_state(
                device, N_CHANNELS, args.target_size, args.target_size, seed_y, seed_x
            )
            seed_loc = (seed_y, seed_x)

            for ckpt, label in [(ckpt_orig, "orig"), (ckpt_retrain, "retrain")]:
                model_nb7.load_state_dict(
                    torch.load(ckpt, map_location=device, weights_only=False)
                )
                model_nb7.eval()
                with torch.no_grad():
                    out = model_nb7(
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
                mse_rgb = compute_mse_rgb(target, pred_rgba)
                comp_rows.append({
                    "img_idx": img_idx,
                    "run": label,
                    "IoU": iou,
                    "Dice": dice,
                    "MSE_RGBA": mse,
                    "MSE_RGB": mse_rgb,
                })
        if comp_rows:
            df_comp = pd.DataFrame(comp_rows)
            pivot_iou_c = df_comp.pivot_table(
                index="img_idx", columns="run", values="IoU", aggfunc="first"
            )
            pivot_dice_c = df_comp.pivot_table(
                index="img_idx", columns="run", values="Dice", aggfunc="first"
            )
            pivot_mse_c = df_comp.pivot_table(
                index="img_idx", columns="run", values="MSE_RGBA", aggfunc="first"
            )
            pivot_mse_rgb_c = df_comp.pivot_table(
                index="img_idx", columns="run", values="MSE_RGB", aggfunc="first"
            )
            with open(out_tex_comp, "w", encoding="utf-8") as f:
                f.write("% NB=7 original vs re-trained metrics (auto-generated)\n")
                f.write("% IoU/Dice on alpha mask; MSE on RGBA; MSE_RGB on RGB only (colour fidelity).\n")
                f.write("% Requires \\usepackage{booktabs} in the preamble.\n")
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{BBBC031: comparison between NB=7 original (4000 steps, LR $10^{-3}$) and re-trained (6000 steps, LR $3\\times10^{-4}$). IoU and Dice use the alpha channel; MSE is on all RGBA channels; MSE\\_RGB is on RGB only (colour fidelity).}\n")
                f.write("\\label{tab:bbbc031_NB7_comparison}\n")
                f.write("\\begin{tabular}{l|cccc|cccc|cccc}\n")
                f.write("\\toprule\n")
                f.write("Image & \\multicolumn{4}{c}{NB=7 original} & \\multicolumn{4}{c}{NB=7 re-trained} & \\multicolumn{4}{c}{$\\Delta$ (retrain $-$ orig)} \\\\\n")
                f.write("(idx) & IoU & Dice & MSE & MSE\\_RGB & IoU & Dice & MSE & MSE\\_RGB & IoU & Dice & MSE & MSE\\_RGB \\\\\n")
                f.write("\\midrule\n")
                for img_idx in sorted(pivot_iou_c.index):
                    iou_o = pivot_iou_c.loc[img_idx, "orig"] if "orig" in pivot_iou_c.columns else float("nan")
                    iou_r = pivot_iou_c.loc[img_idx, "retrain"] if "retrain" in pivot_iou_c.columns else float("nan")
                    d_o = pivot_dice_c.loc[img_idx, "orig"] if "orig" in pivot_dice_c.columns else float("nan")
                    d_r = pivot_dice_c.loc[img_idx, "retrain"] if "retrain" in pivot_dice_c.columns else float("nan")
                    m_o = pivot_mse_c.loc[img_idx, "orig"] if "orig" in pivot_mse_c.columns else float("nan")
                    m_r = pivot_mse_c.loc[img_idx, "retrain"] if "retrain" in pivot_mse_c.columns else float("nan")
                    mrgb_o = pivot_mse_rgb_c.loc[img_idx, "orig"] if "orig" in pivot_mse_rgb_c.columns else float("nan")
                    mrgb_r = pivot_mse_rgb_c.loc[img_idx, "retrain"] if "retrain" in pivot_mse_rgb_c.columns else float("nan")
                    diou = iou_r - iou_o if not (np.isnan(iou_o) or np.isnan(iou_r)) else float("nan")
                    ddice = d_r - d_o if not (np.isnan(d_o) or np.isnan(d_r)) else float("nan")
                    dmse = m_r - m_o if not (np.isnan(m_o) or np.isnan(m_r)) else float("nan")
                    dmse_rgb = mrgb_r - mrgb_o if not (np.isnan(mrgb_o) or np.isnan(mrgb_r)) else float("nan")
                    delta_iou = f"{diou:+.3f}" if not np.isnan(diou) else "---"
                    delta_d = f"{ddice:+.3f}" if not np.isnan(ddice) else "---"
                    delta_m = f"{dmse:+.4e}" if not np.isnan(dmse) else "---"
                    delta_mrgb = f"{dmse_rgb:+.4e}" if not np.isnan(dmse_rgb) else "---"
                    f.write(
                        f"{img_idx} & {iou_o:.3f} & {d_o:.3f} & {m_o:.4e} & {mrgb_o:.4e} & {iou_r:.3f} & {d_r:.3f} & {m_r:.4e} & {mrgb_r:.4e} & {delta_iou} & {delta_d} & {delta_m} & {delta_mrgb} \\\\\n"
                    )
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            print(f"NB=7 comparison LaTeX table saved: {out_tex_comp}")

    # If we have both 6k and 12k retrain dirs, compute 6k vs 12k metrics table
    if nb7_retrain_dir is not None and nb7_retrain_12k_dir is not None:
        out_tex_6k12k = out_tex.parent / "bbbc031_metrics_NB7_6k_vs_12k.tex"
        rows_6k12k = []
        model_nb7 = ExtendedMixtureNCANoise(
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
        for img_idx in range(min(5, len(example_images))):
            image_name = example_images[img_idx]
            img_path = images_dir / f"{image_name}_CELLMASK.png"
            if not img_path.exists():
                continue
            ckpt_6k = nb7_retrain_dir / f"bbbc031_stochastic_NB7_img{img_idx}.pth"
            ckpt_12k = nb7_retrain_12k_dir / f"bbbc031_stochastic_NB7_img{img_idx}.pth"
            if not ckpt_6k.exists() or not ckpt_12k.exists():
                continue
            target = load_microscopy_image(img_path, args.target_size, padding=0).to(device)
            seed_y, seed_x = get_seed_locations_from_csv(df_gt, image_name, args.target_size)
            init_state = build_init_state(
                device, N_CHANNELS, args.target_size, args.target_size, seed_y, seed_x
            )
            seed_loc = (seed_y, seed_x)
            for ckpt, label in [(ckpt_6k, "6k"), (ckpt_12k, "12k")]:
                model_nb7.load_state_dict(
                    torch.load(ckpt, map_location=device, weights_only=False)
                )
                model_nb7.eval()
                with torch.no_grad():
                    out = model_nb7(
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
                rows_6k12k.append({
                    "img_idx": img_idx,
                    "run": label,
                    "IoU": compute_iou(gt_mask, pred_mask),
                    "Dice": compute_dice(gt_mask, pred_mask),
                    "MSE_RGBA": compute_mse_rgba(target, pred_rgba),
                })
        if rows_6k12k:
            df_6k12k = pd.DataFrame(rows_6k12k)
            pivot_iou_ = df_6k12k.pivot_table(
                index="img_idx", columns="run", values="IoU", aggfunc="first"
            )
            pivot_dice_ = df_6k12k.pivot_table(
                index="img_idx", columns="run", values="Dice", aggfunc="first"
            )
            pivot_mse_ = df_6k12k.pivot_table(
                index="img_idx", columns="run", values="MSE_RGBA", aggfunc="first"
            )
            with open(out_tex_6k12k, "w", encoding="utf-8") as f:
                f.write("% NB=7 6000 vs 12000 steps metrics (auto-generated)\n")
                f.write("% Requires \\usepackage{booktabs} in the preamble.\n")
                f.write("\\begin{table}[ht]\n")
                f.write("\\centering\n")
                f.write("\\caption{BBBC031: segmentation metrics for NB=7 re-trained run at 6000 steps vs extended run at 12000 steps (LR $3\\times10^{-4}$, milestones at 4500, 6000, 7500, 9500).}\n")
                f.write("\\label{tab:bbbc031_NB7_6k_vs_12k}\n")
                f.write("\\begin{tabular}{l|ccc|ccc|ccc}\n")
                f.write("\\toprule\n")
                f.write("Image & \\multicolumn{3}{c}{6k steps} & \\multicolumn{3}{c}{12k steps} & \\multicolumn{3}{c}{$\\Delta$ (12k $-$ 6k)} \\\\\n")
                f.write("(idx) & IoU & Dice & MSE & IoU & Dice & MSE & IoU & Dice & MSE \\\\\n")
                f.write("\\midrule\n")
                for img_idx in sorted(pivot_iou_.index):
                    iou_6 = pivot_iou_.loc[img_idx, "6k"] if "6k" in pivot_iou_.columns else float("nan")
                    iou_12 = pivot_iou_.loc[img_idx, "12k"] if "12k" in pivot_iou_.columns else float("nan")
                    d_6 = pivot_dice_.loc[img_idx, "6k"] if "6k" in pivot_dice_.columns else float("nan")
                    d_12 = pivot_dice_.loc[img_idx, "12k"] if "12k" in pivot_dice_.columns else float("nan")
                    m_6 = pivot_mse_.loc[img_idx, "6k"] if "6k" in pivot_mse_.columns else float("nan")
                    m_12 = pivot_mse_.loc[img_idx, "12k"] if "12k" in pivot_mse_.columns else float("nan")
                    diou = iou_12 - iou_6 if not (np.isnan(iou_6) or np.isnan(iou_12)) else float("nan")
                    ddice = d_12 - d_6 if not (np.isnan(d_6) or np.isnan(d_12)) else float("nan")
                    dmse = m_12 - m_6 if not (np.isnan(m_6) or np.isnan(m_12)) else float("nan")
                    delta_iou = f"{diou:+.3f}" if not np.isnan(diou) else "---"
                    delta_d = f"{ddice:+.3f}" if not np.isnan(ddice) else "---"
                    delta_m = f"{dmse:+.4e}" if not np.isnan(dmse) else "---"
                    f.write(
                        f"{img_idx} & {iou_6:.3f} & {d_6:.3f} & {m_6:.4e} & {iou_12:.3f} & {d_12:.3f} & {m_12:.4e} & {delta_iou} & {delta_d} & {delta_m} \\\\\n"
                    )
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
            print(f"NB=7 6k vs 12k LaTeX table saved: {out_tex_6k12k}")


if __name__ == "__main__":
    main()
