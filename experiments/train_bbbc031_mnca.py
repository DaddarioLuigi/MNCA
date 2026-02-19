"""
Training minimale di un Mixture NCA su BBBC031 (ricostruzione di una singola immagine CELLMASK),
allineato allo stile del notebook `experiment_microscopy` del paper originale.

Obiettivo: avere un modello di esempio `model_mix_microscopy.pth` da usare poi in
`bbbc031_mnca_demo.py` e per una figura nella tesi.

Uso tipico:
  python experiments/train_bbbc031_mnca.py \
      --dataset_dir "/Users/luigidaddario/Downloads/BBBC031_v1_dataset" \
      --csv_path "/Users/luigidaddario/Downloads/BBBC031_v1_DatasetGroundTruth (1).csv" \
      --checkpoint_path models/model_mix_microscopy.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.utils_images import standard_update_net, train_nca


ORIGINAL_SIZE = 950
N_CHANNELS = 24
N_RULES = 5
HIDDEN_DIM = 128
ALIVE_CHANNEL = 3


def load_microscopy_image(path: Path, target_size: int, padding: int = 0) -> torch.Tensor:
    """Carica immagine CELLMASK come tensore (1, 4, H, W) in [0,1], con resize e padding."""
    img = Image.open(path).convert("RGBA")
    if target_size is not None:
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
    if padding > 0:
        img_tensor = torch.nn.functional.pad(
            img_tensor, (padding, padding, padding, padding), mode="constant", value=0
        )
    return img_tensor.float()


def get_seed_locations_from_csv(
    df: pd.DataFrame, image_name: str, target_size: int, original_size: int = ORIGINAL_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restituisce (seed_y, seed_x) in coordinate griglia [0, target_size-1].
    In PyTorch (B,C,H,W): H = riga = y, W = colonna = x.
    """
    rows = df[df["ImageName"] == image_name]
    if rows.empty:
        raise ValueError(f"No rows for ImageName={image_name}")
    scale = target_size / original_size
    x = np.round(rows["LocationX"].values * scale).astype(int).clip(0, target_size - 1)
    y = np.round(rows["LocationY"].values * scale).astype(int).clip(0, target_size - 1)
    return y, x


def rgba_to_display(t: torch.Tensor) -> np.ndarray:
    """(1,4,H,W) o (4,H,W) -> (H,W,3) per imshow (composite RGB con alpha)."""
    if t.dim() == 4:
        t = t[0]
    rgb = t[:3].permute(1, 2, 0).cpu().numpy()
    a = t[3:4].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    out = (1.0 - a + rgb * a).clip(0, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train semplice di Mixture NCA su una immagine BBBC031 (CELLMASK)."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/Users/luigidaddario/Downloads/BBBC031_v1_dataset"),
        help="Cartella BBBC031_v1_dataset (contiene Images/).",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("/Users/luigidaddario/Downloads/BBBC031_v1_DatasetGroundTruth (1).csv"),
        help="Path al CSV di ground truth BBBC031.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=REPO_ROOT / "models" / "model_mix_microscopy.pth",
        help="Dove salvare il modello addestrato.",
    )
    parser.add_argument(
        "--example_image",
        type=str,
        default="ProcessPlateSparse_wA03_s06_z1_t1",
        help="Nome immagine (senza _CELLMASK.png) da usare come target.",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=96,
        help="Risoluzione quadrata a cui ridimensionare la maschera (come nel paper).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo (cuda o cpu).",
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=4000,
        help="Numero di passi di ottimizzazione (train_nca).",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=10,
        help="Minimo di passi di automa per episodio (num_steps[0]).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20,
        help="Massimo di passi di automa per episodio (num_steps[1]).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per train_nca (pool training).",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=512,
        help="Dimensione del pool di stati CA.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate Adam.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.3,
        help="Fattore gamma per MultiStepLR.",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="*",
        default=(1500, 3000),
        help="Milestones per MultiStepLR.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=200,
        help="Frequenza di logging di train_nca.",
    )
    parser.add_argument(
        "--neighborhood_size",
        type=int,
        default=5,
        help="Dimensione vicinato (Extended MNCA). Es. 3 o 7 per confronti.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Usa Stochastic Mixture (ExtendedMixtureNCANoise) invece di ExtendedMixtureNCA.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    images_dir = args.dataset_dir / "Images"
    img_path = images_dir / f"{args.example_image}_CELLMASK.png"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Carica target e metadati
    target = load_microscopy_image(img_path, args.target_size, padding=0).to(device)
    df = pd.read_csv(args.csv_path, sep=";")
    seed_y, seed_x = get_seed_locations_from_csv(
        df, args.example_image, args.target_size
    )
    seed_loc = (seed_y, seed_x)

    # Modello: Extended (Mixture o Stochastic Mixture) con neighborhood_size
    if args.stochastic:
        model_mix = ExtendedMixtureNCANoise(
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
            neighborhood_size=args.neighborhood_size,
        )
        model_label = "Stochastic Mixture"
    else:
        model_mix = ExtendedMixtureNCA(
            update_nets=standard_update_net,
            num_rules=N_RULES,
            state_dim=N_CHANNELS,
            hidden_dim=HIDDEN_DIM,
            dropout=0.0,
            temperature=1.0,
            device=device,
            use_alive_mask=True,
            alive_threshold=0.1,
            alive_channel=ALIVE_CHANNEL,
            maintain_seed=True,
            residual=True,
            grid_type="square",
            modality="image",
            filter_type="sobel",
            seed_value=1.0,
            neighborhood_size=args.neighborhood_size,
        )
        model_label = "Extended Mixture"

    print("Config:", model_label, ", neighborhood_size =", args.neighborhood_size)
    print("Inizio training su immagine:", args.example_image, f"({args.total_steps} step)")
    # total_steps+1 perché train_nca fa range(total_steps+1) iterazioni
    pbar = tqdm(
        total=args.total_steps + 1,
        unit=" step",
        desc="Training",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] loss={postfix}",
        dynamic_ncols=True,
        mininterval=0.2,
    )

    def on_progress(step: int, total_steps: int, loss: float) -> None:
        pbar.update(1)
        pbar.set_postfix_str(f"{loss:.2e}")

    results = train_nca(
        model_mix,
        target,
        device=device,
        num_steps=(args.min_steps, args.max_steps),
        learning_rate=args.learning_rate,
        decay=0.0,
        milestones=list(args.milestones),
        gamma=args.gamma,
        batch_size=args.batch_size,
        state_dim=N_CHANNELS,
        seed_loc=seed_loc,
        pool_size=args.pool_size,
        total_steps=args.total_steps,
        print_every=args.print_every,
        return_history=False,
        temperature=1.0,
        min_temperature=1.0,
        anneal_rate=0.0,
        straight_through=True,
        init_black=False,
        progress_callback=on_progress,
    )
    pbar.close()

    trained_model = results.get("final_model", model_mix.cpu())

    # Salva pesi
    torch.save(trained_model.state_dict(), args.checkpoint_path)
    print("Modello salvato in", args.checkpoint_path)

    # Figura di confronto GT vs predizione finale (per la tesi)
    trained_model = trained_model.to(device)
    trained_model.eval()
    with torch.no_grad():
        init_state = torch.zeros(
            1, N_CHANNELS, args.target_size, args.target_size, device=device
        )
        init_state[0, ALIVE_CHANNEL:, seed_y, seed_x] = 1.0
        out = trained_model(
            init_state,
            args.max_steps,
            seed_loc=seed_loc,
            return_history=False,
            sample_non_differentiable=True,
            straight_through=True,
        )
        pred_rgba = out[:, :4].clamp(0, 1)

    target_np = rgba_to_display(target)
    pred_np = rgba_to_display(pred_rgba)

    figs_dir = REPO_ROOT / "thesis-latex" / "figs" / "bbbc031"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(target_np)
    axes[0].set_title("Ground truth (CELLMASK)")
    axes[0].axis("off")
    axes[1].imshow(pred_np)
    axes[1].set_title(f"MNCA (dopo {args.max_steps} step)")
    axes[1].axis("off")
    suffix = f"_NB{args.neighborhood_size}" + ("_stochastic" if args.stochastic else "")
    fig.suptitle(
        f"BBBC031: GT vs {model_label} NB={args.neighborhood_size} ({args.example_image})",
        fontsize=11,
    )
    fig.tight_layout()
    out_fig = figs_dir / f"bbbc031_gt_vs_mnca_trained_here{suffix}.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figura di confronto salvata in", out_fig)


if __name__ == "__main__":
    main()

