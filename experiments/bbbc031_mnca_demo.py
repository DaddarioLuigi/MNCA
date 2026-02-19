"""
Demo MNCA su dataset BBBC031: carica un'immagine CELLMASK, crea seed dalle annotazioni,
(opzionale) carica un modello pre-addestrato e genera predizione; salva figure per la tesi.

Configurazione allineata al notebook experiment_microscopy del paper MNCA:
- state_dim=24, modality image, seed sui centroidi ground truth.
Se non si fornisce un checkpoint, vengono comunque salvate target e seed (setup per la tesi).

Uso:
  # Solo esplorazione (target + seed)
  python experiments/bbbc031_mnca_demo.py --dataset_dir /path/to/BBBC031_v1_dataset \\
       --csv_path "/path/to/BBBC031_v1_DatasetGroundTruth (1).csv" --out_dir thesis-latex/figs/bbbc031

  # Con modello pre-addestrato (target vs predizione)
  python experiments/bbbc031_mnca_demo.py ... --checkpoint path/to/model_mix_microscopy.pth --num_steps 96
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mix_NCA.MixtureNCA import MixtureNCA
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.utils_images import standard_update_net

# Parametri allineati al notebook microscopy (paper MNCA)
ORIGINAL_SIZE = 950  # risoluzione originale BBBC031
N_CHANNELS = 24
N_RULES = 5
HIDDEN_DIM = 128
ALIVE_CHANNEL = 3  # canale "alpha" per alive mask


def load_microscopy_image(path: Path, target_size: int, padding: int = 0) -> torch.Tensor:
    """Carica immagine CELLMASK come tensore (1, 4, H, W) in [0,1], con resize e padding."""
    img = Image.open(path).convert("RGBA")
    if target_size is not None:
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img)) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
    if padding > 0:
        img_tensor = torch.nn.functional.pad(
            img_tensor, (padding,) * 4, mode="constant", value=0
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


def build_init_state(
    device: torch.device,
    state_dim: int,
    height: int,
    width: int,
    seed_y: np.ndarray,
    seed_x: np.ndarray,
) -> torch.Tensor:
    """
    Stato iniziale: zeri con seed attivi sui centroidi (canali 3: = 1).
    """
    init_state = torch.zeros(1, state_dim, height, width, device=device)
    init_state[0, ALIVE_CHANNEL:, seed_y, seed_x] = 1.0
    return init_state


def rgba_to_display(t: torch.Tensor) -> np.ndarray:
    """(1,4,H,W) o (4,H,W) -> (H,W,3) per imshow (composite RGB con alpha)."""
    if t.dim() == 4:
        t = t[0]
    rgb = t[:3].permute(1, 2, 0).cpu().numpy()
    a = t[3:4].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    # Sfondo bianco, premoltiplicato: out = (1-a) + a*rgb semplificato
    out = (1.0 - a + rgb * a).clip(0, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo MNCA su BBBC031 per la tesi.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/Users/luigidaddario/Downloads/BBBC031_v1_dataset"),
        help="Cartella BBBC031 (contiene Images/)",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("/Users/luigidaddario/Downloads/BBBC031_v1_DatasetGroundTruth (1).csv"),
        help="CSV ground truth",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "thesis-latex" / "figs" / "bbbc031",
        help="Cartella output figure",
    )
    parser.add_argument(
        "--example_image",
        type=str,
        default="ProcessPlateSparse_wA03_s06_z1_t1",
        help="Nome immagine (senza _CELLMASK.png)",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=96,
        help="Lato griglia (come nel paper)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path al .pth del modello Mixture NCA microscopy (opzionale)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=96,
        help="Passi di rollout se si usa checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--neighborhood_size",
        type=int,
        default=5,
        help="Dimensione vicinato (es. 3 o 7 per checkpoint Extended/Stochastic).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Carica checkpoint Stochastic Mixture (ExtendedMixtureNCANoise).",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Salva un video dell'evoluzione NCA (richiede checkpoint).",
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        default=None,
        help="Path di output per il video (default: out_dir/bbbc031_evolution_NB{nb}.mp4).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    images_dir = args.dataset_dir / "Images"
    img_path = images_dir / f"{args.example_image}_CELLMASK.png"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Carica target e metadati
    target = load_microscopy_image(img_path, args.target_size, padding=0)
    df = pd.read_csv(args.csv_path, sep=";")
    seed_y, seed_x = get_seed_locations_from_csv(
        df, args.example_image, args.target_size
    )

    # Figura 1: Target + posizioni seed (sempre salvata)
    target_np = rgba_to_display(target)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(target_np)
    ax.scatter(seed_x, seed_y, c="red", s=6, alpha=0.8, label="Seed (centroidi GT)")
    ax.set_title(f"BBBC031: target e seed\n{args.example_image}")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "bbbc031_target_and_seed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Salvato: bbbc031_target_and_seed.png")

    if args.checkpoint is None or not args.checkpoint.exists():
        print("Nessun checkpoint fornito o file non trovato. Salvo solo target+seed.")
        print("Per generare la predizione MNCA, addestra il modello (es. da experiment_microscopy) e passa --checkpoint.")
        return

    # Costruisci modello (Mixture, Extended o Stochastic) e carica pesi
    if args.stochastic:
        model = ExtendedMixtureNCANoise(
            update_nets=standard_update_net,
            num_rules=N_RULES,
            state_dim=N_CHANNELS,
            hidden_dim=HIDDEN_DIM,
            dropout=0,
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
    elif args.neighborhood_size == 3:
        model = MixtureNCA(
            update_nets=standard_update_net,
            num_rules=N_RULES,
            state_dim=N_CHANNELS,
            hidden_dim=HIDDEN_DIM,
            dropout=0,
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
        )
    else:
        model = ExtendedMixtureNCA(
            update_nets=standard_update_net,
            num_rules=N_RULES,
            state_dim=N_CHANNELS,
            hidden_dim=HIDDEN_DIM,
            dropout=0,
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
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Stato iniziale e rollout
    init_state = build_init_state(
        device, N_CHANNELS, args.target_size, args.target_size, seed_y, seed_x
    )
    need_history = args.save_video
    with torch.no_grad():
        out = model(
            init_state,
            args.num_steps,
            seed_loc=(seed_y, seed_x),
            return_history=need_history,
            sample_non_differentiable=True,
            straight_through=True,
        )
    if need_history:
        # out: (T+1, B, C, H, W)
        history = out
        pred_rgba = history[-1, :, :4].clamp(0, 1)
    else:
        # out: (B, C, H, W)
        pred_rgba = out[:, :4].clamp(0, 1)
    pred_np = rgba_to_display(pred_rgba)

    # Figura 2: Target vs Predizione
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(target_np)
    axes[0].set_title("Ground truth (CELLMASK)")
    axes[0].axis("off")
    axes[1].imshow(pred_np)
    axes[1].set_title(f"MNCA (dopo {args.num_steps} step)")
    axes[1].axis("off")
    fig.suptitle(f"BBBC031: confronto GT vs MNCA — {args.example_image}", fontsize=11)
    fig.tight_layout()
    out_name = "bbbc031_gt_vs_mnca.png"
    if args.stochastic:
        out_name = f"bbbc031_gt_vs_mnca_NB{args.neighborhood_size}_stochastic.png"
    fig.savefig(args.out_dir / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Salvato:", out_name)

    if args.save_video and need_history:
        video_path = args.video_path or (
            args.out_dir / f"bbbc031_evolution_NB{args.neighborhood_size}.mp4"
        )
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        frame_images = [
            rgba_to_display(history[t, 0, :4].clamp(0, 1))
            for t in range(history.shape[0])
        ]
        fig_v, ax_v = plt.subplots(figsize=(5, 5))
        ax_v.axis("off")
        im = ax_v.imshow(frame_images[0])
        ax_v.set_title(f"Step 0/{len(frame_images) - 1}")

        def _update(t):
            im.set_array(frame_images[min(t, len(frame_images) - 1)])
            ax_v.set_title(f"Step {min(t, len(frame_images) - 1)}/{len(frame_images) - 1}")
            return [im]

        anim = animation.FuncAnimation(
            fig_v, _update, frames=len(frame_images), interval=80, blit=True, repeat=True
        )
        try:
            anim.save(str(video_path), writer="ffmpeg", fps=12, bitrate=1800)
            print("Video salvato:", video_path)
        except Exception as e:
            gif_path = video_path.with_suffix(".gif")
            try:
                anim.save(str(gif_path), writer="pillow", fps=12)
                print("Video non disponibile (ffmpeg?), salvato GIF:", gif_path)
            except Exception as e2:
                print("Errore salvataggio video/GIF:", e, e2)
        plt.close(fig_v)

    print("Fatto. Output in", args.out_dir)


if __name__ == "__main__":
    main()
