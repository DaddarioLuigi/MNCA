"""
Esplorazione dataset BBBC031 per la tesi.
Genera statistiche, grafici e figure da inserire nel capitolo risultati.

Uso:
  python experiments/explore_bbbc031.py --dataset_dir /path/to/BBBC031_v1_dataset \\
       --csv_path "/path/to/BBBC031_v1_DatasetGroundTruth (1).csv" --out_dir thesis-latex/figs/bbbc031
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Root del repo per import
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_metadata(csv_path: Path, sep: str = ";") -> pd.DataFrame:
    """Carica il CSV di ground truth BBBC031."""
    df = pd.read_csv(csv_path, sep=sep)
    return df


def get_cells_per_image(df: pd.DataFrame) -> pd.Series:
    """Conteggio celle per ImageName."""
    return df.groupby("ImageName").size()


def plot_cells_per_image_histogram(counts: pd.Series, out_path: Path) -> None:
    """Istogramma numero di celle per immagine."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(counts.values, bins=min(50, len(counts.unique())), color="steelblue", edgecolor="white")
    ax.set_xlabel("Celle per immagine")
    ax.set_ylabel("Numero di immagini")
    ax.set_title("Distribuzione del numero di celle per immagine (BBBC031)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_process_id_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Distribuzione ProcessID (tipo di processo/cella)."""
    fig, ax = plt.subplots(figsize=(5, 3))
    pid_counts = df["ProcessID"].value_counts().sort_index()
    ax.bar(pid_counts.index.astype(str), pid_counts.values, color="teal", edgecolor="white")
    ax.set_xlabel("ProcessID")
    ax.set_ylabel("Numero di celle")
    ax.set_title("Distribuzione ProcessID (BBBC031)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_example_image_with_centroids(
    images_dir: Path,
    df: pd.DataFrame,
    image_name: str,
    out_path: Path,
    scale_to_size: int | None = 256,
    original_size: int = 950,
) -> None:
    """
    Carica un'immagine CELLMASK e sovrappone i centroidi dal CSV.
    image_name: nome senza suffisso _CELLMASK.png (es. ProcessPlateSparse_wA03_s06_z1_t1).
    """
    fname = f"{image_name}_CELLMASK.png"
    img_path = images_dir / fname
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)

    rows = df[df["ImageName"] == image_name]
    if rows.empty:
        raise ValueError(f"No rows for ImageName={image_name}")

    h_orig, w_orig = img_arr.shape[:2]
    if scale_to_size and (w_orig != scale_to_size or h_orig != scale_to_size):
        img_small = Image.fromarray(img_arr).resize((scale_to_size, scale_to_size), Image.Resampling.LANCZOS)
        img_arr = np.array(img_small)
        scale = scale_to_size / original_size
        x = (rows["LocationX"].values * scale).astype(int)
        y = (rows["LocationY"].values * scale).astype(int)
    else:
        scale = 1.0 if w_orig == original_size else w_orig / original_size
        x = (rows["LocationX"].values * scale).astype(int)
        y = (rows["LocationY"].values * scale).astype(int)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_arr)
    ax.scatter(x, y, c="red", s=8, alpha=0.9, label="Centroidi (ground truth)")
    ax.set_title(f"Esempio BBBC031: {image_name}")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_stats_table(counts: pd.Series, df: pd.DataFrame, out_path: Path) -> None:
    """Salva statistiche in formato testo e (opzionale) righe per LaTeX."""
    n_images = len(counts)
    n_cells = int(counts.sum())
    mean_c = float(counts.mean())
    median_c = float(counts.median())
    min_c = int(counts.min())
    max_c = int(counts.max())

    lines = [
        "n_images: " + str(n_images),
        "n_cells: " + str(n_cells),
        "cells_per_image_mean: " + str(round(mean_c, 2)),
        "cells_per_image_median: " + str(round(median_c, 2)),
        "cells_per_image_min: " + str(min_c),
        "cells_per_image_max: " + str(max_c),
    ]
    out_path.write_text("\n".join(lines) + "\n")

    # Tabella LaTeX
    tex_path = out_path.with_suffix(".tex")
    tex = r"""\begin{table}[ht]
\centering
\caption{Statistiche dataset BBBC031.}
\label{tab:bbbc031_stats}
\begin{tabular}{lc}
\toprule
Metrica & Valore \\
\midrule
Numero immagini & %d \\
Numero celle & %d \\
Celle per immagine (media) & %.2f \\
Celle per immagine (mediana) & %.1f \\
Celle per immagine (min--max) & %d--%d \\
\bottomrule
\end{tabular}
\end{table}
""" % (n_images, n_cells, mean_c, median_c, min_c, max_c)
    tex_path.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Esplorazione dataset BBBC031 per la tesi.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("/Users/luigidaddario/Downloads/BBBC031_v1_dataset"),
        help="Cartella BBBC031_v1_dataset (contiene Images/ e Masks/)",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=Path("/Users/luigidaddario/Downloads/BBBC031_v1_DatasetGroundTruth (1).csv"),
        help="Path al CSV ground truth",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "thesis-latex" / "figs" / "bbbc031",
        help="Cartella di output per figure e statistiche",
    )
    parser.add_argument(
        "--example_image",
        type=str,
        default="ProcessPlateSparse_wA03_s06_z1_t1",
        help="Nome immagine di esempio (senza _CELLMASK.png)",
    )
    args = parser.parse_args()

    images_dir = args.dataset_dir / "Images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.csv_path)
    counts = get_cells_per_image(df)

    # Statistiche
    save_stats_table(counts, df, args.out_dir / "bbbc031_stats.txt")
    print("Statistiche salvate in", args.out_dir / "bbbc031_stats.txt")

    # Istogramma celle per immagine
    plot_cells_per_image_histogram(counts, args.out_dir / "bbbc031_cells_per_image.png")
    print("Figura salvata: bbbc031_cells_per_image.png")

    # Distribuzione ProcessID
    plot_process_id_distribution(df, args.out_dir / "bbbc031_process_id.png")
    print("Figura salvata: bbbc031_process_id.png")

    # Esempio immagine con centroidi
    try:
        plot_example_image_with_centroids(
            images_dir, df, args.example_image, args.out_dir / "bbbc031_example_centroids.png"
        )
        print("Figura salvata: bbbc031_example_centroids.png")
    except Exception as e:
        print("Attenzione: esempio immagine non generato:", e)

    print("Fatto. Output in", args.out_dir)


if __name__ == "__main__":
    main()
