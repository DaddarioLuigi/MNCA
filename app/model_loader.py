"""
Caricamento modelli NCA addestrati e rollout.
Usato dall'app Streamlit per test interattivi.
"""
from pathlib import Path
import sys

import numpy as np
import torch

# Root del repo
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mix_NCA.TissueModel import ComplexCellType
from mix_NCA.utils_simulations import classification_update_net, grid_to_channels_batch
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise


N_CELL_TYPES = len(ComplexCellType)
STATE_DIM = 6
N_RULES = 5
HIDDEN_DIM = 128

# Solo risultati completi (nessun checkpoint di fase: _phase1_TL35.pt ecc.)
MIXTURE_CKPT_NAMES = [
    "mixture_nca_curriculum.pt",
    "mixture_nca.pt",
    "mixture_nca_1000.pt",
]
STOCHASTIC_CKPT_NAMES = [
    "stochastic_mix_nca_curriculum.pt",
    "stochastic_mix_nca.pt",
    "stochastic_mix_nca_1000.pt",
]


def _is_phase_checkpoint(path: Path) -> bool:
    """True se il file è un checkpoint di fase (es. _phase1_TL35.pt), da escludere."""
    return "_phase" in path.name.lower()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_update_net_fn(device):
    def update_net_wrapper(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
        return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)
    return update_net_wrapper


def load_mixture_nca(exp_dir: Path, device, checkpoint_name: str = None):
    """Carica Mixture NCA da cartella esperimento (es. NB_3). Solo checkpoint completi, no fasi."""
    exp_dir = Path(exp_dir)
    for name in (checkpoint_name,) if checkpoint_name else MIXTURE_CKPT_NAMES:
        ckpt = exp_dir / name
        if ckpt.exists() and not _is_phase_checkpoint(ckpt):
            break
    else:
        raise FileNotFoundError(
            f"No complete Mixture checkpoint found in {exp_dir} "
            "(looked for: " + ", ".join(MIXTURE_CKPT_NAMES) + "; phase checkpoints excluded)."
        )
    nb_size = _nb_from_dir(exp_dir)
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
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt


def load_stochastic_nca(exp_dir: Path, device, checkpoint_name: str = None):
    """Carica Stochastic Mixture NCA da cartella esperimento. Solo checkpoint completi, no fasi."""
    exp_dir = Path(exp_dir)
    for name in (checkpoint_name,) if checkpoint_name else STOCHASTIC_CKPT_NAMES:
        ckpt = exp_dir / name
        if ckpt.exists() and not _is_phase_checkpoint(ckpt):
            break
    else:
        raise FileNotFoundError(
            f"No complete Stochastic checkpoint found in {exp_dir} "
            "(looked for: " + ", ".join(STOCHASTIC_CKPT_NAMES) + "; phase checkpoints excluded)."
        )
    nb_size = _nb_from_dir(exp_dir)
    update_net_fn = _make_update_net_fn(device)
    model = ExtendedMixtureNCANoise(
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
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt


def _nb_from_dir(exp_dir: Path) -> int:
    name = exp_dir.name
    if name.startswith("NB_"):
        return int(name.split("_")[1])
    return 3


def grid_to_channels_batch_safe(grids, n_cell_types, device):
    """Wrapper che accetta anche singolo ndarray 2D."""
    if isinstance(grids, np.ndarray) and grids.ndim == 2:
        grids = [grids]
    return grid_to_channels_batch(grids, n_cell_types, device=device)


def run_rollout(model, initial_grid: np.ndarray, n_steps: int, device, stochastic: bool, seed: int = None):
    """
    Esegue rollout del modello a partire da una griglia iniziale.
    initial_grid: (H, W) int, valori 0..n_cell_types-1
    Ritorna: list di (H,W) numpy array (uno per step incluso stato iniziale).
    seed: opzionale; usato solo se stochastic=True per riproducibilità (es. 42 = stesso degli script).
    """
    if stochastic and seed is not None:
        torch.manual_seed(seed)
        try:
            np.random.seed(seed)
        except Exception:
            pass
    x0 = grid_to_channels_batch_safe([initial_grid], N_CELL_TYPES, device)
    with torch.no_grad():
        out = model(x0, n_steps, return_history=True, sample_non_differentiable=stochastic)
    if isinstance(out, tuple):
        frames_t = out[1]  # (x, frames)
    else:
        frames_t = out
    # frames_t: (T, 1, C, H, W) o (T, C, H, W)
    if frames_t.dim() == 5:
        frames_t = frames_t[:, 0]
    # (T, C, H, W) -> lista di (H,W) label
    frames_np = []
    for t in range(frames_t.shape[0]):
        lab = frames_t[t, :N_CELL_TYPES].argmax(dim=0).cpu().numpy()
        frames_np.append(lab)
    return frames_np


def _color_to_rgba(color: str):
    """Converte colore (hex #RRGGBB o nome matplotlib) in (r,g,b,255)."""
    if color.startswith("#") and len(color) >= 7:
        h = color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (r, g, b, 255)
    # Nomi comuni (ComplexCellType EMPTY = 'white')
    names = {"white": (255, 255, 255, 255), "black": (0, 0, 0, 255)}
    return names.get(color.lower(), (255, 255, 255, 255))


def frame_to_rgba(grid: np.ndarray) -> np.ndarray:
    """Converte griglia (H,W) di label in immagine RGBA (H,W,4) per visualizzazione."""
    colors = [_color_to_rgba(ct.get_color()) for ct in ComplexCellType]
    h, w = grid.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for i, rgba in enumerate(colors):
        if i < 6:
            out[grid == i] = rgba
    return out


# Colori RGB (R,G,B) per decodifica canvas -> griglia
_CELL_COLORS_RGB = np.array([_color_to_rgba(ct.get_color())[:3] for ct in ComplexCellType], dtype=np.float32)


def canvas_image_to_grid(canvas_image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Decodifica l'immagine del canvas (H,W,4) in griglia di label (h,w).
    Ogni pixel viene mappato al tipo di cellula con colore RGB più vicino.
    canvas_image viene ridimensionato a target_shape (h, w).
    """
    from PIL import Image
    h, w = target_shape[0], target_shape[1]
    if canvas_image.shape[0] != h or canvas_image.shape[1] != w:
        pil = Image.fromarray(canvas_image.astype(np.uint8))
        pil = pil.resize((w, h), Image.NEAREST)
        canvas_image = np.array(pil)
    rgb = canvas_image[:, :, :3].astype(np.float32)
    # (H,W,3) vs (6,3) -> distanza per pixel
    diff = rgb[:, :, np.newaxis, :] - _CELL_COLORS_RGB[np.newaxis, np.newaxis, :, :]
    dist = (diff ** 2).sum(axis=-1)
    grid = np.argmin(dist, axis=2).astype(np.int64)
    return grid
