"""
MNCA Streamlit App: interactive model test and experiments dashboard.

Run from repo root:
  streamlit run app/main.py
"""
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.model_loader import (
    get_device,
    load_mixture_nca,
    load_stochastic_nca,
    run_rollout,
    frame_to_rgba,
)
from app.dashboard import (
    discover_experiments,
    load_all_metrics,
    plot_metrics_dashboard,
    plot_metrics_single_nb,
)
from mix_NCA.TissueModel import ComplexCellType, create_complex_model_example


# --- Config and style ---
st.set_page_config(
    page_title="MNCA - Test & Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 1.8rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0.5rem; }
    .sub-header { color: #5a7a9e; font-size: 1rem; margin-bottom: 1.5rem; }
    .metric-card { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); padding: 1rem 1.25rem; border-radius: 10px; margin: 0.5rem 0; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    div[data-testid="stVerticalBlock"] > div { padding-top: 0.25rem; }
    /* NCA grids: no interpolation, pixel rendering */
    div[data-testid="stImage"] img, section.main img { image-rendering: pixelated; image-rendering: -moz-crisp-edges; image-rendering: crisp-edges; }
</style>
""", unsafe_allow_html=True)


# Target size for images (NEAREST scaling, no blur)
DISPLAY_IMAGE_SIDE = 720

def init_session_state():
    if "rollout_frames" not in st.session_state:
        st.session_state.rollout_frames = []
    if "rollout_gt_frames" not in st.session_state:
        st.session_state.rollout_gt_frames = []
    if "current_grid" not in st.session_state:
        st.session_state.current_grid = None
    if "model_cache" not in st.session_state:
        st.session_state.model_cache = {}


# Default paths (tissue_simulation_extended with curriculum)
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "results_extended" / "tissue_simulation_extended"
DEFAULT_HISTORIES_PATH = REPO_ROOT / "notebooks" / "histories_300x500.npy"


def load_initial_state_from_histories(hist_path: Path, sim_index: int, time_index: int) -> np.ndarray:
    histories = np.load(hist_path, allow_pickle=True)
    h = histories[sim_index]
    if isinstance(h, np.ndarray) and h.ndim == 3:
        return h[time_index].copy()
    return np.array(h[time_index]).copy()


def load_gt_frames_from_histories(
    hist_path: Path, sim_index: int, time_index: int, n_steps: int
) -> list:
    """Extract ground-truth (simulator) frames from time_index to time_index+n_steps (inclusive)."""
    histories = np.load(hist_path, allow_pickle=True)
    h = histories[sim_index]
    if isinstance(h, np.ndarray) and h.ndim == 3:
        end = min(time_index + n_steps + 1, h.shape[0])
        return [h[t].copy() for t in range(time_index, end)]
    seq = list(h) if hasattr(h, "__iter__") else [h]
    end = min(time_index + n_steps + 1, len(seq))
    return [np.array(seq[t]).copy() for t in range(time_index, end)]


def resize_for_display(img_rgba: np.ndarray, target_side: int = DISPLAY_IMAGE_SIDE) -> np.ndarray:
    """Resize image so longest side = target_side using NEAREST (sharp, no interpolation)."""
    from PIL import Image as PILImage
    h, w = img_rgba.shape[:2]
    longest = max(h, w)
    if longest == 0:
        return img_rgba
    scale = target_side / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = PILImage.fromarray(img_rgba)
    pil = pil.resize((new_w, new_h), PILImage.NEAREST)
    return np.array(pil)


def render_test_model_page():
    init_session_state()
    st.markdown('<p class="main-header">NCA model test</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Load a checkpoint, choose the initial state and click Simulate. '
        'Then use the <strong>Frame</strong> slider to browse steps in real time.</p>',
        unsafe_allow_html=True,
    )

    device = get_device()
    results_base = st.sidebar.text_input(
        "Results folder (curriculum learning)",
        value=str(DEFAULT_RESULTS_DIR),
        help="Same folder used for training: experiments/results_extended/tissue_simulation_extended.",
    )
    results_base = Path(results_base)
    # Only direct NB_* subfolders (e.g. NB_2, NB_3, NB_5) under tissue_simulation_extended
    all_nb_dirs = []
    if results_base.exists():
        for d in results_base.iterdir():
            if d.is_dir() and d.name.startswith("NB_"):
                all_nb_dirs.append((d.name, d))
        all_nb_dirs.sort(key=lambda x: (int(x[0].split("_")[1]) if x[0].startswith("NB_") else 0))

    if not all_nb_dirs:
        st.warning("No NB_* folder found. Check the results path (e.g. experiments/results_extended/tissue_simulation_extended).")
        return

    nb_options = [x[0] for x in all_nb_dirs]
    nb_choice = st.sidebar.selectbox("Experiment (Neighborhood)", nb_options, index=0)
    exp_dir = next(x[1] for x in all_nb_dirs if x[0] == nb_choice)

    model_type = st.sidebar.radio("Model", ["Mixture NCA", "Stochastic Mixture NCA"], index=0)
    stochastic = "Stochastic" in model_type

    cache_key = (str(exp_dir), model_type)
    if cache_key not in st.session_state.model_cache:
        try:
            if stochastic:
                model, ckpt_path = load_stochastic_nca(exp_dir, device)
            else:
                model, ckpt_path = load_mixture_nca(exp_dir, device)
            st.session_state.model_cache[cache_key] = (model, ckpt_path.name)
        except FileNotFoundError as e:
            st.sidebar.error(str(e))
            return
    model, selected_ckpt = st.session_state.model_cache[cache_key]
    st.sidebar.markdown("**Checkpoint in use**")
    st.sidebar.code(selected_ckpt, language=None)

    # --- Initial state (same histories as in training) ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Initial state (histories as in training)**")
    hist_path_input = st.sidebar.text_input(
        "Histories file",
        value=str(DEFAULT_HISTORIES_PATH),
        help="Same .npy file used to train the models (e.g. notebooks/histories.npy).",
    )
    hist_path = Path(hist_path_input)
    use_histories = hist_path.exists()
    if use_histories:
        histories = np.load(hist_path, allow_pickle=True)
        n_sims = len(histories)
        sim_index = st.sidebar.slider("Simulation index", 0, max(0, n_sims - 1), min(0, n_sims - 1))
        seq = histories[sim_index]
        n_times = len(seq) if hasattr(seq, "__len__") else (seq.shape[0] if isinstance(seq, np.ndarray) else 1)
        time_index = st.sidebar.slider("Initial frame (time)", 0, max(0, n_times - 1), 0)
        initial_grid = load_initial_state_from_histories(hist_path, sim_index, time_index)
    else:
        # Example grid: a few stem cells
        st.sidebar.info("Histories file not found. Using an example grid.")
        try:
            mod = create_complex_model_example(8)
            initial_grid, _ = mod.simulate(1)
            initial_grid = initial_grid[0]
        except Exception:
            initial_grid = np.zeros((50, 50), dtype=np.int64)
            initial_grid[25, 25] = 1  # uno stem al centro
        sim_index = time_index = 0

    n_steps = st.sidebar.slider("Simulation steps", 5, 500, 100, 5)
    seed = None
    if stochastic:
        seed_input = st.sidebar.number_input(
            "Seed (0 = random)",
            min_value=0,
            value=0,
            step=1,
            help="Set e.g. 42 to reproduce the same figures as in curriculum scripts.",
        )
        if seed_input and int(seed_input) != 0:
            seed = int(seed_input)
    run_clicked = st.sidebar.button("Simulate", type="primary")

    # Run simulation; frame slider updates display only
    if run_clicked:
        with st.spinner("Running simulation..."):
            frames = run_rollout(model, initial_grid.copy(), n_steps, device, stochastic, seed=seed)
        st.session_state.rollout_frames = frames
        if use_histories and hist_path.exists():
            gt_frames = load_gt_frames_from_histories(hist_path, sim_index, time_index, n_steps)
            st.session_state.rollout_gt_frames = gt_frames
        else:
            st.session_state.rollout_gt_frames = []
        st.success(f"Simulation completed: {len(frames)} frames.")

    # Slider in a fragment (Streamlit 1.37+): only this block reruns, much more responsive
    if st.session_state.rollout_frames:

        def _render_frames():
            frames = st.session_state.rollout_frames
            gt_frames = st.session_state.get("rollout_gt_frames") or []
            n_frames = len(frames)
            step_index = st.slider("Frame", 0, n_frames - 1, 0, 1, key="step_slider")

            gt_index = min(step_index, len(gt_frames) - 1) if gt_frames else -1
            has_gt = gt_frames and gt_index >= 0

            if has_gt:
                col_gt, col_pred = st.columns(2)
                with col_gt:
                    gt_frame = gt_frames[gt_index]
                    img_gt = resize_for_display(frame_to_rgba(gt_frame))
                    st.image(img_gt, use_container_width=False, caption=f"**Ground truth** (simulator) — step {gt_index}")
                with col_pred:
                    pred_frame = frames[step_index]
                    img_pred = resize_for_display(frame_to_rgba(pred_frame))
                    st.image(img_pred, use_container_width=False, caption=f"**Predicted** (model) — step {step_index}")
            else:
                current_frame = frames[step_index]
                img = resize_for_display(frame_to_rgba(current_frame))
                st.image(img, use_container_width=False, caption=f"Predicted — step {step_index} / {n_frames - 1}")
                if not use_histories:
                    st.caption("Ground truth not available: load a histories file for comparison.")

        fragment = getattr(st, "fragment", None)
        if fragment is not None:
            fragment(_render_frames)()
        else:
            _render_frames()
    else:
        st.info("Click Simulate in the sidebar to start the simulation, then use the Frame slider to browse steps.")


def render_dashboard_page():
    init_session_state()
    st.markdown('<p class="main-header">Experiments dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Aggregated results by neighborhood size and model type.</p>', unsafe_allow_html=True)

    results_base = st.sidebar.text_input(
        "Results folder (dashboard)",
        value=str(DEFAULT_RESULTS_DIR),
        key="dashboard_results",
        help="Same curriculum folder: experiments/results_extended/tissue_simulation_extended.",
    )
    results_base = Path(results_base)
    if not results_base.exists():
        st.warning("Results folder does not exist. Set a valid path in the sidebar.")
        return

    experiments = discover_experiments(results_base)
    if not experiments:
        st.warning("No NB_* subfolder found. Check the path (e.g. tissue_simulation_extended).")
        return

    st.sidebar.metric("Experiments (NB_) found", len(experiments))

    tab1, tab2, tab3 = st.tabs(["By NB_ (experiment summary)", "Metrics table (all)", "Overview charts"])

    with tab1:
        st.caption("For each NB_ subfolder: configuration, saved checkpoints, metrics and charts.")
        for exp in experiments:
            nb_name = exp["path"].name
            nb_size = exp["nb_size"]
            with st.expander(f"**{nb_name}** — Neighborhood size {nb_size}", expanded=True):
                s = exp.get("summary") or {}
                if s:
                    st.markdown("**Configuration**")
                    cfg_cols = ["neighborhood_size", "n_epochs", "time_length", "update_every", "n_cell_types", "n_rules", "hidden_dim", "learning_rate", "curriculum"]
                    cfg = {k: s[k] for k in cfg_cols if k in s}
                    if cfg:
                        st.json(cfg)
                    if s.get("curriculum") and s.get("curriculum_schedule"):
                        st.markdown("**Curriculum**")
                        st.json(s["curriculum_schedule"])
                else:
                    st.write("No summary.json in this folder.")

                ckpts = exp.get("checkpoints") or []
                st.markdown("**Saved checkpoints** (final results)")
                if ckpts:
                    for c in ckpts:
                        st.code(c, language=None)
                else:
                    st.write("No complete .pt file in this folder.")

                st.markdown("**Biological metrics**")
                if exp.get("has_metrics"):
                    df_nb = pd.read_csv(exp["metrics_path"])
                    st.dataframe(df_nb, use_container_width=True, height=180)
                    fig_nb = plot_metrics_single_nb(df_nb, nb_size)
                    if fig_nb:
                        st.plotly_chart(fig_nb, use_container_width=True)
                else:
                    st.write("No biological_metrics.csv file.")

    with tab2:
        df_all = load_all_metrics(experiments)
        if df_all.empty:
            st.write("No aggregated metrics available.")
        else:
            st.dataframe(df_all, use_container_width=True, height=400)

    with tab3:
        df_all = load_all_metrics(experiments)
        if df_all.empty:
            st.write("No data for overview charts.")
        else:
            fig = plot_metrics_dashboard(df_all)
            st.plotly_chart(fig, use_container_width=True)


def main():
    st.sidebar.title("MNCA App")
    page = st.sidebar.radio(
        "Page",
        ["Model test", "Experiments dashboard"],
        index=0,
    )
    st.sidebar.markdown("---")
    if page == "Model test":
        render_test_model_page()
    else:
        render_dashboard_page()


if __name__ == "__main__":
    main()
