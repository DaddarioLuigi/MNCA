import torch
import numpy as np
import os
import json
import sys
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mix_NCA.utils_simulations import grid_to_channels_batch, train_nca_dyn, classification_update_net
from mix_NCA.ExtendedNCA import ExtendedNCA
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.BiologicalMetrics import compare_generated_distributions
from mix_NCA.TissueModel import ComplexCellType
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


def _parse_csv_ints(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _build_curriculum_schedule(time_lengths, epochs, lrs, update_evers):
    """
    Build a list of phase configs for curriculum learning.

    Each returned phase dict contains:
      - time_length: int
      - n_epochs: int
      - lr: float
      - update_every: int
    """
    if not time_lengths:
        raise ValueError("curriculum_time_lengths must be non-empty")

    def _broadcast_or_validate(name, values, target_len):
        if len(values) == 1 and target_len > 1:
            return values * target_len
        if len(values) != target_len:
            raise ValueError(
                f"{name} must have length 1 or match curriculum_time_lengths "
                f"(got {len(values)} vs {target_len})"
            )
        return values

    n = len(time_lengths)
    epochs = _broadcast_or_validate("curriculum_epochs", epochs, n)
    lrs = _broadcast_or_validate("curriculum_lrs", lrs, n)
    update_evers = _broadcast_or_validate("curriculum_update_every", update_evers, n)

    schedule = []
    for tl, ne, lr, ue in zip(time_lengths, epochs, lrs, update_evers):
        if tl <= 0:
            raise ValueError(f"Invalid time_length in curriculum: {tl}")
        if ne <= 0:
            raise ValueError(f"Invalid n_epochs in curriculum: {ne}")
        if lr <= 0:
            raise ValueError(f"Invalid lr in curriculum: {lr}")
        if ue <= 0:
            raise ValueError(f"Invalid update_every in curriculum: {ue}")
        schedule.append({"time_length": tl, "n_epochs": ne, "lr": lr, "update_every": ue})
    return schedule


def _train_with_schedule(
    *,
    model,
    model_label: str,
    histories,
    schedule,
    base_train_kwargs,
    final_ckpt_path: str,
    loss_curve_path: str,
):
    """
    Train a model using a curriculum schedule (multiple phases) and save a final checkpoint.
    Also saves a per-phase loss curve (concatenated) for quick inspection.
    """
    if os.path.exists(final_ckpt_path):
        print(f"Found {final_ckpt_path}, loading {model_label}...")
        model.load_state_dict(torch.load(final_ckpt_path, weights_only=True))
        return None

    all_losses = []
    for i, phase in enumerate(schedule, start=1):
        print(
            f"  Phase {i}/{len(schedule)} for {model_label}: "
            f"time_length={phase['time_length']}, n_epochs={phase['n_epochs']}, "
            f"lr={phase['lr']}, update_every={phase['update_every']}"
        )

        phase_losses = train_nca_dyn(
            model=model,
            target_states=histories,
            n_epochs=phase["n_epochs"],
            time_length=phase["time_length"],
            update_every=phase["update_every"],
            lr=phase["lr"],
            return_losses=True,
            **base_train_kwargs,
        )
        if phase_losses is not None:
            all_losses.extend(list(phase_losses))

        # Save intermediate checkpoint for resumability / debugging
        phase_ckpt = final_ckpt_path.replace(".pt", f"_phase{i}_TL{phase['time_length']}.pt")
        try:
            torch.save(model.state_dict(), phase_ckpt)
        except Exception as e:
            print(f"    Warning: could not save intermediate checkpoint {phase_ckpt}: {e}")

    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Saved {model_label} to {final_ckpt_path}")
    if all_losses:
        np.save(loss_curve_path, np.asarray(all_losses))
    return all_losses


def _generate_videos_for_models(models, histories, n_steps, nb_size, output_dir, device, n_samples=3):
    """
    Generate videos showing the evolution of NCA models
    
    Args:
        models: Dictionary of model name -> model instance
        histories: List of true state histories
        n_steps: Number of steps to simulate
        nb_size: Neighborhood size
        output_dir: Directory to save videos
        device: Computing device
        n_samples: Number of sample simulations to create videos for
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("Warning: matplotlib not available, skipping video generation")
        return
    
    video_dir = Path(output_dir) / 'videos' / f'steps_{n_steps}'
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Select a few sample histories
    sample_indices = np.linspace(0, len(histories) - 1, min(n_samples, len(histories)), dtype=int)
    
    for model_name, model in models.items():
        print(f"      Generating video for {model_name}...")
        
        for sample_idx in sample_indices:
            # Get initial state
            grid_state = histories[sample_idx][0]
            initial_state = grid_to_channels_batch([grid_state], len(ComplexCellType), device)
            
            # Run simulation and collect history
            model.eval()
            with torch.no_grad():
                result = model(initial_state, n_steps, return_history=True)
                
                # Extract "history" if returned, otherwise treat result as a single frame
                history = result[1] if isinstance(result, tuple) and len(result) > 1 else result

            n_cell_types = len(ComplexCellType)

            def _as_numpy_label_grid(frame_like):
                """
                Convert a frame-like object to a 2D numpy array of integer cell labels.
                Accepts torch tensors or numpy arrays in shapes:
                  - (C,H,W)
                  - (B,C,H,W) (uses first batch)
                  - (H,W) (already labels)
                """
                if isinstance(frame_like, np.ndarray):
                    arr = frame_like
                    if arr.ndim == 2:
                        return arr
                    t = torch.from_numpy(arr)
                elif isinstance(frame_like, torch.Tensor):
                    t = frame_like
                else:
                    return None

                # Move to CPU for indexing and ensure tensor
                if isinstance(t, torch.Tensor):
                    t = t.detach().cpu()

                if t.ndim == 4:
                    # (B,C,H,W) or (T,C,H,W) - caller should have flattened time already.
                    # Prefer interpreting as batch and taking the first element.
                    t = t[0]
                if t.ndim == 3:
                    # (C,H,W): slice to biological channels then argmax over channel dimension
                    t = t[:n_cell_types].argmax(dim=0)
                if t.ndim == 2:
                    return t.numpy()
                return None

            def _flatten_to_frame_list(history_like):
                """
                Flatten history to a list of frame-likes, handling common layouts:
                  - list/tuple of frames
                  - torch.Tensor of shape (T,B,C,H,W) or (T,C,H,W) or (B,C,H,W)
                  - numpy arrays with the same conventions
                """
                if history_like is None:
                    return []

                if isinstance(history_like, (list, tuple)):
                    out = []
                    for item in history_like:
                        out.extend(_flatten_to_frame_list(item))
                    return out

                if isinstance(history_like, np.ndarray):
                    arr = history_like
                    # Treat (T,...) as time-major if first dim > 1 and array has >= 4 dims or looks like (T,C,H,W)
                    if arr.ndim >= 4 and arr.shape[0] > 1:
                        return [arr[i] for i in range(arr.shape[0])]
                    return [arr]

                if isinstance(history_like, torch.Tensor):
                    t = history_like
                    # (T,B,C,H,W)
                    if t.ndim == 5:
                        return [t[i] for i in range(t.shape[0])]
                    # (T,C,H,W) OR (B,C,H,W)
                    if t.ndim == 4 and t.shape[0] > 1 and t.shape[1] == n_cell_types:
                        # Very likely (T,C,H,W)
                        return [t[i] for i in range(t.shape[0])]
                    return [t]

                return []

            # Convert frames to numpy arrays (cell type classifications)
            frame_images = []
            for frame_like in _flatten_to_frame_list(history):
                labels = _as_numpy_label_grid(frame_like)
                if labels is not None:
                    frame_images.append(labels)
            
            if len(frame_images) == 0:
                continue
            
            # Create color map for cell types
            colors = plt.cm.tab10(np.linspace(0, 1, n_cell_types))
            cmap = plt.cm.colors.ListedColormap(colors)
            
            # Create animation
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.axis('off')
            
            im = ax.imshow(frame_images[0], cmap=cmap, vmin=0, vmax=n_cell_types-1, animated=True)
            
            def update_frame(frame_num):
                im.set_array(frame_images[min(frame_num, len(frame_images)-1)])
                ax.set_title(f'{model_name} - Step {min(frame_num, len(frame_images)-1)}/{len(frame_images)-1}\nNB={nb_size}')
                return [im]
            
            anim = animation.FuncAnimation(
                fig, update_frame, frames=len(frame_images),
                interval=100, blit=True, repeat=True
            )
            
            # Save video
            video_path = video_dir / f'{model_name}_sample_{sample_idx}_nb_{nb_size}.mp4'
            try:
                anim.save(str(video_path), writer='ffmpeg', fps=10, bitrate=1800)
                print(f"        Saved: {video_path}")
            except Exception as e:
                print(f"        Warning: Could not save video {video_path}: {e}")
                # Fallback: save as GIF
                try:
                    gif_path = video_path.with_suffix('.gif')
                    anim.save(str(gif_path), writer='pillow', fps=10)
                    print(f"        Saved as GIF: {gif_path}")
                except Exception as e2:
                    print(f"        Error saving GIF: {e2}")
            
            plt.close(fig)


def get_device(device_preference="auto"):
    """
    Get the best available device.
    
    Args:
        device_preference: "auto", "cuda", "mps", or "cpu"
    
    Returns:
        str: Device string ("cuda", "mps", or "cpu")
    """
    if device_preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    elif device_preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif device_preference == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        if device_preference not in ("cpu", "cuda", "mps", "auto"):
            print(f"Warning: Unknown device '{device_preference}', falling back to 'cpu'")
        return "cpu"


def run_experiment(histories_path, output_dir, neighborhood_sizes,
                   n_epochs=800, time_length=500, update_every=1,
                   n_cell_types=6, device="auto", n_evaluations=10,
                   step_lengths=[25, 100, 500], generate_videos=False,
                   curriculum=False, curriculum_schedule=None):
    """
    Train and evaluate NCA models with different neighborhood sizes on biological simulations
    
    Args:
        histories_path: Path to histories.npy file
        output_dir: Directory to save results
        neighborhood_sizes: List of neighborhood sizes to test (e.g., [3, 4, 5, 6, 7])
        n_epochs: Number of training epochs
        time_length: Length of training window
        update_every: Steps between updates
        n_cell_types: Number of cell types
        device: Computing device ("auto", "cuda", "mps", or "cpu")
        n_evaluations: Number of evaluations for stochastic models
    """
    # Auto-detect best device if needed
    device = get_device(device)
    print(f"Using device: {device}")
    
    # Load histories
    print(f"Loading histories from {histories_path}...")
    histories = np.load(histories_path, allow_pickle=True)
    print(f"Loaded {len(histories)} simulations")
    
    # Training hyperparameters (same as base experiment)
    HIDDEN_DIM = 128
    STATE_DIM = 6
    N_RULES = 5
    LEARNING_RATE = 0.001
    MILESTONES = [500]
    GAMMA = 0.1
    TEMPERATURE = 5
    MIN_TEMPERATURE = 0.1
    ANNEAL_RATE = 0.006
    LOSS_TYPE = "mse"

    if curriculum:
        if not curriculum_schedule:
            raise ValueError("curriculum=True but curriculum_schedule is empty")
        print("\nCurriculum learning enabled. Schedule:")
        for i, phase in enumerate(curriculum_schedule, start=1):
            print(
                f"  - Phase {i}: time_length={phase['time_length']}, "
                f"n_epochs={phase['n_epochs']}, lr={phase['lr']}, "
                f"update_every={phase['update_every']}"
            )
    
    base_dir = os.path.join(output_dir, "tissue_simulation_extended")
    os.makedirs(base_dir, exist_ok=True)
    
    # Results storage
    all_results = {}
    
    for nb_size in neighborhood_sizes:
        print(f"\n{'='*60}")
        print(f"=== Neighborhood size: {nb_size}x{nb_size} ===")
        print(f"{'='*60}\n")
        
        exp_dir = os.path.join(base_dir, f"NB_{nb_size}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create models with extended neighborhood
        print("Initializing models...")
        
        # Create a wrapper function that passes device to classification_update_net
        # MixtureNCA calls update_nets with: (state_dim * input_mult, hidden_dim, state_dim, device)
        # We need to accept all 4 arguments but only pass the first 3 to classification_update_net
        def make_update_net_fn(device):
            def update_net_wrapper(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
                # device_arg is passed by MixtureNCA but we use the device from closure
                return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)
            return update_net_wrapper
        
        update_net_fn = make_update_net_fn(device)
        
        nca = ExtendedNCA(
            update_net=classification_update_net(6 * 3, n_channels_out=6, device=device),
            hidden_dim=HIDDEN_DIM,
            maintain_seed=False,
            use_alive_mask=False,
            state_dim=STATE_DIM,
            residual=False,
            neighborhood_size=nb_size,
            device=device
        )
        
        # NCA with noise (GCA - Gaussian Cellular Automata)
        nca_with_noise = ExtendedNCA(
            update_net=classification_update_net(6 * 3, n_channels_out=6 * 2, device=device),
            hidden_dim=HIDDEN_DIM,
            maintain_seed=False,
            use_alive_mask=False,
            state_dim=STATE_DIM,
            residual=False,
            distribution="normal",
            neighborhood_size=nb_size,
            device=device
        )
        nca_with_noise.random_updates = True
        
        mix_nca = ExtendedMixtureNCA(
            update_nets=update_net_fn,
            hidden_dim=HIDDEN_DIM,
            maintain_seed=False,
            use_alive_mask=False,
            state_dim=STATE_DIM,
            num_rules=N_RULES,
            residual=False,
            temperature=3,
            neighborhood_size=nb_size,
            device=device
        )
        
        stochastic_mix_nca = ExtendedMixtureNCANoise(
            update_nets=update_net_fn,
            hidden_dim=HIDDEN_DIM,
            maintain_seed=False,
            use_alive_mask=False,
            state_dim=STATE_DIM,
            num_rules=N_RULES,
            residual=False,
            temperature=3,
            neighborhood_size=nb_size,
            device=device
        )
        
        # Train Standard NCA
        std_path = os.path.join(exp_dir, 'standard_nca.pt' if not curriculum else 'standard_nca_curriculum.pt')
        std_loss_path = os.path.join(exp_dir, 'standard_nca_loss_curve.npy' if not curriculum else 'standard_nca_curriculum_loss_curve.npy')
        if curriculum:
            print(f"\nTraining Standard NCA with curriculum (nb={nb_size})...")
            _train_with_schedule(
                model=nca,
                model_label=f"Standard NCA (nb={nb_size})",
                histories=histories,
                schedule=curriculum_schedule,
                base_train_kwargs={"n_cell_types": n_cell_types, "device": device, "loss_type": LOSS_TYPE},
                final_ckpt_path=std_path,
                loss_curve_path=std_loss_path,
            )
        else:
            if os.path.exists(std_path):
                print(f"Found {std_path}, loading Standard NCA (nb={nb_size})...")
                nca.load_state_dict(torch.load(std_path, weights_only=True))
            else:
                print(f"\nTraining Standard NCA (nb={nb_size})...")
                std_losses = train_nca_dyn(
                    nca, histories,
                    n_cell_types=n_cell_types,
                    n_epochs=n_epochs,
                    time_length=time_length,
                    update_every=update_every,
                    device=device,
                    lr=LEARNING_RATE,
                    return_losses=True,
                    loss_type=LOSS_TYPE
                )
                torch.save(nca.state_dict(), std_path)
                print(f"Saved Standard NCA to {std_path}")
                np.save(std_loss_path, np.asarray(std_losses))
        
        # Train NCA with Noise
        gca_path = os.path.join(exp_dir, 'nca_with_noise.pt' if not curriculum else 'nca_with_noise_curriculum.pt')
        gca_loss_path = os.path.join(exp_dir, 'nca_with_noise_loss_curve.npy' if not curriculum else 'nca_with_noise_curriculum_loss_curve.npy')
        if curriculum:
            print(f"\nTraining NCA with Noise with curriculum (nb={nb_size})...")
            # Keep the same relative LR scaling as the non-curriculum path (LR/10)
            scaled_schedule = [
                {**p, "lr": p["lr"] / 10.0} for p in curriculum_schedule
            ]
            _train_with_schedule(
                model=nca_with_noise,
                model_label=f"NCA with Noise (nb={nb_size})",
                histories=histories,
                schedule=scaled_schedule,
                base_train_kwargs={"n_cell_types": n_cell_types, "device": device, "loss_type": LOSS_TYPE},
                final_ckpt_path=gca_path,
                loss_curve_path=gca_loss_path,
            )
        else:
            if os.path.exists(gca_path):
                print(f"Found {gca_path}, loading NCA with Noise (nb={nb_size})...")
                nca_with_noise.load_state_dict(torch.load(gca_path, weights_only=True))
            else:
                print(f"\nTraining NCA with Noise (nb={nb_size})...")
                gca_losses = train_nca_dyn(
                    nca_with_noise, histories,
                    n_cell_types=n_cell_types,
                    n_epochs=n_epochs,
                    time_length=time_length,
                    update_every=update_every,
                    device=device,
                    lr=LEARNING_RATE / 10,
                    return_losses=True,
                    loss_type=LOSS_TYPE
                )
                torch.save(nca_with_noise.state_dict(), gca_path)
                print(f"Saved NCA with Noise to {gca_path}")
                np.save(gca_loss_path, np.asarray(gca_losses))
        
        # Train Mixture NCA
        mix_path = os.path.join(exp_dir, 'mixture_nca.pt' if not curriculum else 'mixture_nca_curriculum.pt')
        mix_loss_path = os.path.join(exp_dir, 'mixture_nca_loss_curve.npy' if not curriculum else 'mixture_nca_curriculum_loss_curve.npy')
        if curriculum:
            print(f"\nTraining Mixture NCA with curriculum (nb={nb_size})...")
            _train_with_schedule(
                model=mix_nca,
                model_label=f"Mixture NCA (nb={nb_size})",
                histories=histories,
                schedule=curriculum_schedule,
                base_train_kwargs={
                    "n_cell_types": n_cell_types,
                    "device": device,
                    "temperature": TEMPERATURE,
                    "min_temperature": MIN_TEMPERATURE,
                    "anneal_rate": ANNEAL_RATE,
                    "loss_type": LOSS_TYPE,
                    "straight_through": False,
                },
                final_ckpt_path=mix_path,
                loss_curve_path=mix_loss_path,
            )
        else:
            if os.path.exists(mix_path):
                print(f"Found {mix_path}, loading Mixture NCA (nb={nb_size})...")
                mix_nca.load_state_dict(torch.load(mix_path, weights_only=True))
            else:
                print(f"\nTraining Mixture NCA (nb={nb_size})...")
                mix_losses = train_nca_dyn(
                    model=mix_nca,
                    target_states=histories,
                    n_cell_types=n_cell_types,
                    n_epochs=n_epochs,
                    time_length=time_length,
                    update_every=update_every,
                    device=device,
                    lr=LEARNING_RATE,
                    temperature=TEMPERATURE,
                    min_temperature=MIN_TEMPERATURE,
                    anneal_rate=ANNEAL_RATE,
                    anneal_temperature=False,
                    loss_type=LOSS_TYPE,
                    straight_through=False,
                    return_losses=True
                )
                torch.save(mix_nca.state_dict(), mix_path)
                print(f"Saved Mixture NCA to {mix_path}")
                np.save(mix_loss_path, np.asarray(mix_losses))
        
        # Train Stochastic Mixture NCA
        stoch_path = os.path.join(exp_dir, 'stochastic_mix_nca.pt' if not curriculum else 'stochastic_mix_nca_curriculum.pt')
        stoch_loss_path = os.path.join(exp_dir, 'stochastic_mix_nca_loss_curve.npy' if not curriculum else 'stochastic_mix_nca_curriculum_loss_curve.npy')
        if curriculum:
            print(f"\nTraining Stochastic Mixture NCA with curriculum (nb={nb_size})...")
            _train_with_schedule(
                model=stochastic_mix_nca,
                model_label=f"Stochastic Mixture NCA (nb={nb_size})",
                histories=histories,
                schedule=curriculum_schedule,
                base_train_kwargs={
                    "n_cell_types": n_cell_types,
                    "device": device,
                    "milestones": MILESTONES,
                    "gamma": GAMMA,
                    "temperature": TEMPERATURE,
                    "min_temperature": MIN_TEMPERATURE,
                    "anneal_rate": ANNEAL_RATE,
                },
                final_ckpt_path=stoch_path,
                loss_curve_path=stoch_loss_path,
            )
        else:
            if os.path.exists(stoch_path):
                print(f"Found {stoch_path}, loading Stochastic Mixture NCA (nb={nb_size})...")
                stochastic_mix_nca.load_state_dict(torch.load(stoch_path, weights_only=True))
            else:
                print(f"\nTraining Stochastic Mixture NCA (nb={nb_size})...")
                stoch_losses = train_nca_dyn(
                    model=stochastic_mix_nca,
                    target_states=histories,
                    n_cell_types=n_cell_types,
                    n_epochs=n_epochs,
                    time_length=time_length,
                    update_every=update_every,
                    device=device,
                    lr=LEARNING_RATE,
                    milestones=MILESTONES,
                    gamma=GAMMA,
                    temperature=TEMPERATURE,
                    min_temperature=MIN_TEMPERATURE,
                    anneal_rate=ANNEAL_RATE,
                    anneal_temperature=False,
                    return_losses=True
                )
                torch.save(stochastic_mix_nca.state_dict(), stoch_path)
                print(f"Saved Stochastic Mixture NCA to {stoch_path}")
                np.save(stoch_loss_path, np.asarray(stoch_losses))
        
        # Evaluation: Compare generated distributions for each step length
        print(f"\nEvaluating models (nb={nb_size})...")
        
        all_step_results = []
        
        for n_steps in step_lengths:
            print(f"\n  Evaluating with {n_steps} steps...")
            
            # Compare distributions
            results_df = compare_generated_distributions(
                histories=histories,
                standard_nca=nca.to(device),
                mixture_nca=mix_nca.to(device),
                stochastic_nca=stochastic_mix_nca.to(device),
                nca_with_noise=nca_with_noise.to(device),
                n_steps=n_steps,
                n_evaluations=n_evaluations,
                device=device,
                deterministic_rule_choice=False,
                # Keep evaluation rule sampling consistent across mixture models
                sample_non_differentiable=False,
                straight_through=True,
                temperature=None
            )
            
            # Add neighborhood size and step length columns
            results_df['Neighborhood Size'] = nb_size
            results_df['Step Length'] = n_steps
            all_step_results.append(results_df)
            
            # Generate video if requested
            if generate_videos:
                print(f"    Generating videos for {n_steps} steps...")
                _generate_videos_for_models(
                    models={
                        'standard_nca': nca.to(device),
                        'mixture_nca': mix_nca.to(device),
                        'stochastic_nca': stochastic_mix_nca.to(device),
                        'nca_with_noise': nca_with_noise.to(device)
                    },
                    histories=histories,
                    n_steps=n_steps,
                    nb_size=nb_size,
                    output_dir=exp_dir,
                    device=device
                )
        
        # Combine results from all step lengths
        combined_results = pd.concat(all_step_results, ignore_index=True)
        
        # Save results
        results_path = os.path.join(exp_dir, 'biological_metrics.csv')
        combined_results.to_csv(results_path, index=False)
        print(f"Saved metrics to {results_path}")
        
        # Store for aggregation
        all_results[nb_size] = combined_results
        
        # Save models summary
        summary = {
            'neighborhood_size': nb_size,
            'n_epochs': n_epochs,
            'time_length': time_length,
            'update_every': update_every,
            'n_cell_types': n_cell_types,
            'n_rules': N_RULES,
            'hidden_dim': HIDDEN_DIM,
            'learning_rate': LEARNING_RATE,
            'curriculum': bool(curriculum),
            'curriculum_schedule': curriculum_schedule if curriculum else None,
            'models_saved': {
                'standard_nca': std_path,
                'nca_with_noise': gca_path,
                'mixture_nca': mix_path,
                'stochastic_mix_nca': stoch_path
            }
        }
        
        with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Aggregate all results
    print(f"\n{'='*60}")
    print("Aggregating results across all neighborhood sizes...")
    print(f"{'='*60}\n")
    
    if all_results:
        aggregated_df = pd.concat(all_results.values(), ignore_index=True)
        aggregated_path = os.path.join(base_dir, 'all_neighborhood_sizes_metrics.csv')
        aggregated_df.to_csv(aggregated_path, index=False)
        print(f"Saved aggregated results to {aggregated_path}")
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train NCA models with different neighborhood sizes on biological simulations')
    parser.add_argument('--histories_path', type=str, default='../notebooks/histories.npy',
                        help='Path to histories.npy file')
    parser.add_argument('--output_dir', type=str, default='results_extended',
                        help='Output directory for results')
    parser.add_argument('--neighborhood_sizes', type=str, default='3,4,5,6,7',
                        help='Comma-separated list of neighborhood sizes, e.g. 3,4,5,6,7')
    parser.add_argument('--n_epochs', type=int, default=800,
                        help='Number of training epochs')
    parser.add_argument('--time_length', type=int, default=500,
                        help='Length of training window (default: 500)')
    parser.add_argument('--update_every', type=int, default=1,
                        help='Steps between updates')
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning over multiple time_length phases')
    parser.add_argument('--curriculum_time_lengths', type=str, default='35,100,500',
                        help='Comma-separated list of time_length values for curriculum phases (default: 35,100,500)')
    parser.add_argument('--curriculum_epochs', type=str, default='250,150,60',
                        help='Comma-separated list of n_epochs per curriculum phase (default: 250,150,60)')
    parser.add_argument('--curriculum_lrs', type=str, default='0.001,0.0005,0.0001',
                        help='Comma-separated list of learning rates per curriculum phase (default: 0.001,0.0005,0.0001)')
    parser.add_argument('--curriculum_update_every', type=str, default='1,1,2',
                        help='Comma-separated list of update_every per curriculum phase (default: 1,1,2)')
    parser.add_argument('--n_cell_types', type=int, default=6,
                        help='Number of cell types')
    parser.add_argument('--device', type=str, default='auto',
                        help='Computing device (auto, cuda, mps, or cpu). "auto" will select the best available device.')
    parser.add_argument('--n_evaluations', type=int, default=10,
                        help='Number of evaluations for stochastic models')
    parser.add_argument('--step_lengths', type=str, default='25,100,500',
                        help='Comma-separated list of step lengths to test (default: 25,100,500)')
    parser.add_argument('--generate_videos', action='store_true',
                        help='Generate videos of model evolution')
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.neighborhood_sizes.split(',') if s.strip()]
    for s in sizes:
        if s not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError(f"Unsupported neighborhood size: {s}. Supported: 1,2,3,4,5,6,7")
    
    step_lengths = [int(s.strip()) for s in args.step_lengths.split(',') if s.strip()]

    curriculum_schedule = None
    if args.curriculum:
        curriculum_schedule = _build_curriculum_schedule(
            time_lengths=_parse_csv_ints(args.curriculum_time_lengths),
            epochs=_parse_csv_ints(args.curriculum_epochs),
            lrs=_parse_csv_floats(args.curriculum_lrs),
            update_evers=_parse_csv_ints(args.curriculum_update_every),
        )
    
    run_experiment(
        histories_path=args.histories_path,
        output_dir=args.output_dir,
        neighborhood_sizes=sizes,
        n_epochs=args.n_epochs,
        time_length=args.time_length,
        update_every=args.update_every,
        n_cell_types=args.n_cell_types,
        device=args.device,
        n_evaluations=args.n_evaluations,
        step_lengths=step_lengths,
        generate_videos=args.generate_videos,
        curriculum=args.curriculum,
        curriculum_schedule=curriculum_schedule,
    )

