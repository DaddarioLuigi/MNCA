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
                
                if isinstance(result, tuple):
                    frames = result[1] if len(result[1]) > 0 else [result[0]]
                else:
                    frames = result if isinstance(result, list) else [result]
            
            # Convert frames to numpy arrays (cell type classifications)
            frame_images = []
            for frame in frames:
                if isinstance(frame, torch.Tensor):
                    # Get cell type classification
                    cell_types = frame.argmax(dim=1).cpu().numpy()
                    if len(cell_types.shape) == 3:
                        cell_types = cell_types[0]  # Remove batch dimension
                    frame_images.append(cell_types)
            
            if len(frame_images) == 0:
                continue
            
            # Create color map for cell types
            n_cell_types = len(ComplexCellType)
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
                   step_lengths=[35, 100, 500], generate_videos=False):
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
        std_path = os.path.join(exp_dir, 'standard_nca.pt')
        if os.path.exists(std_path):
            print(f"Found {std_path}, loading Standard NCA (nb={nb_size})...")
            nca.load_state_dict(torch.load(std_path, weights_only=True))
        else:
            print(f"\nTraining Standard NCA (nb={nb_size})...")
            train_nca_dyn(
                nca, histories,
                n_cell_types=n_cell_types,
                n_epochs=n_epochs,
                time_length=time_length,
                update_every=update_every,
                device=device,
                lr=LEARNING_RATE
            )
            # Save Standard NCA
            torch.save(nca.state_dict(), std_path)
            print(f"Saved Standard NCA to {std_path}")
        
        # Train NCA with Noise
        gca_path = os.path.join(exp_dir, 'nca_with_noise.pt')
        if os.path.exists(gca_path):
            print(f"Found {gca_path}, loading NCA with Noise (nb={nb_size})...")
            nca_with_noise.load_state_dict(torch.load(gca_path, weights_only=True))
        else:
            print(f"\nTraining NCA with Noise (nb={nb_size})...")
            train_nca_dyn(
                nca_with_noise, histories,
                n_cell_types=n_cell_types,
                n_epochs=n_epochs,
                time_length=time_length,
                update_every=update_every,
                device=device,
                lr=LEARNING_RATE / 10  # Lower LR for noise model as in notebook
            )
            # Save NCA with Noise
            torch.save(nca_with_noise.state_dict(), gca_path)
            print(f"Saved NCA with Noise to {gca_path}")
        
        # Train Mixture NCA
        mix_path = os.path.join(exp_dir, 'mixture_nca.pt')
        if os.path.exists(mix_path):
            print(f"Found {mix_path}, loading Mixture NCA (nb={nb_size})...")
            mix_nca.load_state_dict(torch.load(mix_path, weights_only=True))
        else:
            print(f"\nTraining Mixture NCA (nb={nb_size})...")
            train_nca_dyn(
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
                loss_type=LOSS_TYPE,
                straight_through=False
            )
            # Save Mixture NCA
            torch.save(mix_nca.state_dict(), mix_path)
            print(f"Saved Mixture NCA to {mix_path}")
        
        # Train Stochastic Mixture NCA
        stoch_path = os.path.join(exp_dir, 'stochastic_mix_nca.pt')
        if os.path.exists(stoch_path):
            print(f"Found {stoch_path}, loading Stochastic Mixture NCA (nb={nb_size})...")
            stochastic_mix_nca.load_state_dict(torch.load(stoch_path, weights_only=True))
        else:
            print(f"\nTraining Stochastic Mixture NCA (nb={nb_size})...")
            train_nca_dyn(
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
                anneal_rate=ANNEAL_RATE
            )
            # Save Stochastic Mixture NCA
            torch.save(stochastic_mix_nca.state_dict(), stoch_path)
            print(f"Saved Stochastic Mixture NCA to {stoch_path}")
        
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
                deterministic_rule_choice=False
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
            'models_saved': {
                'standard_nca': os.path.join(exp_dir, 'standard_nca.pt'),
                'nca_with_noise': os.path.join(exp_dir, 'nca_with_noise.pt'),
                'mixture_nca': os.path.join(exp_dir, 'mixture_nca.pt'),
                'stochastic_mix_nca': os.path.join(exp_dir, 'stochastic_mix_nca.pt')
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
    parser.add_argument('--n_cell_types', type=int, default=6,
                        help='Number of cell types')
    parser.add_argument('--device', type=str, default='auto',
                        help='Computing device (auto, cuda, mps, or cpu). "auto" will select the best available device.')
    parser.add_argument('--n_evaluations', type=int, default=10,
                        help='Number of evaluations for stochastic models')
    parser.add_argument('--step_lengths', type=str, default='35,100,500',
                        help='Comma-separated list of step lengths to test (default: 35,100,500)')
    parser.add_argument('--generate_videos', action='store_true',
                        help='Generate videos of model evolution')
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.neighborhood_sizes.split(',') if s.strip()]
    for s in sizes:
        if s not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError(f"Unsupported neighborhood size: {s}. Supported: 1,2,3,4,5,6,7")
    
    step_lengths = [int(s.strip()) for s in args.step_lengths.split(',') if s.strip()]
    
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
        generate_videos=args.generate_videos
    )

