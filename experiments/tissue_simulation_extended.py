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


def run_experiment(histories_path, output_dir, neighborhood_sizes,
                   n_epochs=800, time_length=35, update_every=1,
                   n_cell_types=6, device="cuda", n_evaluations=10):
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
        device: Computing device
        n_evaluations: Number of evaluations for stochastic models
    """
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
        nca = ExtendedNCA(
            update_net=classification_update_net(6 * 3, n_channels_out=6),
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
            update_net=classification_update_net(6 * 3, n_channels_out=6 * 2),
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
            update_nets=classification_update_net,
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
            update_nets=classification_update_net,
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
        torch.save(nca.state_dict(), os.path.join(exp_dir, 'standard_nca.pt'))
        print(f"Saved Standard NCA to {exp_dir}/standard_nca.pt")
        
        # Train NCA with Noise
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
        torch.save(nca_with_noise.state_dict(), os.path.join(exp_dir, 'nca_with_noise.pt'))
        print(f"Saved NCA with Noise to {exp_dir}/nca_with_noise.pt")
        
        # Train Mixture NCA
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
        torch.save(mix_nca.state_dict(), os.path.join(exp_dir, 'mixture_nca.pt'))
        print(f"Saved Mixture NCA to {exp_dir}/mixture_nca.pt")
        
        # Train Stochastic Mixture NCA
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
        torch.save(stochastic_mix_nca.state_dict(), os.path.join(exp_dir, 'stochastic_mix_nca.pt'))
        print(f"Saved Stochastic Mixture NCA to {exp_dir}/stochastic_mix_nca.pt")
        
        # Evaluation: Compare generated distributions
        print(f"\nEvaluating models (nb={nb_size})...")
        
        # Compare distributions
        results_df = compare_generated_distributions(
            histories=histories,
            standard_nca=nca.to(device),
            mixture_nca=mix_nca.to(device),
            stochastic_nca=stochastic_mix_nca.to(device),
            nca_with_noise=nca_with_noise.to(device),
            n_steps=35,
            n_evaluations=n_evaluations,
            device=device,
            deterministic_rule_choice=False
        )
        
        # Add neighborhood size column
        results_df['Neighborhood Size'] = nb_size
        
        # Save results
        results_path = os.path.join(exp_dir, 'biological_metrics.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Saved metrics to {results_path}")
        
        # Store for aggregation
        all_results[nb_size] = results_df
        
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
    parser.add_argument('--time_length', type=int, default=35,
                        help='Length of training window')
    parser.add_argument('--update_every', type=int, default=1,
                        help='Steps between updates')
    parser.add_argument('--n_cell_types', type=int, default=6,
                        help='Number of cell types')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device (cuda or cpu)')
    parser.add_argument('--n_evaluations', type=int, default=10,
                        help='Number of evaluations for stochastic models')
    args = parser.parse_args()
    
    sizes = [int(s.strip()) for s in args.neighborhood_sizes.split(',') if s.strip()]
    for s in sizes:
        if s not in (3, 4, 5, 6, 7):
            raise ValueError(f"Unsupported neighborhood size: {s}. Supported: 3,4,5,6,7")
    
    run_experiment(
        histories_path=args.histories_path,
        output_dir=args.output_dir,
        neighborhood_sizes=sizes,
        n_epochs=args.n_epochs,
        time_length=args.time_length,
        update_every=args.update_every,
        n_cell_types=args.n_cell_types,
        device=args.device,
        n_evaluations=args.n_evaluations
    )

