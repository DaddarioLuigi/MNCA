#!/usr/bin/env python3
"""
Script to generate histories.npy file for tissue simulation experiments.
This script replicates the code from notebooks/tissue_simulation_MNCA.ipynb
"""

import numpy as np
import sys
import os

# Add parent directory to path to import mix_NCA modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from mix_NCA.TissueModel import create_complex_model_example

def generate_histories(output_path='notebooks/histories.npy', n_simulations=1000, n_steps=500):
    """
    Generate histories.npy file by running tissue simulations.
    
    Args:
        output_path: Path where to save histories.npy
        n_simulations: Number of simulations to run
        n_steps: Number of steps per simulation
    """
    print(f"Generating {n_simulations} simulations with {n_steps} steps each...")
    
    histories = []
    spatial_models = []
    
    for i in range(n_simulations):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i + 1}/{n_simulations} simulations completed")
        
        n_stems = np.random.randint(5, high=15)
        model = create_complex_model_example(n_stems)
        history, _ = model.simulate(n_steps)
        histories.append(history)
        spatial_models.append(model)
    
    # Save the histories
    print(f"Saving histories to {output_path}...")
    np.save(output_path, histories)
    print(f"Successfully saved {len(histories)} simulations to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate histories.npy file for tissue simulations')
    parser.add_argument('--output_path', type=str, default='notebooks/histories.npy',
                        help='Path where to save histories.npy (default: notebooks/histories.npy)')
    parser.add_argument('--n_simulations', type=int, default=1000,
                        help='Number of simulations to run (default: 1000)')
    parser.add_argument('--n_steps', type=int, default=500,
                        help='Number of steps per simulation (default: 500)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    generate_histories(
        output_path=args.output_path,
        n_simulations=args.n_simulations,
        n_steps=args.n_steps
    )

