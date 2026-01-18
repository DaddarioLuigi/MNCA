"""
Scientific Analysis of Neighborhood Size Impact on NCA Models

This script performs a comprehensive scientific analysis comparing NCA models
trained with different neighborhood sizes (3, 4, 5, 6, 7).

Analysis includes:
1. Model evaluation and metric computation
2. Performance trend analysis
3. Computational complexity analysis
4. Model-specific comparisons
5. Visualization of results
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import json
import time
from pathlib import Path
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, norm
from itertools import combinations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mix_NCA.utils_simulations import grid_to_channels_batch, classification_update_net
from mix_NCA.ExtendedNCA import ExtendedNCA
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.BiologicalMetrics import compare_generated_distributions
from mix_NCA.TissueModel import ComplexCellType
from experiments.tissue_simulation_extended import get_device


class NeighborhoodSizeAnalyzer:
    """Comprehensive analyzer for neighborhood size experiments"""
    
    def __init__(self, results_dir: str, histories_path: str, 
                 device: str = "auto", n_evaluations: int = 10,
                 step_lengths: List[int] = [35, 100, 500],
                 model_files: Optional[Dict[str, List[str]]] = None,
                 experiment_subdir: Optional[str] = "tissue_simulation_extended"):
        """
        Initialize analyzer
        
        Args:
            results_dir: Base results directory. By default, the analyzer will look inside
                results_dir/experiment_subdir. If you already pass the experiment folder itself
                (e.g., .../tissue_simulation_extended), it will be used directly.
            histories_path: Path to histories.npy file
            device: Computing device
            n_evaluations: Number of evaluations for stochastic models
            model_files: Optional mapping from model label -> list of checkpoint filenames to try
                inside each NB_k folder. If omitted, defaults to the standard checkpoint names.
            experiment_subdir: Subfolder under results_dir that contains NB_* folders.
                - Default: "tissue_simulation_extended" (backwards compatible)
                - Set to None to treat results_dir as the experiment folder directly.
        """
        self.results_dir = Path(results_dir)
        self.histories_path = histories_path
        self.device = get_device(device)
        self.n_evaluations = n_evaluations
        self.step_lengths = step_lengths if isinstance(step_lengths, list) else [step_lengths]

        # Resolve experiment directory:
        # - If experiment_subdir is None: treat results_dir as already pointing at the experiment folder
        # - Else: use results_dir/experiment_subdir, but avoid double-appending if results_dir already
        #   points to that folder.
        if experiment_subdir is None:
            self.base_dir = self.results_dir
        else:
            exp_subdir = str(experiment_subdir)
            if self.results_dir.name == exp_subdir:
                self.base_dir = self.results_dir
            else:
                self.base_dir = self.results_dir / exp_subdir
        
        # Hyperparameters (should match training)
        self.HIDDEN_DIM = 128
        self.STATE_DIM = 6
        self.N_RULES = 5
        
        # Load histories
        print(f"Loading histories from {histories_path}...")
        self.histories = np.load(histories_path, allow_pickle=True)
        print(f"Loaded {len(self.histories)} simulations")
        
        # Storage for results
        self.metrics_data = {}
        self.computational_times = {}

        # Which checkpoint filenames to search for in each NB_k folder.
        # Keys must match the labels used in _evaluate_models.
        default_model_files: Dict[str, List[str]] = {
            "Mixture NCA": ["mixture_nca_1000.pt", "mixture_nca.pt"],
            "Stochastic Mixture NCA": ["stochastic_mix_nca_1000.pt", "stochastic_mix_nca.pt"],
        }
        self.model_files: Dict[str, List[str]] = default_model_files
        if model_files:
            # Allow partial override (e.g., only override one model’s filenames)
            for k, v in model_files.items():
                if isinstance(v, str):
                    self.model_files[k] = [v]
                else:
                    self.model_files[k] = list(v)
        
    def load_or_evaluate_models(self, neighborhood_sizes: List[int] = [1, 2, 3, 4, 5, 6, 7],
                                force_recompute: bool = False,
                                model_files: Optional[Dict[str, List[str]]] = None):
        """
        Load existing metrics or evaluate models to compute metrics
        
        Args:
            neighborhood_sizes: List of neighborhood sizes to analyze
            force_recompute: If True, recompute metrics even if CSV exists
            model_files: Optional override for which checkpoint filenames to search for
                inside each NB_k folder (same format as in __init__).
        """
        print(f"\n{'='*60}")
        print("Loading/Evaluating Models")
        print(f"{'='*60}\n")
        
        if model_files:
            for k, v in model_files.items():
                if isinstance(v, str):
                    self.model_files[k] = [v]
                else:
                    self.model_files[k] = list(v)

        all_results = []
        
        for nb_size in neighborhood_sizes:
            exp_dir = self.base_dir / f"NB_{nb_size}"
            
            if not exp_dir.exists():
                print(f"Warning: Directory {exp_dir} does not exist. Skipping nb_size={nb_size}")
                continue
            
            # Check if metrics CSV exists
            metrics_path = exp_dir / 'biological_metrics.csv'
            
            if metrics_path.exists() and not force_recompute:
                print(f"Loading existing metrics for nb_size={nb_size}...")
                df = pd.read_csv(metrics_path)
                df['Neighborhood Size'] = nb_size
                all_results.append(df)
            else:
                print(f"Evaluating models for nb_size={nb_size}...")
                df = self._evaluate_models(nb_size, exp_dir)
                all_results.append(df)
        
        if all_results:
            self.metrics_df = pd.concat(all_results, ignore_index=True)
            # Save aggregated results
            aggregated_path = self.base_dir / 'all_neighborhood_sizes_metrics.csv'
            self.metrics_df.to_csv(aggregated_path, index=False)
            print(f"\nSaved aggregated metrics to {aggregated_path}")
        else:
            raise ValueError("No results found. Please check that models exist.")
    
    def _evaluate_models(self, nb_size: int, exp_dir: Path) -> pd.DataFrame:
        """Evaluate models for a specific neighborhood size and save raw data for statistical tests
        Only evaluates Mixture NCA and Stochastic Mixture NCA models."""
        
        def make_update_net_fn(device):
            def update_net_wrapper(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
                return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)
            return update_net_wrapper
        
        update_net_fn = make_update_net_fn(self.device)
        
        # Initialize only Mixture and Stochastic models
        mix_nca = ExtendedMixtureNCA(
            update_nets=update_net_fn,
            hidden_dim=self.HIDDEN_DIM,
            maintain_seed=False,
            use_alive_mask=False,
            state_dim=self.STATE_DIM,
            num_rules=self.N_RULES,
            residual=False,
            temperature=3,
            neighborhood_size=nb_size,
            device=self.device
        )
        
        stochastic_mix_nca = ExtendedMixtureNCANoise(
            update_nets=update_net_fn,
            hidden_dim=self.HIDDEN_DIM,
            maintain_seed=False,
            use_alive_mask=False,
            state_dim=self.STATE_DIM,
            num_rules=self.N_RULES,
            residual=False,
            temperature=3,
            neighborhood_size=nb_size,
            device=self.device
        )
        
        # Load model weights (configurable filenames)
        def _normalize_candidates(candidates: List[str]) -> List[str]:
            out: List[str] = []
            for c in candidates:
                c = str(c)
                if not c.endswith(".pt"):
                    c = f"{c}.pt"
                out.append(c)
            return out

        def load_model_file(model, label: str):
            candidates = self.model_files.get(label, [])
            if isinstance(candidates, str):
                candidates = [candidates]
            candidates = _normalize_candidates(list(candidates))

            if not candidates:
                raise ValueError(
                    f"No checkpoint candidates configured for '{label}'. "
                    f"Configure via NeighborhoodSizeAnalyzer(..., model_files={{'{label}': ['file.pt', ...]}}) "
                    f"or via load_or_evaluate_models(model_files=...)."
                )

            for fname in candidates:
                p = exp_dir / fname
                if p.exists():
                    model.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))
                    print(f"    Loaded {label}: {fname}")
                    return

            raise FileNotFoundError(
                f"No checkpoint found for '{label}' in {exp_dir}. Tried: {candidates}"
            )

        load_model_file(mix_nca, "Mixture NCA")
        load_model_file(stochastic_mix_nca, "Stochastic Mixture NCA")
        
        # Evaluate and get raw data (only Mixture and Stochastic)
        results_df, raw_data, raw_per_sim_data = self._evaluate_with_raw_data(
            mix_nca=mix_nca.to(self.device),
            stochastic_mix_nca=stochastic_mix_nca.to(self.device),
            nb_size=nb_size
        )
        
        # Save raw data for statistical tests
        raw_data_path = exp_dir / 'raw_metrics_data.pkl'
        import pickle
        with open(raw_data_path, 'wb') as f:
            pickle.dump(raw_data, f)
        print(f"Saved raw data to {raw_data_path}")

        # Save per-simulation raw data (one row per history/sample)
        raw_per_sim_path = exp_dir / 'raw_metrics_per_simulation.pkl'
        with open(raw_per_sim_path, 'wb') as f:
            pickle.dump(raw_per_sim_data, f)
        print(f"Saved per-simulation raw data to {raw_per_sim_path}")

        try:
            raw_per_sim_df = pd.DataFrame(raw_per_sim_data)
            raw_per_sim_csv = exp_dir / 'raw_metrics_per_simulation.csv'
            raw_per_sim_df.to_csv(raw_per_sim_csv, index=False)
            print(f"Saved per-simulation raw data CSV to {raw_per_sim_csv}")
        except Exception as e:
            print(f"Warning: could not save per-simulation CSV: {e}")
        
        # Save individual results
        metrics_path = exp_dir / 'biological_metrics.csv'
        results_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
        
        return results_df
    
    def _evaluate_with_raw_data(self, mix_nca, stochastic_mix_nca, nb_size: int):
        """Evaluate models and return both summary and raw data (only Mixture and Stochastic)
        
        OPTIMIZATION: Processes all histories in a single batch instead of one-by-one,
        which is orders of magnitude faster (reduces from O(N*M) to O(1) forward passes).
        """
        from mix_NCA.utils_simulations import grid_to_channels_batch
        from mix_NCA.BiologicalMetrics import BiologicalMetrics
        from mix_NCA.TissueModel import ComplexCellType
        from tqdm import tqdm
        import torch.nn.functional as F
        
        # Collect all true states and stack them into a batch
        # OPTIMIZATION: Process all histories in batch instead of individually
        initial_states_list = []
        for hist in self.histories:
            grid_state = hist[0]
            encoded_state = grid_to_channels_batch(grid_state, len(ComplexCellType), self.device)
            initial_states_list.append(encoded_state)
        
        # Stack all initial states into a single batch [N, C, H, W]
        # This allows processing all 1000 histories in one forward pass instead of 1000 separate passes
        initial_states_batch = torch.cat(initial_states_list, dim=0)
        
        # Stack all true states
        true_dataset = torch.cat([torch.tensor(ts[-1]).to(self.device).unsqueeze(0) for ts in self.histories], dim=0)
        
        # Store raw data for statistical tests (dataset-level metrics per evaluation)
        raw_data = {
            'Model Type': [],
            'Neighborhood Size': [],
            'Step Length': [],
            'Evaluation': [],
            'KL Divergence': [],
            'Chi-Square': [],
            'Categorical MMD': [],
            'Tumor Size Diff': [],
            'Border Size Diff': [],
            'Spatial Variance Diff': []
        }

        # Store raw per-simulation data (one row per history/sample)
        raw_per_sim_data = {
            'Model Type': [],
            'Neighborhood Size': [],
            'Step Length': [],
            'Evaluation': [],
            'Simulation': [],
            'KL Divergence': [],
            'Chi-Square': [],
            'Categorical MMD': [],
            'Tumor Size Diff': [],
            'Border Size Diff': [],
            'Spatial Variance Diff': [],
        }

        # --- Helpers for per-simulation metrics (operate on integer grids [N,H,W]) ---
        n_types = len(ComplexCellType)
        empty_idx = ComplexCellType.EMPTY.value

        def _per_sample_type_probs(x_int: torch.Tensor) -> torch.Tensor:
            """Return per-sample categorical distribution over cell types: [N, n_types]."""
            n = x_int.shape[0]
            flat = x_int.view(n, -1)
            counts = torch.zeros((n, n_types), device=x_int.device, dtype=torch.float32)
            for t in range(n_types):
                counts[:, t] = (flat == t).float().sum(dim=1)
            if 0 <= empty_idx < n_types:
                counts[:, empty_idx] = 0.0
            probs = counts / (counts.sum(dim=1, keepdim=True) + 1e-8)
            return probs

        def _per_sample_kl_chi(true_int: torch.Tensor, gen_int: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            p = _per_sample_type_probs(true_int)
            q = _per_sample_type_probs(gen_int)
            kl = torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)), dim=1)
            chi = torch.sum((p - q) ** 2 / (p + q + 1e-8), dim=1)
            return kl, chi

        def _per_sample_categorical_mmd_proxy(true_int: torch.Tensor, gen_int: torch.Tensor) -> torch.Tensor:
            """Per-simulation proxy for categorical MMD: MMD on batches of size 1 => 2*(1 - match_rate)."""
            tflat = true_int.view(true_int.shape[0], -1)
            gflat = gen_int.view(gen_int.shape[0], -1)
            match_rate = (tflat == gflat).float().mean(dim=1)
            return 2.0 * (1.0 - match_rate)

        def _border_size_and_spatial_variance(x_int: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute per-sample border size and spatial variance for non-empty mask."""
            mask = (x_int > 0).float()  # [N,H,W]

            # Border size via edge detection conv (batch)
            kernel = torch.tensor([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ], device=self.device).float() / 8.0
            kernel = kernel.view(1, 1, 3, 3)
            edges = (torch.abs(F.conv2d(mask.unsqueeze(1), kernel, padding=1)) > 0.25).float()
            border_size = edges.sum(dim=(1, 2, 3))  # [N]

            # Spatial variance of non-empty positions (vectorized)
            n, h, w = mask.shape
            yy = torch.arange(h, device=self.device, dtype=torch.float32).view(1, h, 1)
            xx = torch.arange(w, device=self.device, dtype=torch.float32).view(1, 1, w)
            cnt = mask.sum(dim=(1, 2)).clamp_min(1.0)
            mean_y = (mask * yy).sum(dim=(1, 2)) / cnt
            mean_x = (mask * xx).sum(dim=(1, 2)) / cnt
            dy2 = (yy - mean_y.view(n, 1, 1)) ** 2
            dx2 = (xx - mean_x.view(n, 1, 1)) ** 2
            spatial_var = (mask * (dy2 + dx2)).sum(dim=(1, 2)) / cnt
            return border_size, spatial_var

        # Precompute true per-sample quantities and normalization denominators
        true_int = true_dataset.long()
        true_tumor_sizes = (true_int > 0).float().view(true_int.shape[0], -1).sum(dim=1)
        mean_true_tumor = true_tumor_sizes.mean().item() + 1e-8
        true_border_sizes, true_spatial_vars = _border_size_and_spatial_variance(true_int)
        mean_true_border = true_border_sizes.mean().item() + 1e-8
        mean_true_spvar = true_spatial_vars.mean().item() + 1e-8
        
        # Calculate total iterations for progress bar
        total_iterations = len(self.step_lengths) * 2 * self.n_evaluations
        print(f"    Processing {len(self.histories)} histories in batch mode (much faster than individual processing)")
        print(f"    Total evaluations: {total_iterations} (across {len(self.step_lengths)} step lengths, 2 models, {self.n_evaluations} evaluations each)")
        pbar = tqdm(total=total_iterations, desc=f"Evaluating NB={nb_size}", unit="eval")
        
        # Evaluate for each step length (only Mixture and Stochastic models)
        for n_steps in self.step_lengths:
            # Evaluate Mixture and Stochastic models
            for name, model in [
                ('Mixture NCA', mix_nca),
                ('Stochastic Mixture NCA', stochastic_mix_nca)
            ]:
                for eval_idx in range(self.n_evaluations):
                    torch.manual_seed(eval_idx)
                    with torch.no_grad():
                        # Process all histories in a single batch instead of one by one
                        # This is MUCH faster than processing them individually
                        result = model(initial_states_batch, n_steps, return_history=True, sample_non_differentiable=True)
                        if isinstance(result, tuple):
                            sample = result[1][-1] if len(result[1]) > 0 else result[0]
                        else:
                            sample = result[-1]
                        # sample shape: [N, C, H, W] -> [N, H, W] after argmax
                        sample = sample.argmax(dim=1)
                        generated = sample  # Already in correct shape [N, H, W]
                    
                    bio_metrics = BiologicalMetrics(true_dataset, generated, list(ComplexCellType), self.device)
                    dist_metrics = bio_metrics.distribution_metrics()
                    spatial_metrics = bio_metrics.spatial_correlation()
                    
                    raw_data['Model Type'].append(name)
                    raw_data['Neighborhood Size'].append(nb_size)
                    raw_data['Step Length'].append(n_steps)
                    raw_data['Evaluation'].append(eval_idx)
                    raw_data['KL Divergence'].append(dist_metrics['kl_divergence'])
                    raw_data['Chi-Square'].append(dist_metrics['chi_square'])
                    raw_data['Categorical MMD'].append(dist_metrics['categorical_mmd'])
                    raw_data['Tumor Size Diff'].append(bio_metrics.tumor_size_distribution())
                    raw_data['Border Size Diff'].append(spatial_metrics['border_size_diff'])
                    raw_data['Spatial Variance Diff'].append(spatial_metrics['spatial_variance_diff'])

                    # Per-simulation metrics (one value per history/sample)
                    gen_int = generated.long()
                    kl_ps, chi_ps = _per_sample_kl_chi(true_int, gen_int)
                    mmd_ps = _per_sample_categorical_mmd_proxy(true_int, gen_int)
                    gen_tumor_sizes = (gen_int > 0).float().view(gen_int.shape[0], -1).sum(dim=1)
                    tumor_ps = (true_tumor_sizes - gen_tumor_sizes).abs() / mean_true_tumor

                    gen_border_sizes, gen_spatial_vars = _border_size_and_spatial_variance(gen_int)
                    border_ps = (true_border_sizes - gen_border_sizes).abs() / mean_true_border
                    spvar_ps = (true_spatial_vars - gen_spatial_vars).abs() / mean_true_spvar

                    n_samples = gen_int.shape[0]
                    for sim_idx in range(n_samples):
                        raw_per_sim_data['Model Type'].append(name)
                        raw_per_sim_data['Neighborhood Size'].append(nb_size)
                        raw_per_sim_data['Step Length'].append(n_steps)
                        raw_per_sim_data['Evaluation'].append(eval_idx)
                        raw_per_sim_data['Simulation'].append(sim_idx)
                        raw_per_sim_data['KL Divergence'].append(float(kl_ps[sim_idx].item()))
                        raw_per_sim_data['Chi-Square'].append(float(chi_ps[sim_idx].item()))
                        raw_per_sim_data['Categorical MMD'].append(float(mmd_ps[sim_idx].item()))
                        raw_per_sim_data['Tumor Size Diff'].append(float(tumor_ps[sim_idx].item()))
                        raw_per_sim_data['Border Size Diff'].append(float(border_ps[sim_idx].item()))
                        raw_per_sim_data['Spatial Variance Diff'].append(float(spvar_ps[sim_idx].item()))
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Create summary DataFrame (for compatibility)
        raw_df = pd.DataFrame(raw_data)
        summary_df = raw_df.groupby(['Model Type', 'Step Length']).agg({
            'KL Divergence': ['mean', 'std'],
            'Chi-Square': ['mean', 'std'],
            'Categorical MMD': ['mean', 'std'],
            'Tumor Size Diff': ['mean', 'std'],
            'Border Size Diff': ['mean', 'std'],
            'Spatial Variance Diff': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        summary_df.columns = ['Model Type'] + [f'{col[0]} {col[1].title()}' if col[1] else col[0] 
                                             for col in summary_df.columns[1:]]
        
        # Format for output (matching original format)
        results_dict = {
            'Model Type': [],
            'Step Length': [],
            'KL Divergence': [],
            'KL Divergence SD': [],
            'Chi-Square': [],
            'Chi-Square SD': [],
            'Categorical MMD': [],
            'Categorical MMD SD': [],
            'Tumor Size Diff': [],
            'Tumor Size Diff SD': [],
            'Border Size Diff': [],
            'Border Size Diff SD': [],
            'Spatial Variance Diff': [],
            'Spatial Variance Diff SD': []
        }
        
        for model_type in raw_df['Model Type'].unique():
            for step_length in raw_df['Step Length'].unique():
                model_data = raw_df[(raw_df['Model Type'] == model_type) & 
                                   (raw_df['Step Length'] == step_length)]
                if len(model_data) == 0:
                    continue
                results_dict['Model Type'].append(model_type)
                results_dict['Step Length'].append(step_length)
                for metric in ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
                              'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']:
                    mean_val = model_data[metric].mean()
                    std_val = model_data[metric].std()
                    results_dict[metric].append(f"{mean_val:.3f}")
                    results_dict[f"{metric} SD"].append(f"±{std_val:.3f}")
        
        results_df = pd.DataFrame(results_dict)
        results_df['Neighborhood Size'] = nb_size
        
        return results_df, raw_data, raw_per_sim_data
    
    def parse_metrics(self) -> pd.DataFrame:
        """Parse metrics from string format to numeric values"""
        if not hasattr(self, 'metrics_df'):
            # Load from CSV if not already loaded
            aggregated_path = self.base_dir / 'all_neighborhood_sizes_metrics.csv'
            if aggregated_path.exists():
                self.metrics_df = pd.read_csv(aggregated_path)
            else:
                raise ValueError("No metrics data available. Please run load_or_evaluate_models() first.")
        df = self.metrics_df.copy()
        
        # Columns to parse (remove SD columns)
        metric_cols = ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
                      'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']
        
        for col in metric_cols:
            if col in df.columns:
                # Convert string to float (handles "±0.000" format)
                df[col] = df[col].astype(str).str.replace('±.*', '', regex=True).astype(float)
        
        return df
    
    def performance_trend_analysis(self) -> pd.DataFrame:
        """Analyze performance trends across neighborhood sizes"""
        print(f"\n{'='*60}")
        print("Performance Trend Analysis")
        print(f"{'='*60}\n")
        
        df = self.parse_metrics()
        
        # Group by model type and neighborhood size
        metric_cols = ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
                      'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']
        
        trend_results = []
        
        for model_type in df['Model Type'].unique():
            model_data = df[df['Model Type'] == model_type]
            
            for metric in metric_cols:
                if metric not in model_data.columns:
                    continue
                
                # Compute mean and std for each neighborhood size
                grouped = model_data.groupby('Neighborhood Size')[metric].agg(['mean', 'std', 'count'])
                
                # Compute correlation with neighborhood size
                sizes = grouped.index.values
                means = grouped['mean'].values
                
                if len(sizes) > 1:
                    # Check if all values are constant (zero variance) - use multiple checks for robustness
                    std_means = np.std(means)
                    unique_means = len(np.unique(means))
                    range_means = np.max(means) - np.min(means)
                    mean_abs = np.abs(np.mean(means))
                    
                    # Check: std is zero, only one unique value, or range is effectively zero
                    is_constant = (std_means < 1e-10 or 
                                  unique_means == 1 or 
                                  (mean_abs > 0 and range_means / mean_abs < 1e-10) or
                                  range_means < 1e-10)
                    
                    if is_constant:
                        # All values are identical - correlation is undefined
                        pearson_r, pearson_p = np.nan, np.nan
                        spearman_r, spearman_p = np.nan, np.nan
                        slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
                    else:
                        # Suppress warnings for constant input (shouldn't happen after check, but just in case)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='.*constant.*correlation.*')
                            warnings.filterwarnings('ignore', message='.*An input array is constant.*')
                            warnings.filterwarnings('ignore', message='.*Precision loss.*')
                            warnings.filterwarnings('ignore', message='.*invalid value.*')
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            warnings.filterwarnings('ignore', category=UserWarning)
                            # Pearson correlation
                            pearson_r, pearson_p = stats.pearsonr(sizes, means)
                            
                            # Spearman correlation (monotonic relationship)
                            spearman_r, spearman_p = stats.spearmanr(sizes, means)
                            
                            # Linear regression slope
                            slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, means)
                    
                    # Calculate improvement metrics
                    mean_nb3 = grouped.loc[3, 'mean'] if 3 in sizes else np.nan
                    mean_nb7 = grouped.loc[7, 'mean'] if 7 in sizes else np.nan
                    
                    if 3 in sizes and 7 in sizes and not np.isnan(mean_nb3) and mean_nb3 != 0:
                        improvement_pct = (mean_nb3 - mean_nb7) / mean_nb3 * 100
                        fold_change = mean_nb7 / mean_nb3 if mean_nb3 != 0 else np.nan
                        abs_diff = mean_nb7 - mean_nb3
                    else:
                        improvement_pct = np.nan
                        fold_change = np.nan
                        abs_diff = np.nan
                    
                    trend_results.append({
                        'Model Type': model_type,
                        'Metric': metric,
                        'Mean_NB3': mean_nb3,
                        'Mean_NB7': mean_nb7,
                        'Improvement_3_to_7': improvement_pct,
                        'Fold_Change_NB7_vs_NB3': fold_change,  # NB7/NB3 ratio (>1 means worse if lower is better)
                        'Absolute_Diff_NB7_vs_NB3': abs_diff,  # NB7 - NB3
                        'Pearson_r': pearson_r,
                        'Pearson_p': pearson_p,
                        'Spearman_r': spearman_r,
                        'Spearman_p': spearman_p,
                        'Slope': slope,
                        'Slope_p': p_value,
                        'Best_NB': sizes[np.argmin(means)],
                        'Worst_NB': sizes[np.argmax(means)]
                    })
                    
                    print(f"{model_type} - {metric}:")
                    if np.isnan(pearson_r):
                        print(f"  Correlation (Pearson): undefined (constant values)")
                        print(f"  Correlation (Spearman): undefined (constant values)")
                        print(f"  Linear trend: undefined (constant values)")
                    else:
                        print(f"  Correlation (Pearson): r={pearson_r:.4f}, p={pearson_p:.4f}")
                        print(f"  Correlation (Spearman): r={spearman_r:.4f}, p={spearman_p:.4f}")
                        print(f"  Linear trend: slope={slope:.6f}, p={p_value:.4f}")
                    print(f"  Best NB: {sizes[np.argmin(means)]}, Worst NB: {sizes[np.argmax(means)]}")
                    if 3 in sizes and 7 in sizes and not np.isnan(improvement_pct):
                        print(f"  Improvement NB3→NB7: {improvement_pct:.2f}%")
                        if not np.isnan(fold_change):
                            print(f"  Fold Change (NB7/NB3): {fold_change:.3f}x")
                        if not np.isnan(abs_diff):
                            print(f"  Absolute Difference (NB7-NB3): {abs_diff:.4f}")
                    print()
        
        return pd.DataFrame(trend_results)
    
    def computational_complexity_analysis(
        self,
        n_samples: int = 5,
        model_type: str = "mixture",
        checkpoint_filenames: Optional[object] = None,
    ) -> pd.DataFrame:
        """
        Measure computational time for different neighborhood sizes
        
        Args:
            n_samples: Number of samples to average over
            model_type: Type of model to analyze. Options:
                - "mixture": Mixture NCA
                - "stochastic": Stochastic Mixture NCA
            checkpoint_filenames: Optional override for which checkpoint file(s) to load
                inside each NB_k folder.
                - If None (default): uses the standard names and tries multiple fallbacks.
                - If str: tries that single filename (e.g., "mixture_nca.pt").
                - If list[str]: tries the provided filenames in order.
        """
        print(f"\n{'='*60}")
        print(f"Computational Complexity Analysis - {model_type.upper()}")
        print(f"{'='*60}\n")
        
        # Map model type to checkpoint filenames and model class (only Mixture and Stochastic)
        model_configs = {
            "mixture": {
                "files": ["mixture_nca_1000.pt", "mixture_nca.pt"],
                "model_class": ExtendedMixtureNCA,
                "init_kwargs": {
                    "update_nets": lambda n_channels, hidden_dims=128, n_channels_out=None, device_arg=None: 
                        classification_update_net(n_channels, hidden_dims, n_channels_out, device=self.device),
                    "num_rules": self.N_RULES,
                    "hidden_dim": self.HIDDEN_DIM,
                    "maintain_seed": False,
                    "use_alive_mask": False,
                    "state_dim": self.STATE_DIM,
                    "residual": False,
                    "temperature": 5
                }
            },
            "stochastic": {
                "files": ["stochastic_mix_nca_1000.pt", "stochastic_mix_nca.pt"],
                "model_class": ExtendedMixtureNCANoise,
                "init_kwargs": {
                    "update_nets": lambda n_channels, hidden_dims=128, n_channels_out=None, device_arg=None: 
                        classification_update_net(n_channels, hidden_dims, n_channels_out, device=self.device),
                    "num_rules": self.N_RULES,
                    "hidden_dim": self.HIDDEN_DIM,
                    "maintain_seed": False,
                    "use_alive_mask": False,
                    "state_dim": self.STATE_DIM,
                    "residual": False,
                    "temperature": 3
                }
            }
        }
        
        if model_type not in model_configs:
            raise ValueError(f"Unknown model_type: {model_type}. Must be one of {list(model_configs.keys())}")
        
        config = model_configs[model_type]
        
        complexity_results = []
        
        def make_update_net_fn(device):
            def update_net_wrapper(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
                return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)
            return update_net_wrapper
        
        update_net_fn = make_update_net_fn(self.device)
        
        # Get a sample initial state
        initial_state = grid_to_channels_batch([self.histories[0][0]], len(ComplexCellType), self.device)
        
        # Store baseline time (NB=3) for normalization
        baseline_time = None
        
        # Normalize checkpoint_filenames -> list[str] (or None)
        if checkpoint_filenames is None:
            checkpoint_candidates = None
        elif isinstance(checkpoint_filenames, str):
            checkpoint_candidates = [checkpoint_filenames]
        else:
            # best-effort: allow any iterable of strings (e.g., list/tuple)
            checkpoint_candidates = list(checkpoint_filenames)

        for nb_size in [1, 2, 3, 4, 5, 6, 7]:
            exp_dir = self.base_dir / f"NB_{nb_size}"
            if not exp_dir.exists():
                continue
            
            # Resolve checkpoint path:
            # - default: try standard filenames for this model_type
            # - override: try user-provided filename(s) in order
            candidate_names = checkpoint_candidates if checkpoint_candidates is not None else config["files"]
            model_file = None
            for fname in candidate_names:
                p = exp_dir / fname
                if p.exists():
                    model_file = p
                    break
            if model_file is None:
                print(f"  Skipping NB_{nb_size}: none of these checkpoints found: {candidate_names}")
                continue
            
            print(f"Testing NB_{nb_size} ({model_type})...")
            
            # Initialize model with correct type
            init_kwargs = config["init_kwargs"].copy()
            init_kwargs["neighborhood_size"] = nb_size
            init_kwargs["device"] = self.device
            
            # Handle update_nets for mixture models
            if model_type in ["mixture", "stochastic"]:
                init_kwargs["update_nets"] = update_net_fn
            
            model = config["model_class"](**init_kwargs)
            
            # Post-initialization setup (e.g., for nca_with_noise)
            if "post_init" in config:
                config["post_init"](model)
            
            # Load weights
            model.load_state_dict(torch.load(model_file, map_location=self.device, weights_only=True))
            model = model.to(self.device)
            model.eval()
            
            # Warm-up
            with torch.no_grad():
                _ = model(initial_state, 5, return_history=False)
            
            # Measure time
            times = []
            n_steps = 35
            
            for _ in range(n_samples):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.time()
                with torch.no_grad():
                    _ = model(initial_state, n_steps, return_history=False)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            # Theoretical complexity: O(nb_size^2) for 2D convolution
            # Explanation: For each pixel in the grid (H×W pixels), we perform a convolution
            # with a kernel of size nb_size × nb_size. The number of operations per pixel
            # is proportional to nb_size², so total complexity is O(H × W × nb_size²).
            # For fixed grid size, this simplifies to O(nb_size²).
            theoretical_ops = nb_size ** 2
            
            # Store baseline time from NB=3 for normalization
            if nb_size == 3:
                baseline_time = mean_time
            
            # Normalize time relative to NB=3 baseline (if available)
            if baseline_time is not None:
                normalized_time = mean_time / baseline_time
            else:
                normalized_time = np.nan
            
            complexity_results.append({
                'Model Type': model_type,
                'Neighborhood Size': nb_size,
                'Mean Time (s)': mean_time,
                'Std Time (s)': std_time,
                'Time per Step (ms)': mean_time / n_steps * 1000,
                'Theoretical O(n²)': theoretical_ops,
                'Normalized Time': normalized_time  # Normalized to NB=3 actual time
            })
            
            print(f"  Mean time: {mean_time:.4f} ± {std_time:.4f} s")
            print(f"  Time per step: {mean_time/n_steps*1000:.2f} ms")
            print(f"  Theoretical complexity factor: {theoretical_ops / (3**2):.2f}x")
            if baseline_time is not None and nb_size > 3:
                print(f"  Actual speedup vs NB=3: {baseline_time/mean_time:.2f}x")
            print()
        
        return pd.DataFrame(complexity_results)
    
    def create_visualizations(self, output_dir: Optional[str] = None):
        """Create comprehensive visualizations using Plotly (only for Mixture NCA and Stochastic Mixture NCA)"""
        if output_dir is None:
            output_dir = self.base_dir / "analysis_plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Creating Visualizations")
        print(f"{'='*60}\n")
        
        df = self.parse_metrics()
        
        # Filter to show only Mixture NCA and Stochastic Mixture NCA
        target_models = ['Mixture NCA', 'Stochastic Mixture NCA']
        df = df[df['Model Type'].isin(target_models)]
        
        metric_cols = ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
                      'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']
        
        # 1. Box plots for each metric
        for metric in metric_cols:
            if metric not in df.columns:
                continue
            
            fig = px.box(df, x='Neighborhood Size', y=metric, color='Model Type',
                        title=f'{metric} by Neighborhood Size and Model Type',
                        labels={'Neighborhood Size': 'Neighborhood Size', metric: metric})
            
            fig.update_layout(
                width=1200,
                height=700,
                font=dict(size=12),
                title_font_size=16,
                legend=dict(title='Model Type', yanchor="top", y=0.99, xanchor="left", x=1.01)
            )
            
            filename = output_dir / f'{metric.replace(" ", "_").lower()}_boxplot.html'
            fig.write_html(str(filename))
            print(f"Saved: {filename}")
        
        # 2. Line plots showing trends with confidence bands
        for metric in metric_cols:
            if metric not in df.columns:
                continue
            
            fig = go.Figure()
            
            # Color palette for models
            colors = px.colors.qualitative.Set2
            
            for idx, model_type in enumerate(df['Model Type'].unique()):
                model_data = df[df['Model Type'] == model_type]
                grouped = model_data.groupby('Neighborhood Size')[metric].agg(['mean', 'std', 'count'])
                
                sizes = grouped.index.values
                means = grouped['mean'].values
                stds = grouped['std'].values
                
                color = colors[idx % len(colors)]
                
                # Add mean line
                fig.add_trace(go.Scatter(
                    x=sizes,
                    y=means,
                    mode='lines+markers',
                    name=model_type,
                    line=dict(color=color, width=3),
                    marker=dict(size=10, color=color),
                    error_y=dict(type='data', array=stds, visible=True)
                ))
                
                # Add confidence band
                fig.add_trace(go.Scatter(
                    x=np.concatenate([sizes, sizes[::-1]]),
                    y=np.concatenate([means + stds, (means - stds)[::-1]]),
                    fill='toself',
                    fillcolor=color,
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo="skip",
                    opacity=0.2
                ))
            
            fig.update_layout(
                title=f'{metric} Trends Across Neighborhood Sizes',
                xaxis_title='Neighborhood Size',
                yaxis_title=metric,
                width=1200,
                height=700,
                font=dict(size=12),
                title_font_size=16,
                hovermode='x unified',
                template='plotly_white'
            )
            
            filename = output_dir / f'{metric.replace(" ", "_").lower()}_trend.html'
            fig.write_html(str(filename))
            print(f"Saved: {filename}")
        
        # 3. Heatmap of performance
        for model_type in df['Model Type'].unique():
            model_data = df[df['Model Type'] == model_type]
            
            # For multiple metrics, create a combined heatmap
            heatmap_data = []
            available_metrics = []
            neighborhood_sizes = sorted(model_data['Neighborhood Size'].unique())
            
            for metric in metric_cols:
                if metric not in model_data.columns:
                    continue
                grouped = model_data.groupby('Neighborhood Size')[metric].mean()
                heatmap_data.append(grouped.values)
                available_metrics.append(metric)
            
            if heatmap_data:
                heatmap_array = np.array(heatmap_data)
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_array,
                    x=[f'NB_{nb}' for nb in neighborhood_sizes],
                    y=available_metrics,
                    colorscale='RdYlGn_r',
                    text=[[f'{val:.3f}' for val in row] for row in heatmap_array],
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title=dict(text="Metric Value<br>(lower is better)", font=dict(size=12)))
                ))
                
                fig.update_layout(
                    title=f'Performance Heatmap: {model_type}',
                    xaxis_title='Neighborhood Size',
                    yaxis_title='Metric',
                    width=900,
                    height=600,
                    font=dict(size=12),
                    title_font_size=16
                )
                
                filename = output_dir / f'{model_type.replace(" ", "_").lower()}_heatmap.html'
                fig.write_html(str(filename))
                print(f"Saved: {filename}")
        
        # 4. Interactive dashboard with all metrics
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metric_cols,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set2
        
        for idx, metric in enumerate(metric_cols):
            if metric not in df.columns:
                continue
            
            row = (idx // 3) + 1
            col = (idx % 3) + 1
            
            for model_idx, model_type in enumerate(df['Model Type'].unique()):
                model_data = df[df['Model Type'] == model_type]
                grouped = model_data.groupby('Neighborhood Size')[metric].agg(['mean', 'std'])
                
                sizes = grouped.index.values
                means = grouped['mean'].values
                stds = grouped['std'].values
                
                color = colors[model_idx % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=means,
                        mode='lines+markers',
                        name=model_type if idx == 0 else '',
                        line=dict(color=color, width=2),
                        marker=dict(size=8, color=color),
                        error_y=dict(type='data', array=stds, visible=True),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="Comprehensive Performance Dashboard",
            height=1000,
            width=1800,
            font=dict(size=10),
            title_font_size=18,
            template='plotly_white'
        )
        
        filename = output_dir / 'comprehensive_dashboard.html'
        fig.write_html(str(filename))
        print(f"Saved: {filename}")
        
        print(f"\nAll visualizations saved to {output_dir}")
    
    def _calculate_effect_size_mannwhitney(self, data1: np.ndarray, data2: np.ndarray, 
                                            u_stat: float) -> float:
        """
        Calculate r effect size for Mann-Whitney U test
        
        Uses a more robust formula that accounts for ties and complete separation.
        For complete separation, uses rank-biserial correlation as alternative.
        
        r = Z / sqrt(N) where Z is derived from U statistic
        Interpretation:
        - |r| < 0.1: negligible
        - 0.1 <= |r| < 0.3: small
        - 0.3 <= |r| < 0.5: medium
        - |r| >= 0.5: large
        
        Args:
            data1: First group data
            data2: Second group data
            u_stat: Mann-Whitney U statistic
            
        Returns:
            r effect size
        """
        n1, n2 = len(data1), len(data2)
        n = n1 + n2
        
        # Check for complete separation
        complete_separation = np.all(data1 < data2) or np.all(data1 > data2) or np.all(data2 < data1) or np.all(data2 > data1)
        
        if complete_separation:
            # With complete separation, all values in one group are > or < all in the other
            # Use rank-biserial correlation: r_rb = 1 - (2*U) / (n1*n2)
            # This gives r_rb = 1.0 or -1.0 for complete separation
            max_u = n1 * n2
            r_rb = 1.0 - (2.0 * u_stat) / max_u
            # Convert to absolute value for effect size interpretation
            r = abs(r_rb)
            # Note: With complete separation, r will always be 1.0 (maximum effect)
            # This is mathematically correct but may appear identical across comparisons
        else:
            # Standard calculation with correction for ties
            # Calculate expected U and standard deviation
            u_expected = n1 * n2 / 2.0
            
            # Account for ties in standard deviation calculation
            all_data = np.concatenate([data1, data2])
            unique_vals, counts = np.unique(all_data, return_counts=True)
            tie_correction = np.sum(counts * (counts**2 - 1)) if len(unique_vals) < n else 0
            
            if tie_correction > 0:
                u_std = np.sqrt((n1 * n2 / (n * (n - 1))) * ((n**3 - n - tie_correction) / 12.0))
            else:
                u_std = np.sqrt(n1 * n2 * (n + 1) / 12.0)
            
            # Calculate Z score
            if u_std > 0:
                z = (u_stat - u_expected) / u_std
            else:
                z = 0.0
            
            # Calculate r effect size
            r = abs(z) / np.sqrt(n)
        
        return r
    
    def _calculate_effect_size_kruskal(self, h_stat: float, n_total: int, k_groups: int) -> float:
        """
        Calculate eta-squared effect size for Kruskal-Wallis test
        
        eta² = (H - k + 1) / (N - k)
        
        Interpretation:
        - eta² < 0.01: negligible
        - 0.01 <= eta² < 0.06: small
        - 0.06 <= eta² < 0.14: medium
        - eta² >= 0.14: large
        
        Args:
            h_stat: Kruskal-Wallis H statistic
            n_total: Total number of observations
            k_groups: Number of groups
            
        Returns:
            eta-squared effect size
        """
        if n_total <= k_groups:
            return 0.0
        
        eta_squared = (h_stat - k_groups + 1) / (n_total - k_groups)
        return max(0.0, eta_squared)  # Ensure non-negative
    
    def statistical_significance_tests(self, alpha: float = 0.05, 
                                       correction_method: str = 'bonferroni',
                                       include_effect_size: bool = True) -> pd.DataFrame:
        """
        Perform statistical significance tests to compare neighborhood sizes
        
        Args:
            alpha: Significance level (default 0.05)
            correction_method: Multiple comparison correction method ('bonferroni' or 'fdr_bh')
        
        Returns:
            DataFrame with test results
        """
        print(f"\n{'='*60}")
        print("Statistical Significance Tests")
        print(f"{'='*60}\n")
        print(f"Significance level (alpha): {alpha}")
        print(f"Multiple comparison correction: {correction_method}")
        if correction_method == 'bonferroni':
            print(f"Note: With 5 groups, there are 10 pairwise comparisons.")
            print(f"Bonferroni corrected alpha = {alpha}/10 = {alpha/10:.6f}")
        print()
        
        # Load raw data from pickle files
        raw_data_list = []
        for nb_size in [1, 2, 3, 4, 5, 6, 7]:
            exp_dir = self.base_dir / f"NB_{nb_size}"
            raw_data_path = exp_dir / 'raw_metrics_data.pkl'
            if raw_data_path.exists():
                import pickle
                with open(raw_data_path, 'rb') as f:
                    raw_data = pickle.load(f)
                    raw_data_list.append(pd.DataFrame(raw_data))
        
        if not raw_data_list:
            print("Warning: No raw data found. Please run load_or_evaluate_models() first.")
            return pd.DataFrame()
        
        raw_df = pd.concat(raw_data_list, ignore_index=True)
        
        metric_cols = ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
                      'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']
        
        test_results = []
        
        for model_type in raw_df['Model Type'].unique():
            model_data = raw_df[raw_df['Model Type'] == model_type]
            
            for metric in metric_cols:
                if metric not in model_data.columns:
                    continue
                
                # Group by neighborhood size
                groups = []
                nb_sizes = []
                for nb_size in sorted(model_data['Neighborhood Size'].unique()):
                    group_data = model_data[model_data['Neighborhood Size'] == nb_size][metric].values
                    # Remove NaN values
                    group_data = group_data[~np.isnan(group_data)]
                    if len(group_data) > 0:
                        groups.append(group_data)
                        nb_sizes.append(nb_size)
                
                if len(groups) < 2:
                    continue
                
                # Check for zero variance (all values identical within groups)
                zero_var_groups = []
                for idx, (nb_size, group_data) in enumerate(zip(nb_sizes, groups)):
                    if np.var(group_data) == 0:
                        zero_var_groups.append(nb_size)
                
                if zero_var_groups:
                    print(f"  Warning: Zero variance detected for {model_type} - {metric} in groups: {zero_var_groups}")
                    print(f"    This may indicate deterministic behavior or insufficient variability.")
                
                    # Kruskal-Wallis test (non-parametric ANOVA)
                try:
                    h_stat, p_value_kw = kruskal(*groups)
                    
                    # Calculate effect size for Kruskal-Wallis
                    n_total = sum(len(g) for g in groups)
                    k_groups = len(groups)
                    eta_squared = self._calculate_effect_size_kruskal(h_stat, n_total, k_groups) if include_effect_size else None
                    
                    # Significance stars
                    if p_value_kw < 0.001:
                        sig_stars = "***"
                    elif p_value_kw < 0.01:
                        sig_stars = "**"
                    elif p_value_kw < 0.05:
                        sig_stars = "*"
                    else:
                        sig_stars = ""
                    
                    result_dict = {
                        'Model Type': model_type,
                        'Metric': metric,
                        'Test': 'Kruskal-Wallis',
                        'H_statistic': h_stat,
                        'p_value': p_value_kw,
                        'Significant': p_value_kw < alpha,
                        'Significance': sig_stars
                    }
                    
                    if include_effect_size:
                        result_dict['Effect_Size'] = eta_squared
                        result_dict['Effect_Size_Type'] = 'eta-squared'
                    
                    test_results.append(result_dict)
                    
                    # Post-hoc pairwise comparisons (Mann-Whitney U)
                    if p_value_kw < alpha and len(groups) > 2:
                        n_comparisons = len(list(combinations(range(len(nb_sizes)), 2)))
                        alpha_corrected = alpha / n_comparisons if correction_method == 'bonferroni' else alpha
                        
                        for i, j in combinations(range(len(nb_sizes)), 2):
                            try:
                                u_stat, p_value_mw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                                
                                # Check for complete separation
                                data_i, data_j = groups[i], groups[j]
                                complete_separation = False
                                if np.all(data_i < data_j) or np.all(data_i > data_j):
                                    complete_separation = True
                                
                                # Apply correction
                                if correction_method == 'bonferroni':
                                    p_value_corrected = min(p_value_mw * n_comparisons, 1.0)
                                else:
                                    p_value_corrected = p_value_mw
                                
                                if p_value_corrected < 0.001:
                                    sig_stars = "***"
                                elif p_value_corrected < 0.01:
                                    sig_stars = "**"
                                elif p_value_corrected < 0.05:
                                    sig_stars = "*"
                                else:
                                    sig_stars = ""
                                
                                # Use alpha_corrected for significance determination (BUG FIX)
                                is_significant = p_value_corrected < alpha_corrected
                                
                                # Calculate effect size for Mann-Whitney U
                                r_effect = self._calculate_effect_size_mannwhitney(data_i, data_j, u_stat) if include_effect_size else None
                                
                                # For complete separation, also calculate mean difference for more informative comparison
                                mean_diff = None
                                if complete_separation and include_effect_size:
                                    mean_diff = abs(np.mean(data_i) - np.mean(data_j))
                                
                                result_dict = {
                                    'Model Type': model_type,
                                    'Metric': metric,
                                    'Test': f'Mann-Whitney U (NB{nb_sizes[i]} vs NB{nb_sizes[j]})',
                                    'H_statistic': u_stat,
                                    'p_value': p_value_corrected,
                                    'Significant': is_significant,
                                    'Significance': sig_stars
                                }
                                
                                if include_effect_size:
                                    result_dict['Effect_Size'] = r_effect
                                    result_dict['Effect_Size_Type'] = 'r'
                                    if mean_diff is not None:
                                        result_dict['Mean_Diff'] = mean_diff
                                
                                test_results.append(result_dict)
                                
                                # Warn about complete separation
                                if complete_separation:
                                    direction = ">" if np.all(data_i > data_j) else "<"
                                    mean_i, mean_j = np.mean(data_i), np.mean(data_j)
                                    print(f"  Note: Complete separation in {model_type} - {metric} (NB{nb_sizes[i]} {direction} NB{nb_sizes[j]}, "
                                          f"mean diff={abs(mean_i-mean_j):.4f})")
                                    
                            except Exception as e:
                                print(f"  Warning: Could not perform post-hoc test for {model_type} - {metric} (NB{nb_sizes[i]} vs NB{nb_sizes[j]}): {e}")
                
                except Exception as e:
                    print(f"  Warning: Could not perform Kruskal-Wallis test for {model_type} - {metric}: {e}")
        
        results_df = pd.DataFrame(test_results)
        
        # Print summary
        print("Kruskal-Wallis Tests (Overall differences):")
        print("-" * 60)
        kw_results = results_df[results_df['Test'] == 'Kruskal-Wallis']
        for _, row in kw_results.iterrows():
            print(f"{row['Model Type']} - {row['Metric']}:")
            print(f"  H={row['H_statistic']:.4f}, p={row['p_value']:.6f} {row['Significance']}")
            print(f"  Significant: {'Yes' if row['Significant'] else 'No'}")
            if include_effect_size and 'Effect_Size' in row and pd.notna(row['Effect_Size']):
                eta_sq = row['Effect_Size']
                # Interpret effect size
                if eta_sq >= 0.14:
                    effect_interp = "large"
                elif eta_sq >= 0.06:
                    effect_interp = "medium"
                elif eta_sq >= 0.01:
                    effect_interp = "small"
                else:
                    effect_interp = "negligible"
                print(f"  Effect Size (η²)={eta_sq:.4f} ({effect_interp})")
            print()
        
        print("\nPost-hoc Pairwise Comparisons (Mann-Whitney U):")
        print("-" * 60)
        posthoc_results = results_df[results_df['Test'] != 'Kruskal-Wallis']
        if len(posthoc_results) > 0:
            significant_count = 0
            for _, row in posthoc_results.iterrows():
                if row['Significant']:
                    significant_count += 1
                    print(f"{row['Model Type']} - {row['Metric']}: {row['Test']}")
                    print(f"  U={row['H_statistic']:.4f}, p={row['p_value']:.6f} {row['Significance']}")
                    if include_effect_size and 'Effect_Size' in row and pd.notna(row['Effect_Size']):
                        r_eff = row['Effect_Size']
                        # Interpret effect size
                        if r_eff >= 0.5:
                            effect_interp = "large"
                        elif r_eff >= 0.3:
                            effect_interp = "medium"
                        elif r_eff >= 0.1:
                            effect_interp = "small"
                        else:
                            effect_interp = "negligible"
                        
                        effect_str = f"  Effect Size (r)={r_eff:.4f} ({effect_interp})"
                        
                        # Add note if effect size is 1.0 (complete separation)
                        if r_eff >= 0.99:
                            effect_str += " [complete separation - max effect]"
                        
                        # Add mean difference if available (more informative for complete separation)
                        if 'Mean_Diff' in row and pd.notna(row['Mean_Diff']):
                            effect_str += f", Mean Diff={row['Mean_Diff']:.4f}"
                        
                        print(effect_str)
                    print()
            
            if significant_count == 0:
                print("No significant pairwise differences found (after Bonferroni correction).")
            else:
                print(f"Total significant comparisons: {significant_count} out of {len(posthoc_results)}")
        else:
            print("No pairwise comparisons performed.")
        
        return results_df


def main():
    """Main analysis pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze neighborhood size impact on NCA models')
    parser.add_argument('--results_dir', type=str, default='results_extended',
                       help='Directory containing trained models')
    parser.add_argument('--histories_path', type=str, default='../histories.npy',
                       help='Path to histories.npy file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Computing device (auto, cuda, mps, or cpu)')
    parser.add_argument('--n_evaluations', type=int, default=10,
                       help='Number of evaluations for stochastic models')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recomputation of metrics even if CSV exists')
    parser.add_argument('--neighborhood_sizes', type=str, default='1,2,3,4,5,6,7',
                       help='Comma-separated list of neighborhood sizes to analyze')
    parser.add_argument('--step_lengths', type=str, default='35,100,500',
                       help='Comma-separated list of step lengths to test')
    parser.add_argument('--skip_plots', action='store_true',
                       help='Skip generating visualizations')
    
    args = parser.parse_args()
    
    neighborhood_sizes = [int(s.strip()) for s in args.neighborhood_sizes.split(',') if s.strip()]
    step_lengths = [int(s.strip()) for s in args.step_lengths.split(',') if s.strip()]
    
    # Initialize analyzer
    analyzer = NeighborhoodSizeAnalyzer(
        results_dir=args.results_dir,
        histories_path=args.histories_path,
        device=args.device,
        n_evaluations=args.n_evaluations,
        step_lengths=step_lengths
    )
    
    # Load or evaluate models
    analyzer.load_or_evaluate_models(
        neighborhood_sizes=neighborhood_sizes,
        force_recompute=args.force_recompute
    )
    
    # Perform analyses
    print("\n" + "="*60)
    print("PERFORMING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Trend analysis
    trend_df = analyzer.performance_trend_analysis()
    trend_path = analyzer.base_dir / "performance_trends.csv"
    trend_df.to_csv(trend_path, index=False)
    print(f"\nSaved trend analysis to {trend_path}")
    
    # Computational complexity
    complexity_df = analyzer.computational_complexity_analysis()
    complexity_path = analyzer.base_dir / "computational_complexity.csv"
    complexity_df.to_csv(complexity_path, index=False)
    print(f"Saved complexity analysis to {complexity_path}")
    
    # Visualizations
    if not args.skip_plots:
        analyzer.create_visualizations()
    
    
    print("\n" + "="*60)
    print("Analysis completed")
    print("="*60)
    print(f"\nResults saved in: {analyzer.base_dir}")
    print("\nKey files generated:")
    print(f"  - all_neighborhood_sizes_metrics.csv: Aggregated metrics")
    print(f"  - performance_trends.csv: Trend analysis")
    print(f"  - computational_complexity.csv: Complexity analysis")
    print(f"  - neighborhood_size_analysis_report.txt: Comprehensive report")
    if not args.skip_plots:
        print(f"  - analysis_plots/: Visualizations")


if __name__ == "__main__":
    main()

