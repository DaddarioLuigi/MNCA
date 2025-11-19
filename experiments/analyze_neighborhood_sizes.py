"""
Scientific Analysis of Neighborhood Size Impact on NCA Models

This script performs a comprehensive scientific analysis comparing NCA models
trained with different neighborhood sizes (3, 4, 5, 6, 7).

Analysis includes:
1. Model evaluation and metric computation
2. Statistical significance testing
3. Performance trend analysis
4. Computational complexity analysis
5. Model-specific comparisons
6. Visualization of results
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
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare
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
                 device: str = "auto", n_evaluations: int = 10):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory containing trained models (e.g., 'results_extended')
            histories_path: Path to histories.npy file
            device: Computing device
            n_evaluations: Number of evaluations for stochastic models
        """
        self.results_dir = Path(results_dir)
        self.histories_path = histories_path
        self.device = get_device(device)
        self.n_evaluations = n_evaluations
        self.base_dir = self.results_dir / "tissue_simulation_extended"
        
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
        
    def load_or_evaluate_models(self, neighborhood_sizes: List[int] = [3, 4, 5, 6, 7],
                                force_recompute: bool = False):
        """
        Load existing metrics or evaluate models to compute metrics
        
        Args:
            neighborhood_sizes: List of neighborhood sizes to analyze
            force_recompute: If True, recompute metrics even if CSV exists
        """
        print(f"\n{'='*60}")
        print("Loading/Evaluating Models")
        print(f"{'='*60}\n")
        
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
        """Evaluate models for a specific neighborhood size and save raw data for statistical tests"""
        
        def make_update_net_fn(device):
            def update_net_wrapper(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
                return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)
            return update_net_wrapper
        
        update_net_fn = make_update_net_fn(self.device)
        
        # Initialize models
        nca = ExtendedNCA(
            update_net=classification_update_net(6 * 3, n_channels_out=6, device=self.device),
            hidden_dim=self.HIDDEN_DIM,
            maintain_seed=False,
            use_alive_mask=False,
            state_dim=self.STATE_DIM,
            residual=False,
            neighborhood_size=nb_size,
            device=self.device
        )
        
        nca_with_noise = ExtendedNCA(
            update_net=classification_update_net(6 * 3, n_channels_out=6 * 2, device=self.device),
            hidden_dim=self.HIDDEN_DIM,
            maintain_seed=False,
            use_alive_mask=False,
            state_dim=self.STATE_DIM,
            residual=False,
            distribution="normal",
            neighborhood_size=nb_size,
            device=self.device
        )
        nca_with_noise.random_updates = True
        
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
        
        # Load model weights (map to current device)
        nca.load_state_dict(torch.load(exp_dir / 'standard_nca.pt', map_location=self.device, weights_only=True))
        nca_with_noise.load_state_dict(torch.load(exp_dir / 'nca_with_noise.pt', map_location=self.device, weights_only=True))
        mix_nca.load_state_dict(torch.load(exp_dir / 'mixture_nca.pt', map_location=self.device, weights_only=True))
        stochastic_mix_nca.load_state_dict(torch.load(exp_dir / 'stochastic_mix_nca.pt', map_location=self.device, weights_only=True))
        
        # Evaluate and get raw data
        results_df, raw_data = self._evaluate_with_raw_data(
            nca=nca.to(self.device),
            mix_nca=mix_nca.to(self.device),
            stochastic_mix_nca=stochastic_mix_nca.to(self.device),
            nca_with_noise=nca_with_noise.to(self.device),
            nb_size=nb_size
        )
        
        # Save raw data for statistical tests
        raw_data_path = exp_dir / 'raw_metrics_data.pkl'
        import pickle
        with open(raw_data_path, 'wb') as f:
            pickle.dump(raw_data, f)
        print(f"Saved raw data to {raw_data_path}")
        
        # Save individual results
        metrics_path = exp_dir / 'biological_metrics.csv'
        results_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
        
        return results_df
    
    def _evaluate_with_raw_data(self, nca, mix_nca, stochastic_mix_nca, nca_with_noise, nb_size: int):
        """Evaluate models and return both summary and raw data"""
        from mix_NCA.utils_simulations import grid_to_channels_batch
        from mix_NCA.BiologicalMetrics import BiologicalMetrics
        from mix_NCA.TissueModel import ComplexCellType
        
        # Collect all true states
        initial_states = []
        for hist in self.histories:
            grid_state = hist[0]
            encoded_state = grid_to_channels_batch(grid_state, len(ComplexCellType), self.device)
            initial_states.append(encoded_state)
        
        # Stack all true states
        true_dataset = torch.cat([torch.tensor(ts[-1]).to(self.device).unsqueeze(0) for ts in self.histories], dim=0)
        
        # Store raw data for statistical tests
        raw_data = {
            'Model Type': [],
            'Neighborhood Size': [],
            'Evaluation': [],
            'KL Divergence': [],
            'Chi-Square': [],
            'Categorical MMD': [],
            'Tumor Size Diff': [],
            'Border Size Diff': [],
            'Spatial Variance Diff': []
        }
        
        # Evaluate Standard NCA (deterministic, evaluate multiple times for consistency)
        with torch.no_grad():
            for eval_idx in range(self.n_evaluations):
                standard_samples = []
                for true_state in initial_states:
                    result = nca(true_state, 35, return_history=True)
                    if isinstance(result, tuple):
                        sample = result[1][-1] if len(result[1]) > 0 else result[0]
                    else:
                        sample = result[-1]
                    standard_samples.append(sample.argmax(dim=1))
                standard_gen = torch.stack(standard_samples).squeeze(1)
                
                bio_metrics = BiologicalMetrics(true_dataset, standard_gen, list(ComplexCellType), self.device)
                dist_metrics = bio_metrics.distribution_metrics()
                spatial_metrics = bio_metrics.spatial_correlation()
                
                raw_data['Model Type'].append('Standard NCA')
                raw_data['Neighborhood Size'].append(nb_size)
                raw_data['Evaluation'].append(eval_idx)
                raw_data['KL Divergence'].append(dist_metrics['kl_divergence'])
                raw_data['Chi-Square'].append(dist_metrics['chi_square'])
                raw_data['Categorical MMD'].append(dist_metrics['categorical_mmd'])
                raw_data['Tumor Size Diff'].append(bio_metrics.tumor_size_distribution())
                raw_data['Border Size Diff'].append(spatial_metrics['border_size_diff'])
                raw_data['Spatial Variance Diff'].append(spatial_metrics['spatial_variance_diff'])
        
        # Evaluate stochastic models
        for name, model in [
            ('Mixture NCA', mix_nca),
            ('Stochastic Mixture NCA', stochastic_mix_nca),
            ('NCA with Noise', nca_with_noise)
        ]:
            for eval_idx in range(self.n_evaluations):
                torch.manual_seed(eval_idx)
                with torch.no_grad():
                    samples = []
                    for true_state in initial_states:
                        if name == 'NCA with Noise':
                            result = model(true_state, 35, return_history=True)
                            if isinstance(result, tuple):
                                sample = result[1][-1] if len(result[1]) > 0 else result[0]
                            else:
                                sample = result[-1]
                            sample = sample.argmax(dim=1)
                        else:
                            result = model(true_state, 35, return_history=True, sample_non_differentiable=True)
                            if isinstance(result, tuple):
                                sample = result[1][-1] if len(result[1]) > 0 else result[0]
                            else:
                                sample = result[-1]
                            sample = sample.argmax(dim=1)
                        samples.append(sample)
                    generated = torch.stack(samples).squeeze(1)
                
                bio_metrics = BiologicalMetrics(true_dataset, generated, list(ComplexCellType), self.device)
                dist_metrics = bio_metrics.distribution_metrics()
                spatial_metrics = bio_metrics.spatial_correlation()
                
                raw_data['Model Type'].append(name)
                raw_data['Neighborhood Size'].append(nb_size)
                raw_data['Evaluation'].append(eval_idx)
                raw_data['KL Divergence'].append(dist_metrics['kl_divergence'])
                raw_data['Chi-Square'].append(dist_metrics['chi_square'])
                raw_data['Categorical MMD'].append(dist_metrics['categorical_mmd'])
                raw_data['Tumor Size Diff'].append(bio_metrics.tumor_size_distribution())
                raw_data['Border Size Diff'].append(spatial_metrics['border_size_diff'])
                raw_data['Spatial Variance Diff'].append(spatial_metrics['spatial_variance_diff'])
        
        # Create summary DataFrame (for compatibility)
        raw_df = pd.DataFrame(raw_data)
        summary_df = raw_df.groupby(['Model Type']).agg({
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
            model_data = raw_df[raw_df['Model Type'] == model_type]
            results_dict['Model Type'].append(model_type)
            for metric in ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
                          'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']:
                mean_val = model_data[metric].mean()
                std_val = model_data[metric].std()
                results_dict[metric].append(f"{mean_val:.3f}")
                results_dict[f"{metric} SD"].append(f"±{std_val:.3f}")
        
        results_df = pd.DataFrame(results_dict)
        results_df['Neighborhood Size'] = nb_size
        
        return results_df, raw_data
    
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
    
    def statistical_tests(self) -> Dict:
        """Perform formal statistical tests to compare neighborhood sizes
        
        Uses raw evaluation data (multiple evaluations per configuration) to perform
        proper statistical tests: Kruskal-Wallis, Mann-Whitney U, t-test, etc.
        """
        print(f"\n{'='*60}")
        print("Formal Statistical Tests")
        print(f"{'='*60}\n")
        
        import pickle
        from scipy.stats import ttest_ind, shapiro, levene
        
        # Load raw data from all neighborhood sizes
        raw_data_all = []
        neighborhood_sizes = sorted([int(d.name.split('_')[1]) for d in self.base_dir.iterdir() 
                                    if d.is_dir() and d.name.startswith('NB_')])
        
        for nb_size in neighborhood_sizes:
            raw_data_path = self.base_dir / f'NB_{nb_size}' / 'raw_metrics_data.pkl'
            if raw_data_path.exists():
                with open(raw_data_path, 'rb') as f:
                    raw_data = pickle.load(f)
                    raw_data_all.append(pd.DataFrame(raw_data))
            else:
                print(f"Warning: Raw data not found for NB_{nb_size}. Run evaluation first.")
                return {}
        
        if not raw_data_all:
            print("Error: No raw data found. Please run evaluation with save_raw_data=True")
            return {}
        
        # Combine all raw data
        raw_df = pd.concat(raw_data_all, ignore_index=True)
        
        results = {}
        metric_cols = ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
                      'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']
        
        for metric in metric_cols:
            if metric not in raw_df.columns:
                continue
            
            print(f"\n{'='*60}")
            print(f"STATISTICAL TESTS: {metric}")
            print(f"{'='*60}\n")
            
            metric_results = {}
            model_types = raw_df['Model Type'].unique()
            
            for model_type in model_types:
                model_data = raw_df[raw_df['Model Type'] == model_type]
                
                # Extract groups for each neighborhood size
                groups = {}
                for nb_size in neighborhood_sizes:
                    nb_data = model_data[model_data['Neighborhood Size'] == nb_size][metric].values
                    if len(nb_data) > 0:
                        groups[nb_size] = nb_data
                
                if len(groups) < 2:
                    continue
                
                print(f"\n{model_type}:")
                print(f"  Sample sizes: {[(nb, len(groups[nb])) for nb in sorted(groups.keys())]}")
                
                # Descriptive statistics
                for nb in sorted(groups.keys()):
                    values = groups[nb]
                    print(f"    NB={nb}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                          f"median={np.median(values):.4f}")
                
                # Test for normality (Shapiro-Wilk test on each group)
                normalities = {}
                for nb in sorted(groups.keys()):
                    if len(groups[nb]) >= 3 and len(groups[nb]) <= 5000:  # Shapiro-Wilk works for 3-5000 samples
                        stat, p = shapiro(groups[nb])
                        normalities[nb] = {'statistic': stat, 'p_value': p, 'normal': p > 0.05}
                    else:
                        normalities[nb] = {'normal': None}  # Cannot test
                
                all_normal = all(n.get('normal', False) for n in normalities.values() if n.get('normal') is not None)
                
                # Test for equal variances (Levene's test)
                group_list = [groups[nb] for nb in sorted(groups.keys())]
                try:
                    levene_stat, levene_p = levene(*group_list)
                    equal_variances = levene_p > 0.05
                except:
                    equal_variances = False
                    levene_stat, levene_p = np.nan, np.nan
                
                print(f"  Normality tests (Shapiro-Wilk):")
                for nb in sorted(normalities.keys()):
                    norm = normalities[nb]
                    if norm.get('normal') is not None:
                        status = "normal" if norm['normal'] else "non-normal"
                        print(f"    NB={nb}: W={norm['statistic']:.4f}, p={norm['p_value']:.4f} ({status})")
                
                print(f"  Equal variances (Levene's test): W={levene_stat:.4f}, p={levene_p:.4f} "
                      f"({'equal' if equal_variances else 'unequal'})")
                
                # Kruskal-Wallis test (non-parametric ANOVA) - always applicable
                try:
                    kw_stat, kw_p = kruskal(*group_list)
                    print(f"\n  Kruskal-Wallis Test (non-parametric ANOVA):")
                    print(f"    H-statistic: {kw_stat:.4f}")
                    print(f"    p-value: {kw_p:.6f}")
                    significance = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'
                    print(f"    Significance: {significance} ({'significant' if kw_p < 0.05 else 'not significant'})")
                    
                    metric_results[model_type] = {
                        'kruskal_wallis': {
                            'statistic': kw_stat,
                            'p_value': kw_p,
                            'significant': kw_p < 0.05
                        },
                        'normality': normalities,
                        'equal_variances': equal_variances
                    }
                    
                    # If significant, do pairwise comparisons
                    if kw_p < 0.05:
                        print(f"\n  Pairwise Comparisons (post-hoc tests):")
                        pairwise_results = {}
                        
                        for i, nb1 in enumerate(sorted(groups.keys())):
                            for j, nb2 in enumerate(sorted(groups.keys())):
                                if i >= j:
                                    continue
                                
                                data1 = groups[nb1]
                                data2 = groups[nb2]
                                
                                # Mann-Whitney U test (non-parametric, always applicable)
                                try:
                                    u_stat, u_p = mannwhitneyu(data1, data2, alternative='two-sided')
                                    u_significant = u_p < 0.05
                                    
                                    # T-test (parametric, if data is normal)
                                    if all_normal and equal_variances:
                                        t_stat, t_p = ttest_ind(data1, data2, equal_var=True)
                                        t_significant = t_p < 0.05
                                    elif all_normal:
                                        t_stat, t_p = ttest_ind(data1, data2, equal_var=False)
                                        t_significant = t_p < 0.05
                                    else:
                                        t_stat, t_p = np.nan, np.nan
                                        t_significant = False
                                    
                                    mean1, mean2 = np.mean(data1), np.mean(data2)
                                    better = nb1 if mean1 < mean2 else nb2  # Lower is better
                                    
                                    pairwise_results[f"{nb1}_vs_{nb2}"] = {
                                        'mann_whitney': {
                                            'statistic': u_stat,
                                            'p_value': u_p,
                                            'significant': u_significant
                                        },
                                        't_test': {
                                            'statistic': t_stat if not np.isnan(t_stat) else None,
                                            'p_value': t_p if not np.isnan(t_p) else None,
                                            'significant': t_significant
                                        },
                                        'mean1': mean1,
                                        'mean2': mean2,
                                        'better': better
                                    }
                                    
                                    if u_significant:
                                        sig = '***' if u_p < 0.001 else '**' if u_p < 0.01 else '*'
                                        print(f"    NB{nb1} vs NB{nb2}:")
                                        print(f"      Mann-Whitney U: U={u_stat:.2f}, p={u_p:.6f} {sig}")
                                        if not np.isnan(t_p):
                                            t_sig = '***' if t_p < 0.001 else '**' if t_p < 0.01 else '*'
                                            print(f"      t-test: t={t_stat:.4f}, p={t_p:.6f} {t_sig}")
                                        print(f"      Means: NB{nb1}={mean1:.4f}, NB{nb2}={mean2:.4f} (better: NB{better})")
                                
                                except Exception as e:
                                    print(f"      Error in pairwise test NB{nb1} vs NB{nb2}: {e}")
                        
                        metric_results[model_type]['pairwise'] = pairwise_results
                
                except Exception as e:
                    print(f"  Error in Kruskal-Wallis test: {e}")
            
            results[metric] = metric_results
        
        return results
    
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
                    # Pearson correlation
                    pearson_r, pearson_p = stats.pearsonr(sizes, means)
                    
                    # Spearman correlation (monotonic relationship)
                    spearman_r, spearman_p = stats.spearmanr(sizes, means)
                    
                    # Linear regression slope
                    slope, intercept, r_value, p_value, std_err = stats.linregress(sizes, means)
                    
                    trend_results.append({
                        'Model Type': model_type,
                        'Metric': metric,
                        'Mean_NB3': grouped.loc[3, 'mean'] if 3 in sizes else np.nan,
                        'Mean_NB7': grouped.loc[7, 'mean'] if 7 in sizes else np.nan,
                        'Improvement_3_to_7': (grouped.loc[3, 'mean'] - grouped.loc[7, 'mean']) / grouped.loc[3, 'mean'] * 100 if (3 in sizes and 7 in sizes) else np.nan,
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
                    print(f"  Correlation (Pearson): r={pearson_r:.4f}, p={pearson_p:.4f}")
                    print(f"  Correlation (Spearman): r={spearman_r:.4f}, p={spearman_p:.4f}")
                    print(f"  Linear trend: slope={slope:.6f}, p={p_value:.4f}")
                    print(f"  Best NB: {sizes[np.argmin(means)]}, Worst NB: {sizes[np.argmax(means)]}")
                    if 3 in sizes and 7 in sizes:
                        improvement = (grouped.loc[3, 'mean'] - grouped.loc[7, 'mean']) / grouped.loc[3, 'mean'] * 100
                        print(f"  Improvement NB3→NB7: {improvement:.2f}%")
                    print()
        
        return pd.DataFrame(trend_results)
    
    def computational_complexity_analysis(self, n_samples: int = 5) -> pd.DataFrame:
        """Measure computational time for different neighborhood sizes"""
        print(f"\n{'='*60}")
        print("Computational Complexity Analysis")
        print(f"{'='*60}\n")
        
        complexity_results = []
        
        def make_update_net_fn(device):
            def update_net_wrapper(n_channels, hidden_dims=128, n_channels_out=None, device_arg=None):
                return classification_update_net(n_channels, hidden_dims, n_channels_out, device=device)
            return update_net_wrapper
        
        update_net_fn = make_update_net_fn(self.device)
        
        # Get a sample initial state
        initial_state = grid_to_channels_batch([self.histories[0][0]], len(ComplexCellType), self.device)
        
        for nb_size in [3, 4, 5, 6, 7]:
            exp_dir = self.base_dir / f"NB_{nb_size}"
            if not exp_dir.exists():
                continue
            
            print(f"Testing NB_{nb_size}...")
            
            # Initialize model
            nca = ExtendedNCA(
                update_net=classification_update_net(6 * 3, n_channels_out=6, device=self.device),
                hidden_dim=self.HIDDEN_DIM,
                maintain_seed=False,
                use_alive_mask=False,
                state_dim=self.STATE_DIM,
                residual=False,
                neighborhood_size=nb_size,
                device=self.device
            )
            
            nca.load_state_dict(torch.load(exp_dir / 'standard_nca.pt', map_location=self.device, weights_only=True))
            nca = nca.to(self.device)
            nca.eval()
            
            # Warm-up
            with torch.no_grad():
                _ = nca(initial_state, 5, return_history=False)
            
            # Measure time
            times = []
            n_steps = 35
            
            for _ in range(n_samples):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.time()
                with torch.no_grad():
                    _ = nca(initial_state, n_steps, return_history=False)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            # Theoretical complexity: O(nb_size^2) for convolution
            theoretical_ops = nb_size ** 2
            
            complexity_results.append({
                'Neighborhood Size': nb_size,
                'Mean Time (s)': mean_time,
                'Std Time (s)': std_time,
                'Time per Step (ms)': mean_time / n_steps * 1000,
                'Theoretical O(n²)': theoretical_ops,
                'Normalized Time': mean_time / (3**2)  # Normalize to NB=3
            })
            
            print(f"  Mean time: {mean_time:.4f} ± {std_time:.4f} s")
            print(f"  Time per step: {mean_time/n_steps*1000:.2f} ms")
            print(f"  Theoretical complexity factor: {theoretical_ops / (3**2):.2f}x")
            print()
        
        return pd.DataFrame(complexity_results)
    
    def create_visualizations(self, output_dir: Optional[str] = None):
        """Create comprehensive visualizations using Plotly"""
        if output_dir is None:
            output_dir = self.base_dir / "analysis_plots"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Creating Visualizations")
        print(f"{'='*60}\n")
        
        df = self.parse_metrics()
        
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
    
    def generate_report(self, output_path: Optional[str] = None):
        """Generate comprehensive analysis report"""
        if output_path is None:
            output_path = self.base_dir / "neighborhood_size_analysis_report.txt"
        else:
            output_path = Path(output_path)
        
        print(f"\n{'='*60}")
        print("Generating Analysis Report")
        print(f"{'='*60}\n")
        
        df = self.parse_metrics()
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NEIGHBORHOOD SIZE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write("This report analyzes the impact of neighborhood size (3-7) on NCA model performance.\n")
            f.write(f"Models analyzed: {', '.join(df['Model Type'].unique())}\n")
            f.write(f"Neighborhood sizes: {sorted(df['Neighborhood Size'].unique())}\n")
            f.write(f"Number of evaluations: {self.n_evaluations}\n\n")
            
            # Best performing configurations
            f.write("BEST PERFORMING CONFIGURATIONS\n")
            f.write("-"*80 + "\n")
            
            metric_cols = ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
                          'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']
            
            for metric in metric_cols:
                if metric not in df.columns:
                    continue
                
                best_idx = df[metric].idxmin()
                best_row = df.loc[best_idx]
                f.write(f"\n{metric}:\n")
                f.write(f"  Best: {best_row['Model Type']} with NB={best_row['Neighborhood Size']} "
                       f"(value: {best_row[metric]:.4f})\n")
            
            # Statistical tests
            f.write("\n\nSTATISTICAL SIGNIFICANCE TESTS\n")
            f.write("-"*80 + "\n")
            
            stat_results = self.statistical_tests()
            for metric, model_results in stat_results.items():
                f.write(f"\n{metric}:\n")
                for model_type, results in model_results.items():
                    if 'kruskal_wallis' in results:
                        kw = results['kruskal_wallis']
                        f.write(f"  {model_type}:\n")
                        f.write(f"    Kruskal-Wallis: H={kw['statistic']:.4f}, "
                               f"p={kw['p_value']:.6f} "
                               f"{'***' if kw['p_value'] < 0.001 else '**' if kw['p_value'] < 0.01 else '*' if kw['p_value'] < 0.05 else '(not significant)'}\n")
            
            # Trend analysis
            f.write("\n\nPERFORMANCE TRENDS\n")
            f.write("-"*80 + "\n")
            
            trend_df = self.performance_trend_analysis()
            for _, row in trend_df.iterrows():
                f.write(f"\n{row['Model Type']} - {row['Metric']}:\n")
                f.write(f"  Correlation (Pearson): r={row['Pearson_r']:.4f}, p={row['Pearson_p']:.6f}\n")
                f.write(f"  Correlation (Spearman): r={row['Spearman_r']:.4f}, p={row['Spearman_p']:.6f}\n")
                f.write(f"  Linear slope: {row['Slope']:.6f}, p={row['Slope_p']:.6f}\n")
                f.write(f"  Best NB: {row['Best_NB']}, Worst NB: {row['Worst_NB']}\n")
                if not np.isnan(row['Improvement_3_to_7']):
                    f.write(f"  Improvement NB3→NB7: {row['Improvement_3_to_7']:.2f}%\n")
            
            # Computational complexity
            f.write("\n\nCOMPUTATIONAL COMPLEXITY\n")
            f.write("-"*80 + "\n")
            
            complexity_df = self.computational_complexity_analysis()
            for _, row in complexity_df.iterrows():
                f.write(f"\nNB={row['Neighborhood Size']}:\n")
                f.write(f"  Mean time: {row['Mean Time (s)']:.4f} ± {row['Std Time (s)']:.4f} s\n")
                f.write(f"  Time per step: {row['Time per Step (ms)']:.2f} ms\n")
                f.write(f"  Complexity factor (vs NB=3): {row['Normalized Time']:.2f}x\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            
            # Find best overall configuration
            # Weight different metrics (lower is better for all)
            normalized_scores = {}
            for model_type in df['Model Type'].unique():
                model_data = df[df['Model Type'] == model_type]
                for nb_size in model_data['Neighborhood Size'].unique():
                    nb_data = model_data[model_data['Neighborhood Size'] == nb_size]
                    
                    # Normalize each metric to [0, 1] and sum
                    score = 0
                    for metric in metric_cols:
                        if metric in nb_data.columns:
                            metric_values = df[df['Model Type'] == model_type][metric]
                            min_val, max_val = metric_values.min(), metric_values.max()
                            if max_val > min_val:
                                normalized = (nb_data[metric].mean() - min_val) / (max_val - min_val)
                                score += normalized
                    
                    normalized_scores[(model_type, nb_size)] = score
            
            best_config = min(normalized_scores.items(), key=lambda x: x[1])
            f.write(f"\nBest overall configuration: {best_config[0][0]} with NB={best_config[0][1]}\n")
            f.write(f"(Based on normalized sum of all metrics)\n\n")
            
            f.write("Key Findings:\n")
            f.write("1. Analyze the statistical significance tests to determine if differences are meaningful\n")
            f.write("2. Consider computational cost vs. performance gain\n")
            f.write("3. Check if larger neighborhoods provide consistent improvements across all metrics\n")
            f.write("4. Evaluate if the improvement justifies the increased computational cost\n")
        
        print(f"Report saved to {output_path}")


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
    parser.add_argument('--neighborhood_sizes', type=str, default='3,4,5,6,7',
                       help='Comma-separated list of neighborhood sizes to analyze')
    parser.add_argument('--skip_plots', action='store_true',
                       help='Skip generating visualizations')
    
    args = parser.parse_args()
    
    neighborhood_sizes = [int(s.strip()) for s in args.neighborhood_sizes.split(',') if s.strip()]
    
    # Initialize analyzer
    analyzer = NeighborhoodSizeAnalyzer(
        results_dir=args.results_dir,
        histories_path=args.histories_path,
        device=args.device,
        n_evaluations=args.n_evaluations
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
    
    # Statistical tests
    stat_results = analyzer.statistical_tests()
    
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
    
    # Generate report
    analyzer.generate_report()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
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

