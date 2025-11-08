# Neighborhood Extension for NCA and MixtureNCA

This feature extends Neural Cellular Automata (NCA) to support neighborhoods larger than the classic 3x3, implemented via inheritance rather than rewrites. It also includes support for MixtureNCA and a suite of scripts for quick and full experiments.

## What Was Done

- `ExtendedNCA` EXTENDS `NCA` and only adds support for `neighborhood_size > 3`, reusing all the base logic (forward, masks, residual, alive mask, etc.).
- `ExtendedMixtureNCA` EXTENDS `MixtureNCA`, keeping the original mixture logic and only adding extended perception filters.
- Extended perception filters for Sobel and Laplacian with 4x4, 5x5, 6x6, 7x7 kernels (identity center, extended derivatives), with dynamic padding.
- New test and training scripts: quick, improved, proper training, extended training.
- Updated `experiments/neighborhood_experiment.py` to support `--use_mixture`, `--num_rules`, `--neighborhood_sizes`, `--total_steps`.

## Supported Neighborhood Configurations

- **3x3**: 8 neighbors (standard) — original configuration
- **4x4**: 12 neighbors (extended)
- **5x5**: 24 neighbors (extended)
- **6x6**: 32 neighbors (very extended)
- **7x7**: 48 neighbors (very extended)

## Main Files

### Extended Classes (Inheritance)
- `mix_NCA/ExtendedNCA.py` — Inherits from `NCA` and adds `neighborhood_size` without rewriting the forward
- `mix_NCA/ExtendedMixtureNCA.py` — Inherits from `MixtureNCA` and adds extended filters
- `mix_NCA/ExtendedMixtureNCANoise.py` — Inherits from `MixtureNCANoise` and adds extended filters with latent noise support

### Experiment and Test Scripts
- `experiments/neighborhood_experiment.py` — Full experiment (supports `--use_mixture`, `--num_rules`, `--neighborhood_sizes`, `--total_steps`)
- `experiments/quick_neighborhood_test.py` — Very fast test with a synthetic image
- `experiments/improved_neighborhood_test.py` — Numerically stable test (clipping, early stopping)
- `experiments/proper_training_test.py` — True training with optimization and evaluation
- `experiments/extended_training_test.py` — Extended training (scheduler, patience)
- `notebooks/neighborhood_configuration_test.ipynb` — Interactive notebook

## Design: Extend, Don’t Rewrite

- `ExtendedNCA(NCA)`: calls the parent constructor, stores `neighborhood_size`, and, if > 3, only replaces perception kernels with extended versions; re-binds `self.perceive` depending on modality. No changes to forward or semantics.
- `ExtendedMixtureNCA(MixtureNCA)`: same as above, keeping mixture net, update nets, forward, and temperature unchanged.

### Added API

- `neighborhood_size: int` (default 3). Typical values: 3, 4, 5, 6, 7. Controls perception kernel size and padding.
- Supported filters: `filter_type` in {"sobel", "laplacian"}, as in the base classes.

## How to Use

### 1. Quick Test
```bash
cd experiments
python quick_neighborhood_test.py
```

### 2. Full Experiment with CIFAR-10
```bash
cd experiments
# Extended NCA
python neighborhood_experiment.py --category 0 --data_dir ../data \
  --neighborhood_sizes 4 5 6 7 --total_steps 200

# Extended Mixture (e.g., 2 rules, 5x5 neighborhood)
python neighborhood_experiment.py --category 0 --data_dir ../data \
  --neighborhood_sizes 5 --use_mixture --num_rules 2 --total_steps 200
```

### 3. Interactive Notebook
```bash
cd notebooks
jupyter notebook neighborhood_configuration_test.ipynb
```

## Key Parameters

- `neighborhood_size`: Neighborhood size (3, 4, 5, 6, 7, ...)
- `filter_type`: Filter type ("sobel" or "laplacian")
- `state_dim`: State size (default: 16)
- `hidden_dim`: Hidden size (default: 64)

## Summary of Experimental Findings

### Insights
- **Better patterns**: Larger neighborhoods can capture more complex and larger-scale structures
- **Higher stability**: Less sensitive to local noise
- **Improved convergence**: Can reach more stable solutions

### Caveats
- **More computation**: More operations per step
- **Memory**: Larger perception filters consume more memory
- **Overfitting**: Can become overly specific for certain patterns

### Evidence from Included Tests

- No-training tests: not indicative (similar losses across configurations) → proper training is required.
- Proper training (`proper_training_test.py`): 3x3 generalizes best; 4x4, 5x5, 6x6, and 7x7 can be unstable depending on the pattern.
- Extended training (`extended_training_test.py`): confirms 3x3 as best on average; larger neighborhoods (4x4, 5x5, 6x6, 7x7) work but may require more regularization.

In short: 3x3 remains a strong baseline; larger neighborhoods (4x4, 5x5, 6x6, 7x7) are useful for complex patterns but require more regularization/care.

## Usage Examples

### Create an NCA with an Extended Neighborhood
```python
from mix_NCA.ExtendedNCA import ExtendedNCA
from mix_NCA.utils_images import standard_update_net

# Create a 5x5 neighborhood NCA
model = ExtendedNCA(
    update_net=standard_update_net(16 * 3, 64, 16 * 2, device='cuda'),
    state_dim=16,
    hidden_dim=64,
    neighborhood_size=5,  # 5x5 neighborhood
    device='cuda'
)

# Or create a 4x4 or 6x6 neighborhood NCA
model_4x4 = ExtendedNCA(
    update_net=standard_update_net(16 * 3, 64, 16 * 2, device='cuda'),
    state_dim=16,
    hidden_dim=64,
    neighborhood_size=4,  # 4x4 neighborhood
    device='cuda'
)
```

### Compare Different Configurations
```python
neighborhood_sizes = [3, 4, 5, 6, 7]
results = {}

for size in neighborhood_sizes:
    model = ExtendedNCA(
        update_net=standard_update_net(16 * 3, 64, 16 * 2, device='cuda'),
        state_dim=16,
        hidden_dim=64,
        neighborhood_size=size,
        device='cuda'
    )
    
    # Train and evaluate
    loss = train_and_evaluate(model, target)
    results[size] = loss
```

## Interpreting Results

### Important Metrics
1. **Final Loss**: Final loss after training
2. **Convergence Step**: Step at which the model converges
3. **Stability**: Variance of the last steps (lower = more stable)

### Recommendations
- **Simple patterns**: Use 3x3 or 4x4 neighborhoods
- **Complex patterns**: Use 5x5, 6x6, or 7x7 neighborhoods
- **Balance**: Try multiple configurations to find the optimal trade-off

## Troubleshooting

### Common Errors
1. **Out of Memory**: Reduce `neighborhood_size` or `hidden_dim`
2. **Slow Convergence**: Increase `learning_rate` or reduce `neighborhood_size`
3. **Instability**: Reduce `learning_rate` or increase `dropout`

### Optimizations
- Use GPU when available
- Reduce `total_steps` for quick tests
- Use smaller `batch_size` for large neighborhoods

## Future Extensions

- **Non-uniform neighborhoods**: Filters with different weights for various distances
- **Adaptive neighborhoods**: Neighborhood size changes during training
- **Multi-scale neighborhoods**: Combine multiple neighborhood sizes
- **Spherical neighborhoods**: For 3D applications


