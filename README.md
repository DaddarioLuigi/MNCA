# Comprehensive Assessment of Mixture Neural Cellular Automata (MNCA) for Modeling Complex Biological Systems

This repository contains the implementation and experimental framework for a thesis project focused on the systematic evaluation of Mixture Neural Cellular Automata (MNCA) models for modeling complex biological systems.

> **Note:** This is an independent repository created for thesis research purposes. It is based on the original MNCA implementation, but has been adapted and extended for comprehensive evaluation and analysis. See [Credits](#credits) section or [CREDITS.md](CREDITS.md) for full attribution.

## Thesis Overview

**Title:** Comprehensive Assessment of Mixture Neural Cellular Automata (MNCA) for Modeling Complex Biological Systems

**Description:** The Mixture of Neural Cellular Automata (MNCAs) model was recently proposed to extend traditional NCAs by integrating probabilistic rules and intrinsic noise, allowing for the robust modelling of stochastic biological processes, such as tissue growth, as well as image morphogenesis. This thesis systematically evaluates the model across diverse experimental settings via a comprehensive analysis of its performance under varying (hyper)parameters and update rules/dynamics. Insights gained from these evaluations will guide potential extensions to improve the model's expressiveness and scalability, while applications to real-world case studies will be central to the work.

## Background

This project is based on the research presented in the paper [2506.20486v1.pdf](papers/2506.20486v1.pdf) and builds upon the original implementation available at [https://github.com/Militeee/MNCA](https://github.com/Militeee/MNCA). The codebase has been adapted and extended to support comprehensive evaluation and analysis of MNCA models, with a particular focus on biological simulations and microscopy image applications.

**Important:** This repository is not a fork of the original project, but rather an independent repository created for thesis research. The code has been adapted, extended, and reorganized to support the specific research objectives outlined in the thesis.

## Project Structure

```
├── mix_NCA/                    # Core MNCA implementations
│   ├── NCA.py                  # Base Neural Cellular Automata implementation
│   ├── MixtureNCA.py           # Mixture of NCAs with probabilistic rule selection
│   ├── MixtureNCANoise.py      # Mixture of NCAs with internal noise
│   ├── ExtendedNCA.py          # Extended NCA variants
│   ├── ExtendedMixtureNCA.py   # Extended Mixture NCA variants
│   ├── ExtendedMixtureNCANoise.py  # Extended Mixture NCA with noise variants
│   ├── TissueModel.py          # Tissue simulation model for biological systems
│   ├── AGB_ABC_model.py        # Agent-based model with ABC parameter inference
│   ├── BiologicalMetrics.py    # Metrics for evaluating biological simulations
│   ├── RobustnessAnalysis.py   # Perturbation and robustness analysis tools
│   └── utils_*.py              # Utility functions for formatting, images, simulations, and Visium data
│
├── notebooks/                  # Jupyter notebooks for analysis and experiments
│   ├── tissue_simulation_MNCA.ipynb        # Tissue simulation experiments with MNCA
│   ├── tissue_simulation_other_models.ipynb # Tissue simulation with ABC models
│   ├── experiment_microscopy.ipynb         # Microscopy image experiments
│   ├── final_stats.ipynb                   # Statistical analysis and results
│   ├── check_gaussian_noise.ipynb          # Noise analysis experiments
│   └── histories.npy                       # Pre-computed tissue simulation data
│
├── experiments/                # Experiment scripts
│   └── tissue_simulation_extended.py       # Extended tissue simulation experiments
│
├── data/                       # Data files for experiments
│   └── emojis/                 # Emoji pattern data
│
├── papers/                     # Research papers and references
│   └── 2506.20486v1.pdf        # Original MNCA paper
│
├── generate_histories.py       # Script to generate tissue simulation histories
├── requirements.txt            # Python dependencies
├── setup.cfg                   # Package configuration
└── Makefile                    # Build and utility commands
```

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd MNCA
```

2. Create a virtual environment (optional but recommended):
```bash
conda create --name mnca python=3.11 -y
conda activate mnca
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using the Makefile:
```bash
make requirements
```

## Core Components

### Neural Cellular Automata Models

The repository implements several variants of Neural Cellular Automata:

- **NCA**: Standard Neural Cellular Automata with deterministic update rules
- **MixtureNCA**: Mixture of multiple update rules with probabilistic selection
- **MixtureNCANoise**: Mixture model with internal stochastic noise
- **Extended variants**: Extended implementations with additional features

### Key Features

- **Flexible grid types**: Support for square and hexagonal grids
- **Multiple perception filters**: Sobel and Laplacian filters for neighborhood perception
- **Biological modeling**: Specialized tools for tissue growth simulations
- **Robustness analysis**: Tools for evaluating model stability under perturbations
- **Biological metrics**: Comprehensive metrics for evaluating biological simulations

## Usage

### Basic Example

```python
from mix_NCA.MixtureNCA import MixtureNCA
from mix_NCA.utils_simulations import classification_update_net

# Create a Mixture NCA model
model = MixtureNCA(
    update_nets=classification_update_net,
    num_rules=5,
    state_dim=16,
    hidden_dim=128,
    device="cuda"
)
```

### Tissue Simulation

The repository includes specialized tools for biological tissue simulations:

```python
from mix_NCA.TissueModel import TissueModel

# Create a tissue model
tissue = TissueModel(
    grid_size=50,
    initial_stem_cells=5,
    stem_division_rate=0.5
)

# Run simulation
history, final_state = tissue.simulate(steps=100)
```

### Running Experiments

Experiments can be run using the provided notebooks or scripts:

- **Tissue simulations**: See `notebooks/tissue_simulation_MNCA.ipynb`
- **Microscopy experiments**: See `notebooks/experiment_microscopy.ipynb`
- **Extended experiments**: Run `experiments/tissue_simulation_extended.py`

## Experiments

The repository includes several experimental frameworks:

1. **Tissue Growth Simulations**: Evaluation of MNCA models on synthetic tissue growth patterns
2. **Microscopy Image Analysis**: Application to real-world microscopy image data
3. **Robustness Analysis**: Systematic evaluation of model stability under various perturbations
4. **Parameter Sensitivity**: Analysis of model performance under varying hyperparameters

## Development

### Code Formatting

Format code using black:
```bash
make format
```

### Linting

Check code style:
```bash
make lint
```

### Cleanup

Remove compiled Python files:
```bash
make clean
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{mnca2024,
  title={Mixture of Neural Cellular Automata},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  note={arXiv:2506.20486}
}
```

## Credits

This repository is based on the original MNCA implementation and research. We gratefully acknowledge the following sources:

### Original MNCA Implementation
- **Repository:** [https://github.com/Militeee/MNCA](https://github.com/Militeee/MNCA)
- **Paper:** [2506.20486v1.pdf](papers/2506.20486v1.pdf) (arXiv:2506.20486)
- **Authors:** Original authors of the Mixture Neural Cellular Automata paper

### Additional Acknowledgments
- The base NCA implementation (`mix_NCA/NCA.py`) is adapted from [https://github.com/greydanus/studying_growth](https://github.com/greydanus/studying_growth), which provides an excellent introduction to Neural Cellular Automata.

### This Repository
This is an independent repository created for thesis research. While based on the original MNCA implementation, the code has been:
- Adapted and extended for comprehensive evaluation
- Reorganized to support thesis research objectives
- Enhanced with additional experimental frameworks
- Focused on biological simulations and microscopy applications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Status

This repository is actively under development as part of an ongoing thesis project. The codebase and documentation will be updated as the research progresses.

