# Guida: Modificare i Notebook per Testare con Diversi Neighborhood Sizes

Questa guida spiega come modificare i notebook esistenti per testare i modelli addestrati con diversi neighborhood sizes (3x3, 4x4, 5x5, 6x6, 7x7).

## Struttura dei File Salvati

Dopo aver eseguito gli script `*_extended.py`, i modelli vengono salvati nella seguente struttura:

```
results_extended/
├── experiment_EMOJI_robustness/
│   ├── NB_3/
│   │   ├── standard_model.pt
│   │   ├── mixture_model.pt
│   │   └── mixture_model_noise.pt
│   ├── NB_4/
│   │   └── ...
│   └── ...
├── experiment_CIFAR_robustness/
│   └── ...
└── experiment_TISSUE_SIMULATION/
    ├── NB_3/
    │   ├── standard_nca.pt
    │   ├── nca_with_noise.pt
    │   ├── mixture_nca.pt
    │   └── stochastic_mix_nca.pt
    └── ...
```

## Modifiche Necessarie ai Notebook

### 1. Modificare le Importazioni

**PRIMA:**
```python
from mix_NCA.NCA import NCA
from mix_NCA.MixtureNCA import MixtureNCA
from mix_NCA.MixtureNCANoise import MixtureNCANoise
```

**DOPO:**
```python
from mix_NCA.ExtendedNCA import ExtendedNCA
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
```

### 2. Modificare l'Inizializzazione dei Modelli

**PRIMA (tissue_simulation_MNCA.ipynb):**
```python
nca = NCA(
    update_net=classification_update_net(6 * 3, n_channels_out=6),
    hidden_dim=128,
    maintain_seed=False,
    use_alive_mask=False,
    state_dim=6,
    residual=False,
    device="cuda:2"
)

mix_nca = MixtureNCA(
    update_nets=classification_update_net,
    hidden_dim=128,
    maintain_seed=False,
    use_alive_mask=False,
    state_dim=6,
    num_rules=5,
    residual=False,
    temperature=3,
    device="cuda:2"
)

stochastic_mix_nca = MixtureNCANoise(
    update_nets=classification_update_net,
    hidden_dim=128,
    maintain_seed=False,
    use_alive_mask=False,
    state_dim=6,
    num_rules=5,
    temperature=3,
    residual=False,
    device="cuda:2"
)
```

**DOPO (per ogni neighborhood size):**
```python
# Definisci i neighborhood sizes da testare
neighborhood_sizes = [3, 4, 5, 6, 7]

# Itera su ogni neighborhood size
for nb_size in neighborhood_sizes:
    print(f"\n=== Testing with Neighborhood Size: {nb_size}x{nb_size} ===")
    
    # Inizializza i modelli con il neighborhood_size specificato
    nca = ExtendedNCA(
        update_net=classification_update_net(6 * 3, n_channels_out=6),
        hidden_dim=128,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=6,
        residual=False,
        neighborhood_size=nb_size,  # <-- AGGIUNGI QUESTO
        device="cuda:2"
    )
    
    mix_nca = ExtendedMixtureNCA(
        update_nets=classification_update_net,
        hidden_dim=128,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=6,
        num_rules=5,
        residual=False,
        temperature=3,
        neighborhood_size=nb_size,  # <-- AGGIUNGI QUESTO
        device="cuda:2"
    )
    
    stochastic_mix_nca = ExtendedMixtureNCANoise(
        update_nets=classification_update_net,
        hidden_dim=128,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=6,
        num_rules=5,
        temperature=3,
        residual=False,
        neighborhood_size=nb_size,  # <-- AGGIUNGI QUESTO
        device="cuda:2"
    )
    
    nca_with_noise = ExtendedNCA(
        update_net=classification_update_net(6 * 3, n_channels_out=6 * 2),
        hidden_dim=128,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=6,
        residual=False,
        distribution="normal",
        neighborhood_size=nb_size,  # <-- AGGIUNGI QUESTO
        device="cuda:2"
    )
    
    # Carica i pesi dal path corretto
    model_dir = f"../results_extended/experiment_TISSUE_SIMULATION/NB_{nb_size}/"
    
    nca.load_state_dict(torch.load(os.path.join(model_dir, 'standard_nca.pt')))
    nca_with_noise.load_state_dict(torch.load(os.path.join(model_dir, 'nca_with_noise.pt')))
    mix_nca.load_state_dict(torch.load(os.path.join(model_dir, 'mixture_nca.pt')))
    stochastic_mix_nca.load_state_dict(torch.load(os.path.join(model_dir, 'stochastic_mix_nca.pt')))
    
    nca_with_noise.random_updates = True
    
    # Esegui i test come prima
    nca.eval()
    mix_nca.eval()
    stochastic_mix_nca.eval()
    nca_with_noise.eval()
    
    # ... resto del codice di test ...
```

### 3. Modifiche per final_stats.ipynb

Per il notebook `final_stats.ipynb`, che carica risultati da esperimenti emoji e CIFAR:

**PRIMA:**
```python
# Carica modelli emoji
model_path = "../results/experiment_EMOJI_robustness/standard_model.pt"
model.load_state_dict(torch.load(model_path))
```

**DOPO:**
```python
# Itera su diversi neighborhood sizes
neighborhood_sizes = [3, 4, 5, 6, 7]

for nb_size in neighborhood_sizes:
    # Carica modelli emoji
    model_dir = f"../results_extended/experiment_EMOJI_robustness/NB_{nb_size}/"
    
    model = ExtendedNCA(
        update_net=standard_update_net(...),
        neighborhood_size=nb_size,  # <-- AGGIUNGI QUESTO
        ...
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, 'standard_model.pt')))
    
    # ... resto del codice ...
```

### 4. Esempio Completo per tissue_simulation_MNCA.ipynb

Ecco un esempio completo di come modificare una cella del notebook:

```python
import torch
import numpy as np
import os
from mix_NCA.ExtendedNCA import ExtendedNCA
from mix_NCA.ExtendedMixtureNCA import ExtendedMixtureNCA
from mix_NCA.ExtendedMixtureNCANoise import ExtendedMixtureNCANoise
from mix_NCA.utils_simulations import classification_update_net
from mix_NCA.BiologicalMetrics import compare_generated_distributions

# Carica le simulazioni
histories = np.load('histories.npy')

# Definisci i neighborhood sizes da testare
neighborhood_sizes = [3, 4, 5, 6, 7]

# Dizionario per salvare i risultati
all_results = {}

for nb_size in neighborhood_sizes:
    print(f"\n{'='*60}")
    print(f"Testing with Neighborhood Size: {nb_size}x{nb_size}")
    print(f"{'='*60}\n")
    
    # Inizializza i modelli
    nca = ExtendedNCA(
        update_net=classification_update_net(6 * 3, n_channels_out=6),
        hidden_dim=128,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=6,
        residual=False,
        neighborhood_size=nb_size,
        device="cuda:2"
    )
    
    nca_with_noise = ExtendedNCA(
        update_net=classification_update_net(6 * 3, n_channels_out=6 * 2),
        hidden_dim=128,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=6,
        residual=False,
        distribution="normal",
        neighborhood_size=nb_size,
        device="cuda:2"
    )
    
    mix_nca = ExtendedMixtureNCA(
        update_nets=classification_update_net,
        hidden_dim=128,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=6,
        num_rules=5,
        residual=False,
        temperature=3,
        neighborhood_size=nb_size,
        device="cuda:2"
    )
    
    stochastic_mix_nca = ExtendedMixtureNCANoise(
        update_nets=classification_update_net,
        hidden_dim=128,
        maintain_seed=False,
        use_alive_mask=False,
        state_dim=6,
        num_rules=5,
        temperature=3,
        residual=False,
        neighborhood_size=nb_size,
        device="cuda:2"
    )
    
    # Carica i pesi
    model_dir = f"../results_extended/experiment_TISSUE_SIMULATION/NB_{nb_size}/"
    
    nca.load_state_dict(torch.load(os.path.join(model_dir, 'standard_nca.pt')))
    nca_with_noise.load_state_dict(torch.load(os.path.join(model_dir, 'nca_with_noise.pt')))
    mix_nca.load_state_dict(torch.load(os.path.join(model_dir, 'mixture_nca.pt')))
    stochastic_mix_nca.load_state_dict(torch.load(os.path.join(model_dir, 'stochastic_mix_nca.pt')))
    
    nca_with_noise.random_updates = True
    
    # Imposta in modalità eval
    nca.eval()
    nca_with_noise.eval()
    mix_nca.eval()
    stochastic_mix_nca.eval()
    
    # Esegui i test
    results_df = compare_generated_distributions(
        histories=histories,
        standard_nca=nca.to("cuda:2"),
        mixture_nca=mix_nca.to("cuda:2"),
        stochastic_nca=stochastic_mix_nca.to("cuda:2"),
        nca_with_noise=nca_with_noise.to("cuda:2"),
        n_steps=35,
        n_evaluations=10,
        device="cuda:2",
        deterministic_rule_choice=False
    )
    
    # Aggiungi il neighborhood size ai risultati
    results_df['Neighborhood Size'] = nb_size
    
    # Salva i risultati
    all_results[nb_size] = results_df
    
    print(f"Completed testing for neighborhood size {nb_size}x{nb_size}")

# Aggrega tutti i risultati
import pandas as pd
if all_results:
    aggregated_df = pd.concat(all_results.values(), ignore_index=True)
    print("\nAggregated Results:")
    print(aggregated_df)
```

## Note Importanti

1. **Path dei Modelli**: Assicurati che i path dei modelli salvati corrispondano alla struttura delle directory create dagli script extended.

2. **Device**: Mantieni lo stesso device (es. "cuda:2") usato durante il training.

3. **Hyperparametri**: Usa gli stessi hyperparametri usati durante il training (hidden_dim, num_rules, temperature, etc.).

4. **Compatibilità**: I modelli ExtendedNCA con `neighborhood_size=3` sono compatibili con i modelli NCA base, ma è comunque meglio usare ExtendedNCA per consistenza.

## Verifica

Dopo le modifiche, verifica che:
- I modelli vengono caricati correttamente
- I test vengono eseguiti per ogni neighborhood size
- I risultati vengono salvati/visualizzati correttamente
- Non ci sono errori di dimensione dei tensori (dovuti a mismatch di neighborhood_size)

