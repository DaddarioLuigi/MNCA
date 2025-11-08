# Guida agli Esperimenti Extended con Diversi Neighborhood Sizes

Questa guida spiega come eseguire gli stessi esperimenti del paper principale ma con diversi neighborhood sizes (3x3, 4x4, 5x5, 6x6, 7x7).

## File Disponibili

### 1. **emoji_experiment_extended.py**
- **Scopo**: Training e test di robustezza su pattern emoji
- **Dati**: Singola immagine emoji (40x40 pixel)
- **Modelli**: Standard NCA, Mixture NCA (6 regole), Mixture NCA with Noise (6 regole)

### 2. **cifar_experiment_extended.py**
- **Scopo**: Training e test di robustezza su immagini CIFAR-10
- **Dati**: Singola immagine da categoria CIFAR-10 (32x32 pixel)
- **Modelli**: Standard NCA, Mixture NCA (6 regole), Mixture NCA with Noise (6 regole)

### 3. **tissue_simulation_extended.py** (NUOVO)
- **Scopo**: Training e valutazione su simulazioni biologiche
- **Dati**: 200 simulazioni di tessuti (35 step temporali, 6 tipi di cellule)
- **Modelli**: Standard NCA, NCA with Noise, Mixture NCA (5 regole), Mixture NCA with Noise (5 regole)

## Hyperparametri (Identici al Paper Principale)

### Esperimenti con Immagini (Emoji/CIFAR-10)
```python
TOTAL_STEPS = 8000
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
POOL_SIZE = 1000
NUM_STEPS = [30, 50]
HIDDEN_DIM = 128 (emoji) / 64 (CIFAR-10)
N_RULES = 6
MILESTONES = [4000, 6000, 7000]
GAMMA = 0.2
DECAY = 3e-5
DROPOUT = 0.2
```

### Esperimenti con Simulazioni Biologiche
```python
n_epochs = 800
time_length = 35
update_every = 1
learning_rate = 0.001
milestones = [500]
gamma = 0.1
hidden_dim = 128
n_rules = 5
state_dim = 6
n_cell_types = 6
```

## Come Eseguire gli Esperimenti

### 1. Esperimenti Emoji

```bash
cd experiments

# Eseguire con tutti i neighborhood sizes (3,4,5,6,7)
python emoji_experiment_extended.py \
    --emoji_code 1F60A \
    --neighborhood_sizes 3,4,5,6,7 \
    --output_dir results_extended

# Oppure solo alcuni sizes
python emoji_experiment_extended.py \
    --emoji_code 1F60A \
    --neighborhood_sizes 3,5,7 \
    --output_dir results_extended

# Modalità quick (per test rapidi)
python emoji_experiment_extended.py \
    --emoji_code 1F60A \
    --neighborhood_sizes 3,4,5,6,7 \
    --quick \
    --output_dir results_extended
```

**Output**: 
- `results_extended/experiment_1F60A_robustness/NB_3/`
- `results_extended/experiment_1F60A_robustness/NB_4/`
- `results_extended/experiment_1F60A_robustness/NB_5/`
- `results_extended/experiment_1F60A_robustness/NB_6/`
- `results_extended/experiment_1F60A_robustness/NB_7/`

Ogni directory contiene:
- Modelli addestrati (`.pt`)
- Loss history (`.json`)
- Training curves (`.png`)
- Risultati robustezza (`.pkl`)
- Grafici (`.png`)
- CSV con metriche (`detailed_metrics.csv`)

### 2. Esperimenti CIFAR-10

```bash
cd experiments

# Eseguire con tutti i neighborhood sizes
python cifar_experiment_extended.py \
    --category 0 \
    --data_dir ../data \
    --neighborhood_sizes 3,4,5,6,7 \
    --output_dir results_extended

# Per una categoria specifica (0-9)
python cifar_experiment_extended.py \
    --category 3 \
    --data_dir ../data \
    --neighborhood_sizes 3,4,5,6,7 \
    --output_dir results_extended
```

**Output**: 
- `results_extended/experiment_{category}_0_robustness/nb_3/`
- `results_extended/experiment_{category}_0_robustness/nb_4/`
- ... (stessa struttura degli emoji)

### 3. Esperimenti Simulazioni Biologiche

```bash
cd experiments

# Eseguire con tutti i neighborhood sizes
python tissue_simulation_extended.py \
    --histories_path ../notebooks/histories.npy \
    --neighborhood_sizes 3,4,5,6,7 \
    --output_dir results_extended \
    --device cuda

# Con parametri personalizzati
python tissue_simulation_extended.py \
    --histories_path ../notebooks/histories.npy \
    --neighborhood_sizes 3,4,5,6,7 \
    --n_epochs 800 \
    --time_length 35 \
    --update_every 1 \
    --n_cell_types 6 \
    --device cuda:0 \
    --n_evaluations 10 \
    --output_dir results_extended
```

**Output**:
- `results_extended/tissue_simulation_extended/NB_3/`
- `results_extended/tissue_simulation_extended/NB_4/`
- `results_extended/tissue_simulation_extended/NB_5/`
- `results_extended/tissue_simulation_extended/NB_6/`
- `results_extended/tissue_simulation_extended/NB_7/`

Ogni directory contiene:
- Modelli addestrati (`.pt`)
- Metriche biologiche (`biological_metrics.csv`)
- Summary (`summary.json`)
- File aggregato: `all_neighborhood_sizes_metrics.csv`

## Struttura Output

```
results_extended/
├── experiment_1F60A_robustness/     # Emoji
│   ├── NB_3/
│   │   ├── standard_model.pt
│   │   ├── mixture_model.pt
│   │   ├── mixture_model_noise.pt
│   │   ├── loss_history.json
│   │   ├── training_loss.png
│   │   ├── deletion_5.pkl
│   │   ├── deletion_10.pkl
│   │   ├── noise_0.1.pkl
│   │   ├── noise_0.25.pkl
│   │   ├── pixel_removal_100.pkl
│   │   ├── pixel_removal_500.pkl
│   │   └── detailed_metrics.csv
│   ├── NB_4/
│   ├── NB_5/
│   ├── NB_6/
│   └── NB_7/
│
├── experiment_airplane_0_robustness/  # CIFAR-10
│   └── (stessa struttura)
│
└── tissue_simulation_extended/        # Simulazioni biologiche
    ├── NB_3/
    │   ├── standard_nca.pt
    │   ├── nca_with_noise.pt
    │   ├── mixture_nca.pt
    │   ├── stochastic_mix_nca.pt
    │   ├── biological_metrics.csv
    │   └── summary.json
    ├── NB_4/
    ├── NB_5/
    ├── NB_6/
    ├── NB_7/
    └── all_neighborhood_sizes_metrics.csv
```

## Confronto tra Neighborhood Sizes

Dopo aver eseguito tutti gli esperimenti, puoi confrontare i risultati:

1. **Per esperimenti immagini**: Leggi i file `detailed_metrics.csv` da ogni `NB_X/` e combinali
2. **Per simulazioni biologiche**: Usa il file `all_neighborhood_sizes_metrics.csv` che viene generato automaticamente

## Note Importanti

1. **Tempo di esecuzione**: Ogni neighborhood size richiede un training completo. Per 5 sizes, il tempo sarà ~5x quello di un singolo esperimento.

2. **Memoria**: Neighborhood sizes più grandi (6x6, 7x7) richiedono più memoria GPU.

3. **Neighborhood Size 3**: Usa i filtri base del NCA standard (compatibile al 100% con il paper originale).

4. **Reproducibilità**: Tutti gli script usano seed fisso (SEED=3) per garantire riproducibilità.

## Testare i Modelli nei Notebook

Dopo aver addestrato i modelli con diversi neighborhood sizes, puoi testarli usando i notebook esistenti. 

**IMPORTANTE**: Devi modificare i notebook per usare le classi `ExtendedNCA`, `ExtendedMixtureNCA`, e `ExtendedMixtureNCANoise` invece delle classi base.

Vedi la guida dettagliata in:
- `../notebooks/GUIDA_MODIFICA_NOTEBOOK_EXTENDED.md` - Guida completa con esempi
- `../notebooks/esempio_test_extended.py` - Codice Python di esempio

### Modifiche Rapide:

1. **Sostituisci le importazioni**:
   ```python
   # Da:
   from mix_NCA.NCA import NCA
   # A:
   from mix_NCA.ExtendedNCA import ExtendedNCA
   ```

2. **Aggiungi `neighborhood_size`** quando inizializzi i modelli:
   ```python
   nca = ExtendedNCA(..., neighborhood_size=3)  # o 4, 5, 6, 7
   ```

3. **Carica i modelli dal path corretto**:
   ```python
   model_path = f"../results_extended/experiment_TISSUE_SIMULATION/NB_{nb_size}/standard_nca.pt"
   ```

## Troubleshooting

- **Out of Memory**: Riduci `batch_size` o `pool_size` per neighborhood sizes grandi
- **Training lento**: Usa `--quick` per test rapidi (riduce steps e runs)
- **File non trovato**: Verifica i path relativi (es. `../notebooks/histories.npy`)
- **Errori nei notebook**: Assicurati di aver modificato le importazioni e aggiunto `neighborhood_size` ai modelli

