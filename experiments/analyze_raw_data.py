"""
Script per analizzare i dati grezzi e identificare problemi statistici
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

# Path ai risultati
BASE_DIR = Path(__file__).parent / "results_extended" / "tissue_simulation_extended"
NEIGHBORHOOD_SIZES = [3, 4, 5, 6, 7]

print("="*80)
print("ANALISI DATI GREZZI - STATISTICAL TESTS")
print("="*80)

# Carica tutti i dati grezzi
raw_data_list = []
for nb_size in NEIGHBORHOOD_SIZES:
    exp_dir = BASE_DIR / f"NB_{nb_size}"
    raw_data_path = exp_dir / 'raw_metrics_data.pkl'
    if raw_data_path.exists():
        with open(raw_data_path, 'rb') as f:
            raw_data = pickle.load(f)
            df = pd.DataFrame(raw_data)
            raw_data_list.append(df)
            print(f"\n✓ Caricato NB_{nb_size}: {len(df)} righe")
    else:
        print(f"\n✗ File non trovato: {raw_data_path}")

if not raw_data_list:
    print("\nERRORE: Nessun dato trovato!")
    exit(1)

raw_df = pd.concat(raw_data_list, ignore_index=True)
print(f"\n{'='*80}")
print(f"TOTALE RIGHE: {len(raw_df)}")
print(f"MODELI: {raw_df['Model Type'].unique()}")
print(f"METRICHE: {[c for c in raw_df.columns if c not in ['Model Type', 'Neighborhood Size', 'Evaluation']]}")
print(f"{'='*80}\n")

metric_cols = ['KL Divergence', 'Chi-Square', 'Categorical MMD', 
              'Tumor Size Diff', 'Border Size Diff', 'Spatial Variance Diff']

# Analisi dettagliata per ogni modello e metrica
for model_type in raw_df['Model Type'].unique():
    print(f"\n{'='*80}")
    print(f"MODEL: {model_type}")
    print(f"{'='*80}")
    
    model_data = raw_df[raw_df['Model Type'] == model_type]
    
    for metric in metric_cols:
        if metric not in model_data.columns:
            continue
        
        print(f"\n--- {metric} ---")
        
        # Raggruppa per neighborhood size
        groups = {}
        for nb_size in sorted(model_data['Neighborhood Size'].unique()):
            group_data = model_data[model_data['Neighborhood Size'] == nb_size][metric].values
            group_data = group_data[~np.isnan(group_data)]
            if len(group_data) > 0:
                groups[nb_size] = group_data
                
                # Statistiche descrittive
                print(f"  NB{nb_size}: n={len(group_data)}, "
                      f"mean={np.mean(group_data):.6f}, "
                      f"std={np.std(group_data):.6f}, "
                      f"min={np.min(group_data):.6f}, "
                      f"max={np.max(group_data):.6f}")
                print(f"    Valori: {group_data}")
        
        if len(groups) < 2:
            print("  ⚠️  Meno di 2 gruppi disponibili, salto il test")
            continue
        
        # Verifica varianza zero
        zero_var_groups = [nb for nb, data in groups.items() if np.var(data) == 0]
        if zero_var_groups:
            print(f"  ⚠️  VARIANZA ZERO per: {zero_var_groups}")
            print(f"      Questi gruppi hanno tutti lo stesso valore!")
        
        # Verifica separazione completa
        group_list = list(groups.values())
        nb_list = list(groups.keys())
        
        # Kruskal-Wallis
        try:
            h_stat, p_value_kw = kruskal(*group_list)
            print(f"  Kruskal-Wallis: H={h_stat:.4f}, p={p_value_kw:.6f}")
        except Exception as e:
            print(f"  ⚠️  Kruskal-Wallis fallito: {e}")
            continue
        
        # Mann-Whitney U per ogni coppia
        print(f"  Mann-Whitney U tests:")
        for i, j in combinations(range(len(nb_list)), 2):
            nb_i, nb_j = nb_list[i], nb_list[j]
            data_i, data_j = group_list[i], group_list[j]
            
            try:
                u_stat, p_value_mw = mannwhitneyu(data_i, data_j, alternative='two-sided')
                
                # Verifica separazione completa
                all_i_less = np.all(data_i < data_j)
                all_i_greater = np.all(data_i > data_j)
                all_j_less = np.all(data_j < data_i)
                all_j_greater = np.all(data_j > data_i)
                
                separation = ""
                if all_i_less or all_j_greater:
                    separation = " [COMPLETE SEPARATION: all NB{} < all NB{}]".format(nb_i, nb_j)
                elif all_i_greater or all_j_less:
                    separation = " [COMPLETE SEPARATION: all NB{} > all NB{}]".format(nb_i, nb_j)
                
                # Verifica U estremi
                n1, n2 = len(data_i), len(data_j)
                max_u = n1 * n2
                u_warning = ""
                if u_stat == 0 or u_stat == max_u:
                    u_warning = " [EXTREME U: {}]".format(u_stat)
                
                print(f"    NB{nb_i} vs NB{nb_j}: U={u_stat:.2f}, p={p_value_mw:.6f}{separation}{u_warning}")
                
            except Exception as e:
                print(f"    ⚠️  NB{nb_i} vs NB{nb_j}: {e}")

print(f"\n{'='*80}")
print("ANALISI COMPLETATA")
print(f"{'='*80}\n")


