# Report Analisi Statistica - Neighborhood Size Tests

## Problemi Identificati

### 1. **Standard NCA: Varianza Zero**
**Problema**: Standard NCA è completamente deterministico. Tutte le 10 valutazioni producono esattamente lo stesso valore per ogni metrica.

**Implicazioni**:
- I test statistici sono tecnicamente validi ma poco informativi
- Non c'è variabilità intrinseca da misurare
- Le differenze tra neighborhood sizes sono reali ma deterministiche

**Esempio**:
- KL Divergence NB3: tutti i 10 valori = 7.579689
- KL Divergence NB4: tutti i 10 valori = 1.689787
- Separazione completa: tutti i valori NB3 > tutti i valori NB4

### 2. **Separazione Completa (Complete Separation)**
**Problema**: Molti confronti mostrano separazione completa, dove tutti i valori di un gruppo sono più grandi/piccoli di tutti i valori dell'altro gruppo.

**Implicazioni**:
- U statistic = 0 o 100 (valori estremi)
- P-values molto piccoli ma potenzialmente artefatti
- Con n=10 per gruppo, la separazione completa è possibile ma sospetta

**Esempi**:
- Mixture NCA KL Divergence: NB5 vs NB6 mostra separazione completa (tutti NB5 < tutti NB6)
- Questo indica differenze molto marcate tra i gruppi

### 3. **Bug nella Correzione Bonferroni** ✅ CORRETTO
**Problema**: Il codice usava `alpha` (0.05) invece di `alpha_corrected` (0.005) per determinare la significatività dopo la correzione.

**Correzione Applicata**:
- Ora usa `alpha_corrected = alpha / n_comparisons` per il controllo di significatività
- Con 5 gruppi → 10 confronti → alpha_corrected = 0.05/10 = 0.005
- I p-values corretti vengono ancora moltiplicati per n_comparisons, ma ora il controllo usa il threshold corretto

### 4. **P-values Identici**
**Problema**: Molti test mostrano p-values identici (es. 0.000159, 0.001827).

**Spiegazione**:
- Con separazione completa e n=10, i p-values discreti sono limitati
- La correzione Bonferroni moltiplica questi valori, producendo valori identici
- Questo è normale con campioni piccoli e separazione completa

## Interpretazione dei Risultati

### Standard NCA
- **Tutte le metriche**: Varianza zero (deterministico)
- **Border Size Diff e Spatial Variance Diff**: Valori identici per tutti i neighborhood sizes → test non validi
- **Altre metriche**: Separazione completa tra tutti i gruppi → differenze reali ma deterministiche

### Modelli Stocastici (Mixture NCA, Stochastic Mixture NCA, NCA with Noise)
- **Varianza presente**: I modelli mostrano variabilità tra le valutazioni
- **Separazione completa frequente**: Indica differenze molto marcate tra neighborhood sizes
- **Alcuni confronti non significativi**: Dopo correzione Bonferroni, alcuni confronti non risultano significativi

## Raccomandazioni

### 1. **Per Standard NCA**
- Considerare che è deterministico: le differenze sono reali ma non c'è variabilità da testare
- Potrebbe essere più appropriato riportare semplicemente i valori delle metriche senza test statistici
- Oppure usare un singolo valore per ogni configurazione invece di 10 valutazioni identiche

### 2. **Per i Modelli Stocastici**
- I risultati sono validi ma con limitazioni:
  - Campioni piccoli (n=10) → potenza statistica limitata
  - Separazione completa → differenze molto marcate
  - Dopo correzione Bonferroni, alcuni confronti potrebbero non essere significativi

### 3. **Miglioramenti Futuri**
- Aumentare n_evaluations da 10 a 30+ per maggiore potenza statistica
- Considerare test alternativi più robusti ai pareggi
- Aggiungere analisi di effect size (non solo significatività)
- Visualizzare le distribuzioni per capire meglio le differenze

## Correzioni Applicate

1. ✅ Bug Bonferroni: ora usa `alpha_corrected` per determinare significatività
2. ✅ Warning per varianza zero
3. ✅ Note per separazione completa
4. ✅ Informazioni sulla correzione nell'output

## Conclusione

I risultati **hanno senso** ma con importanti caveat:
- Standard NCA: deterministico → test poco informativi
- Modelli stocastici: differenze reali ma con campioni piccoli
- Bug corretto: ora la correzione Bonferroni è applicata correttamente
- Separazione completa: indica differenze molto marcate, ma con n=10 è da interpretare con cautela

**Raccomandazione finale**: I risultati sono validi ma limitati dal campionamento piccolo. Considerare di aumentare n_evaluations per future analisi.

