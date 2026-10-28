# Analisi del codice di `submission_clean.py`

Riferimento per la pulizia del codice (solo codice, non il testo). Analisi su `bc8d1b7` ("refactoring", 1812 righe);
i numeri di riga della sezione 2-5 sono quelli di quella versione, **prima** delle modifiche della sezione 2.

Stato: `[x]` fatto, `[ ]` aperto, `[~]` valutato e lasciato com'è (motivo indicato).

## 1. Strumenti

| strumento | esito |
|---|---|
| ruff, regole di default (nessuna configurazione nel progetto) | 35 errori: 31 x E702 (punto e virgola), E401, E402, E731, E741 |
| ruff esteso (F, B, PLW, PERF, ERA, ARG...) | nessun F401/F821: nessun import inutilizzato e nessun nome non definito |
| vulture, confidenza 0 | 5 voci, tutte falsi positivi (vedi sotto) |
| ty | 1 diagnostica, riga 1788: `ax.text(x_, m + 0.02, ...)` (tipo di pandas, innocua) |
| analisi AST | nessuna funzione irraggiungibile, nessuna definita due volte |
| `.py` contro `.ipynb` | allineati (0 righe di differenza) |

Comandi: `uvx ruff check code/submission_clean.py`, `uvx vulture --min-confidence 0 code/submission_clean.py`,
`uvx ty check code/submission_clean.py`.

Falsi positivi di vulture: `benchmark` e `deterministic` (flag di `torch.backends`, righe 182-183), `forward`
(metodo di `nn.Module`, righe 431 e 482), `inp` (argomento obbligato dell'hook di PyTorch, riga 504).

## 2. Codice morto o inutile

- [x] **`RECIPE` era un no-op.** Usata solo in `FINAL_CONFIG = {..., **RECIPE, ...}` e tutte e quattro le sue chiavi
  (`weight_decay`, `label_smoothing`, `dropout`, `random_erasing`) venivano sovrascritte subito dopo. Rimossa la costante
  e il `**RECIPE`. Verificato che `FINAL_CONFIG` resta identico (14 chiavi, stessi valori).
- [x] **`fit(..., verbose=True)`**: il parametro non era mai passato a `False`. Tolto.
- [x] **`result_path`** era usata una sola volta (in `run_experiment`): inlined.
- [x] **Variabile `kind` inutilizzata** nel ciclo delle ablation (`for name, (kind, over) ...`): ora `(_, over)`.
- [~] **`count_gflops`**: non è morta, la usa la tabella parametri/GFLOPs. Va solo notato che `run_experiment` non salva
  più `gflops` nel JSON (i run vecchi lo hanno, quelli nuovi no).
- [~] **`final` (variabile)**: è usata nel `print` della stessa cella, quindi non è morta.
- [x] **Tre modi per dire "il modello finale"**: `FINAL_NET`/`FINAL_CONFIG` restano l'unica fonte (training e
  ablation). Rimossi `final_network()` e il controllo di coerenza dopo `make_ablations`; il confronto tra la scelta
  della regola di selezione e `FINAL_NET`/`FINAL_CONFIG` (con WARNING) sta ora nella cella dei risultati del longer
  training, accanto a `select(...)`.
- [~] **Costanti e helper usati una volta sola**: `summary`, `simplicity`, `STEPS`/`cumulative`, `LR_SWEEP`,
  `TUNE_LRS`, `RETRAIN`, `RESULTS_ARCHIVE_ID`/`FETCH_RESULTS`/`fetch_results`. Sono configurazione o funzioni
  leggibili: non ne ho tolto nessuna.

## 3. Riferimenti fragili tra celle (stato globale implicito)

- [x] **`grid` era sia una variabile sia una funzione.** La funzione 2B è ora `grid_table`; la variabile locale del
  ciclo di tuning (riga 1032) resta `grid`.
- [ ] **`names`** è riassegnata in 6 punti (righe 923, 1033, 1205, 1385, 1699, 1780); le celle alle righe 1068 e 1129
  dipendono dal valore della riga 1033.
- [ ] **Altre variabili globali riusate**: `groups` (4 volte), `cfg`, `reference` (un run in una cella, un dict di
  override in un'altra), `best`, `selected`, `t`, `n`.
- [ ] **`finalists()` può restituire `None`** e ogni chiamante fa `or []` (righe 1033, 1174, 1205).
- [ ] **`_RUNS` globale mutabile** (righe 661, 692, 701, 724), con `global` in due funzioni (PLW0603). Sostituire con
  `@cache` su `load_runs` e `cache_clear()` dopo ogni run.
- [ ] **Tre meccanismi di cache**: `_DATA` (dizionario a mano) accanto ai `@cache` di `load_split` e
  `train_statistics`. Uniformare con `@cache` su `get_data`.

## 4. Possibili incongruenze

- [ ] **`fetch_results` (riga 146)**: `tar.extractall` sovrascrive i JSON locali a ogni esecuzione; riallenando con
  `TRAIN = True` un run con lo stesso nome di uno già nell'archivio, il run nuovo viene perso al prossimo avvio. Inoltre
  scarica l'archivio a ogni esecuzione anche se i risultati sono già in locale (~2 MB, richiede rete e quota gdown).
- [ ] **`best_lr()` (riga 739)** restituisce un run (riga di tabella), non un learning rate.
- [ ] **Valori fissi duplicati**: `n_classes=100` come default in `AircraftNet` (439) e `build_model` (486) nonostante
  `N_CLASSES`; `batch_size=256` (statistiche) e `512` (validazione) sono numeri nel codice.
- [ ] **API privata di torchvision**: `self.base._labels` (riga 324) può rompersi tra versioni.
- [ ] **Chiavi di configurazione incoerenti**: `schedule` e `pretrained` non stanno in `BASE` e si leggono con
  `cfg.get(...)`, le altre con `cfg[...]`.
- [ ] **Step 6 (riga 1738)**: la condizione è `len(s6_runs) < 3` ma poi si accede a tre chiavi precise: con tre run
  diversi darebbe `KeyError`.
- [ ] **Formattazione mista**: 64 righe oltre 120 caratteri (la più lunga 873), 63 righe con punto e virgola;
  `ruff format --line-length 120` cambierebbe circa 600 righe.
- [ ] **Le immagini del testo** (6 `attachment:...png`) esistono solo dentro l'`.ipynb`: rigenerare l'`.ipynb` dal solo
  `.py` le perderebbe.

## 5. Pulizie minori (ruff)

- [ ] `import gdown, tarfile, tempfile` su una riga (E401, riga 147).
- [ ] `from functools import cache` a metà file (E402, riga 294).
- [ ] `same = lambda ...` (E731, riga 1291), variabile `l` (E741, riga 535).
- [ ] `x = x.double()` sovrascrive la variabile del ciclo (PLW2901, riga 343).
- [ ] 12 x `zip()` senza `strict=` (B905).
- [ ] Concatenazioni da sostituire con unpacking (RUF005, righe 600, 1081, 1111) e dict comprehension da sostituire con
  `dict.fromkeys` (C420, riga 1081).
- [ ] Marker `# [run]` residui (righe 1186, 1312, 1468): convenzione vecchia.
- [ ] Caratteri unicode ambigui (`x`, `-`) in stringhe e commenti (RUF001/RUF003).
- [ ] `float(m)` alla riga 1788 per far sparire l'unico avviso di `ty`.

## 6. Cosa è a posto

Nessun import inutilizzato, nessuna funzione raggiungibile solo da altre funzioni morte, nessun nome non definito,
nessun riferimento a funzioni rimosse (cercati `run_sweep`, `load_group`, `torchinfo`, `kagglehub` e altri), `.py` e
`.ipynb` allineati.

## 7. Verifiche dopo le modifiche della sezione 2

- `py_compile` e `uvx ty check`: invariati (1 diagnostica, la stessa); `uvx ruff check`: stessi 35 errori di stile;
  `uvx vulture`: stesse 5 voci.
- Notebook intero con `TRAIN = False` sui risultati veri: gira fino in fondo (15 figure).
- Percorso di training con dataset finto (2 epoche): `run_experiment` salva `t_s42_seed42.json` e la seconda chiamata
  carica il run salvato.
- `jupytext --sync`: `.ipynb` allineato, i 6 allegati delle immagini sono rimasti.
