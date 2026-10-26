# implementations

- [x] gestione univoca dei download delle run e dei modelli tramite gdown
- [x] control runs (for bn) in ablation study
- [x] torch view diagram (part 1) [https://github.com/mert-kurttutan/torchview] -> Typst
- [x] diagrammini di juan
- [x] spiegazione sul perché l'ablation della batch norm è così impattante
- [x] mettere la visualizzazione della rete di ale al posto di torchinfo
- [x] decidere se tenere run experiments -> Si
- [x] aggiungere la parte 2b di miky (report: step 1-5, ladder, confronto; carica i run salvati)
  - [x] risultati di `final_long` e step 6 mancanti in `results/part2B` e nell'archivio condiviso (le celle li saltano)
  - [ ] rendere la 2B riallenabile (freeze warmup, layer decay, color jitter, Mixup/CutMix nel trainer CPU) oppure dichiarare che serve `part2B.py`
- [x] pulire le diagnostiche di `ty` in `submission_clean` (`uvx ty check code/submission_clean.py`: 0 diagnostiche)
- [ ] best-model -> retrain/dl results/dl wights
  - [ ] caricare modello su cartella condivisa

# to remove

- [x] drive and colab only libraries
- [x] caching
- [x] gpu acceleration
- [x] useless torchinfo prints (training)
- [x] mixed precision in CUDA (?)

# simplify

- [ ] test runner
- [ ] more function reuse
- [ ] spiegazioni e motivazioni
