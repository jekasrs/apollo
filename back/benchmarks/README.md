# Benchmarks

## `run_meld_modality_suite.py`

Сетка на одном **MELD** `samples.pkl`:

- **Keras**: CNN, DNN, LSTM × модальности `a`, `t`, `at`
- **Apollo**: `**apollo_rgcn`** (RGCN по типам рёбер) и `**apollo_gcn**` (GCN первого слоя) × те же модальности

Результат: `results/meld_modality_suite/` — подкаталоги, в каждом `metrics.json` (test accuracy, weighted F1, F1 по эмоциям); сводка `SUMMARY_TABLES.md` и `all_metrics.json`.

Запуск: см. `--help` и docstring в скрипте.