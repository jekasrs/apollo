# Сводная таблица экспериментов

Заполняйте вручную или скриптом после `eval.py`. Колонки — пример; добавляйте свои.

| run_id | dataset | modalities | use_pause | dev metric | best epoch | test acc | test wF1 | test macro F1 | checkpoint path | примечание |
|--------|---------|------------|-----------|------------|------------|----------|----------|---------------|-----------------|------------|
| apollo_meld_at_r01 | MELD | at | yes | weighted_f1 | — | — | — | — | results/apollo_meld_at_r01/model.pt | |
| | | | | | | | | | | |

Примечания:

- `run_id` совпадает с аргументом `--run-id` у `train.py`.
- Для честного test используйте веса из `model.pt` и один и тот же `samples.pkl`, что при обучении.
