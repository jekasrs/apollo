#!/usr/bin/env bash
# Полный цикл: bootstrap samples.pkl → MPNet MELD → Wav2Vec MELD → финальный preprocess с дообученными энкодерами.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT/back"
export PYTHONPATH="$PWD"
PY="${PYTHON:-python3}"

TEXT_OUT="${APOLLO_FT_TEXT_OUT:-$ROOT/results/encoders/finetune_mpnet_meld}"
W2V_OUT="${APOLLO_FT_WAV2VEC_OUT:-$ROOT/results/encoders/finetune_wav2vec_meld}"
TEXT_EPOCHS="${APOLLO_FT_TEXT_EPOCHS:-4}"
W2V_EPOCHS="${APOLLO_FT_WAV2VEC_EPOCHS:-5}"

echo "[1/4] Bootstrap preprocess (SBERT + backbone Wav2Vec, без APOLLO_FINETUNED_*) …"
unset APOLLO_FINETUNED_TEXT APOLLO_FINETUNED_WAV2VEC 2>/dev/null || true
"$PY" dataset/preprocess/preprocess.py

echo "[2/4] Fine-tune MPNet на MELD → $TEXT_OUT"
mkdir -p "$(dirname "$TEXT_OUT")"
"$PY" dataset/finetune/finetune_text_meld.py --out_dir "$TEXT_OUT" --epochs "$TEXT_EPOCHS"

echo "[3/4] Fine-tune Wav2Vec на MELD → $W2V_OUT (доп. аргументы: передать скрипту, напр. --device cpu)"
"$PY" dataset/finetune/finetune_wav2vec_meld.py --out_dir "$W2V_OUT" --epochs "$W2V_EPOCHS" "$@"

echo "[4/4] Финальный preprocess с дообученными энкодерами …"
export APOLLO_FINETUNED_TEXT="$TEXT_OUT"
export APOLLO_FINETUNED_WAV2VEC="$W2V_OUT"
"$PY" dataset/preprocess/preprocess.py

echo "Готово: dataset/preprocess/samples/samples.pkl"
