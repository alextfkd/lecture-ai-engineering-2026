#!/usr/bin/env bash
# 同じ質問を、4モデル × 温度2種 × 3回 実行し、回答を lectures/01/ に残す。
# 使い方（リポジトリ直下でも、このディレクトリからでも可）:
#   bash lectures/01/compare.sh
#
# 無料枠のレート制限を避けるため、呼び出しのあいだに SLEEP_SEC 秒待つ。
# 1回だけ試すなら RUNS=1 SLEEP_SEC=0 bash lectures/01/compare.sh

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

QUESTION="${QUESTION:-LoRAとは何ですか？}"
OUT_DIR="lectures/01"
RUNS="${RUNS:-3}"
SLEEP_SEC="${SLEEP_SEC:-15}"

MODELS=(
  gemini-3.5-flash-lite
  gemini-3.1-flash-lite
  gemini-3.8-flash
)
TEMPS=(0 1)

mkdir -p "$OUT_DIR"

for model in "${MODELS[@]}"; do
  for temp in "${TEMPS[@]}"; do
    for run in $(seq 1 "$RUNS"); do
      out="${OUT_DIR}/ans_${model}_t${temp}_${run}.txt"
      echo "=== ${model}  temperature=${temp}  run=${run}  -> ${out}"
      AIE_MODEL="$model" AIE_TEMPERATURE="$temp" \
        uv run python -m app.ask "$QUESTION" > "$out" 2>&1
      status=$?
      if [ "$status" -ne 0 ]; then
        echo "failed (exit ${status}): ${out}"
      fi
      if [ "$SLEEP_SEC" -gt 0 ]; then
        sleep "$SLEEP_SEC"
      fi
    done
  done
done

echo "done: ${OUT_DIR}/ans_*.txt"
