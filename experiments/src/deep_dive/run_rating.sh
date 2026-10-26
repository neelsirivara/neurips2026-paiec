#!/bin/bash
# PAIEC item rating on the Mac under the GPU lock (COMPUTE_POLICY.md). Usage: run_rating.sh [extra rate_items.py args]
set -u
LOCK="$HOME/neurips/GPU_LOCK"
W="$HOME/neurips/paiec"
if ! mkdir "$LOCK" 2>/dev/null; then echo "$(date '+%F %T') GPU lock held by: $(cat "$LOCK/owner" 2>/dev/null)"; exit 2; fi
echo "paiec_item_rating pid=$$ since=$(date '+%Y-%m-%d %H:%M:%S')" > "$LOCK/owner"
trap 'rm -rf "$LOCK"; echo "$(date "+%F %T") lock released"' EXIT
free=$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
echo "$(date '+%F %T') lock taken; memory free ${free}%"
if [ "${free:-0}" -lt 30 ]; then echo "memory free ${free}% < 30%: not loading"; exit 3; fi
export HF_HOME="$HOME/neurips/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
"$HOME/neurips/mlenv/bin/python" -I "$W/rate_items.py" --items "$W/deep_dive_items.jsonl" --out "$W/${OUT:-ratings_qwen35_4b.jsonl}" \
  --backend hf --model "${MODEL:-Qwen/Qwen3.5-4B}" --threads 2 --order "$(cat "$W/rating_order.txt")" "$@"
echo "$(date '+%F %T') rate_items exit $?"
