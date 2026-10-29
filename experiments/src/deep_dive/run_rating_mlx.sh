#!/bin/bash
# PAIEC Qwen3-14B (MLX 4-bit) item rating under the GPU lock (COMPUTE_POLICY.md); waits for the lock to be free.
set -u
LOCK="$HOME/neurips/GPU_LOCK"
W="$HOME/neurips/paiec"
PY="$HOME/neurips/mlx_venv/bin/python"
OUT="$W/ratings_qwen3_14b.jsonl"
MODEL="$HOME/neurips/hf_cache/hub/models--mlx-community--Qwen3-14B-4bit/snapshots/a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4"
while [ -e "$LOCK" ]; do sleep 60; done
free=$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
while [ "${free:-0}" -lt 30 ]; do echo "$(date '+%F %T') waiting: memory free ${free}%"; sleep 120; free=$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); done
if ! mkdir "$LOCK" 2>/dev/null; then echo "$(date '+%F %T') lock taken by someone else first: $(cat "$LOCK/owner" 2>/dev/null)"; exit 2; fi
echo "paiec_item_rating_q14 pid=$$ since=$(date '+%Y-%m-%d %H:%M:%S')" > "$LOCK/owner"
trap 'rm -rf "$LOCK"; echo "$(date "+%F %T") lock released"' EXIT
echo "$(date '+%F %T') lock taken; memory free ${free}%"
export HF_HOME="$HOME/neurips/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# repair a possibly truncated last line (crash) so appended records start on a new line
[ -s "$OUT" ] && "$PY" -c "import sys;p=sys.argv[1];b=open(p,'rb').read();open(p,'ab').write(b'\n') if b and not b.endswith(b'\n') else None" "$OUT"
ORDER="$(cat "$W/rating_order.txt")"
"$PY" -I "$W/rate_items_mlx.py" --items "$W/deep_dive_items.jsonl" --out "$OUT" --order "$ORDER" --model "$MODEL" --limit 20
n_ok=$("$PY" -c "import json,os;r=[json.loads(l) for l in open('$OUT')] if os.path.exists('$OUT') else [];print(sum(1 for x in r if x.get('s1') is not None and not x.get('error')))")
echo "$(date '+%F %T') smoke: $n_ok of 20 ok"
if [ "${n_ok:-0}" -lt 18 ]; then echo "smoke failed; not continuing"; exit 5; fi
"$PY" -I "$W/rate_items_mlx.py" --items "$W/deep_dive_items.jsonl" --out "$OUT" --order "$ORDER" --model "$MODEL"
echo "$(date '+%F %T') rate_items_mlx exit $?"
