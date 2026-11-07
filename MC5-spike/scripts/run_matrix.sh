#!/bin/bash
# MC5-1 spike measurement matrix (sequential: memory-bound, one run at a time).
# Q8: bf16 no-LM, bf16 + 0.6B LM, 4-bit DiT at 30 s / 120 s / 240 s.
# Q4: 3 prompts x {no-LM, 0.6B LM, 1.7B LM} at 30 s, fixed seed, plus one duplicate for reproducibility.
set -u
# VAE decode in fp16: the f32 MLX VAE peaks above the 18 GiB machine at 4 min (probe showed
# >=13.8 GB and heavy swap); upstream supports the switch (ACESTEP_MLX_VAE_FP16).
export ACESTEP_MLX_VAE_FP16=1
PY=/private/tmp/mc5/ACE-Step-1.5/.venv/bin/python
DIR=/private/tmp/mc5
M=$DIR/scripts/mc5_measure.py
OUT=$DIR/out
mkdir -p "$OUT/json" "$OUT/wav"
LOG=$OUT/matrix.log
echo "matrix start $(date)" >> "$LOG"

run() {
  local tag=$1; shift
  echo "=== $(date +%H:%M:%S) $tag" | tee -a "$LOG"
  "$PY" "$M" "$@" --tag "$tag" --out "$OUT/json/$tag.json" --wav-dir "$OUT/wav" >> "$OUT/$tag.stdout.log" 2>&1
  echo "--- exit $? $tag $(date +%H:%M:%S)" | tee -a "$LOG"
}

# ---- Q8: memory/time matrix (seed 12345, instrumental prompt)
for d in 30 120 240; do
  run "q8-bf16-nolm-${d}s" --config bf16 --lm none --duration "$d" --seed 12345 \
      --prompt-id instrumental --staged-free
done
for d in 30 120 240; do
  run "q8-bf16-lm06-${d}s" --config bf16 --lm 0.6B --duration "$d" --seed 12345 \
      --prompt-id instrumental --staged-free
done
for d in 30 120 240; do
  run "q8-4bit-nolm-${d}s" --config bf16-4bit --lm none --duration "$d" --seed 12345 \
      --prompt-id instrumental --staged-free
done

# ---- Q4: LM A/B listening set (seed 20251106, 30 s). LM runs use batch=2 so the LM's
#      seeded batch path (the only seeded one) makes the fixed seed meaningful.
for p in instrumental english_vocal chinese_vocal; do
  run "q4-nolm-${p}" --config bf16 --lm none --duration 30 --seed 20251106 \
      --prompt-id "$p" --staged-free
done
for p in instrumental english_vocal chinese_vocal; do
  run "q4-lm06-${p}" --config bf16 --lm 0.6B --duration 30 --seed 20251106 \
      --prompt-id "$p" --staged-free --batch 2
done
for p in instrumental english_vocal chinese_vocal; do
  run "q4-lm17-${p}" --config bf16 --lm 1.7B --duration 30 --seed 20251106 \
      --prompt-id "$p" --staged-free --batch 2
done

# ---- reproducibility check: same settings as q4-lm06-instrumental, fresh process
run "q4-lm06-instrumental-repro" --config bf16 --lm 0.6B --duration 30 --seed 20251106 \
    --prompt-id instrumental --staged-free --batch 2

echo "matrix done $(date)" >> "$LOG"
