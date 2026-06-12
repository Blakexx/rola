#!/bin/bash
# LM A/B: global-sym vs global-asym vs per-state-asym, vanilla training, FineWeb-Edu.
# Tests at LM scale: (a) does asym have a training problem? (b) global vs per-state?
# 125M (hidden 768 / 12 layers / 12 heads), FineWeb-Edu slice on D: SSD, ~260M tokens/arm.
set -e
cd /mnt/c/Users/Blake/Documents/VSCode/CLA
source local_env.sh
PY=.venv/bin/python
for INST in rola-rla-sym rola-rla-asym rola-rla-asym-ps; do
  echo "==================== $INST  ($(date -u +%H:%M)Z) ===================="
  $PY train_lm.py \
    --rola_instance "$INST" \
    --hidden_size 768 --num_hidden_layers 12 --num_heads 12 \
    --num_states 16 --d_qk 16 --d_v 16 \
    --dataset fineweb-edu --max_train_samples 600000 \
    --block_size 1024 --batch_size 16 --grad_accum 4 \
    --lr 3e-4 --warmup 200 --max_steps 4000 --eval_steps 1000 \
    --out_dir "$ROLA_LM_RUNS/lmab_${INST}" \
  && echo "DONE $INST" || echo "FAIL $INST"
done
echo "==================== LM A/B complete ($(date -u +%H:%M)Z) ===================="
