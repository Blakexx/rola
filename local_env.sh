# Local LM data on the D: SSD (216 GB), NOT the Ubuntu vhdx on C: (~22 GB free).
# Keeps HF downloads + checkpoints off C: so the vhdx stops growing.
# Usage:  source local_env.sh   (before train_lm.py / eval_lm.py)
export ROLA_DATA=/mnt/d/rola_data
export HF_HOME="$ROLA_DATA/hf"
export HF_DATASETS_CACHE="$ROLA_DATA/hf/datasets"
export HF_HUB_CACHE="$ROLA_DATA/hf/hub"
export ROLA_LM_RUNS="$ROLA_DATA/lm_runs"   # pass to train_lm.py: --out_dir "$ROLA_LM_RUNS/<name>"
echo "[local_env] HF cache + checkpoints -> $ROLA_DATA (D: SSD)"
