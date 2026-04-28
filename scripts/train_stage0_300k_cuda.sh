#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/skyarena_logs
LOG_FILE="/tmp/skyarena_logs/train_stage0_300k_cuda_$(date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${PYTHON:-python}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/train_mappo.py \
  --config configs/mappo_skyarena_stage0_no_attack.yaml \
  --device cuda \
  --total_env_steps 300000 \
  --eval_interval 50000 \
  --gui_eval_human \
  2>&1 | tee "$LOG_FILE"
