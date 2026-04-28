#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/skyarena_logs
LOG_FILE="/tmp/skyarena_logs/eval_stage0_cpu_$(date +%Y%m%d_%H%M%S).log"
PYTHON_BIN="${PYTHON:-python}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" scripts/train_mappo.py \
  --config configs/mappo_skyarena_stage0_no_attack.yaml \
  --device cpu \
  --total_env_steps 512 \
  --eval_interval 512 \
  --disable_gui_eval \
  2>&1 | tee "$LOG_FILE"
