#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

CHECKPOINT="${1:-}"
if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT="$(ls -1t train_dir/skyarena_mappo_train/stage0_no_attack_movement_v2/checkpoints/step_*.pt | head -n 1)"
fi

"$PYTHON_BIN" scripts/evaluate_mappo.py \
  --config configs/mappo_skyarena_stage0_no_attack.yaml \
  --checkpoint "$CHECKPOINT" \
  --episodes 3 \
  --device cuda \
  --max_steps 1200 \
  --gui_eval_human \
  --kind gui_eval_manual
