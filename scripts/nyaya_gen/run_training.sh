#!/bin/zsh
# Trains Nyaya-Gen, the local answer writer. Safe to run detached:
#   nohup caffeinate -i scripts/nyaya_gen/run_training.sh > logs/training.log 2>&1 &
# Progress: tail -f logs/training.log
# Settings are sized for an 8 GB M2 Air; close other apps while it runs.
cd "$(dirname "$0")/../.."
PY=./.venv/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
[ -f data/generator/train.jsonl ] || $PY scripts/nyaya_gen/prepare_generator_data.py --max-train 4000
echo "[$(date +%T)] Nyaya-Gen training"
$PY scripts/nyaya_gen/train_generator.py --grad-accum 8 --eval-every 50 --eval-examples 40 --max-minutes 75
echo "[$(date +%T)] TRAINING DONE"
