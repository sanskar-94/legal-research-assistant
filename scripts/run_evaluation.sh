#!/bin/zsh
# Final evaluation of Nyaya-Gen, the local answer writer that replaces the API LLM.
#   nohup caffeinate -i scripts/run_evaluation.sh > logs/evaluation.log 2>&1 &
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
OUT=data/evaluation

echo "[$(date +%T)] 1/3 Run D: Nyaya-Gen (fine-tuned SmolLM2-135M, fully local)"
LLM_BACKEND=local $PY scripts/evaluate.py --output $OUT/eval_results_local.json

echo "[$(date +%T)] 2/3 Retrieval tables III, VI-VIII"
$PY scripts/retrieval_experiments.py --output $OUT/retrieval_experiments.json

echo "[$(date +%T)] 3/3 Run D0: base SmolLM2-135M (not fine-tuned), for comparison"
[ -f models/smollm2-base/config.json ] || $PY -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
b='HuggingFaceTB/SmolLM2-135M-Instruct'
AutoTokenizer.from_pretrained(b).save_pretrained('models/smollm2-base')
AutoModelForCausalLM.from_pretrained(b).save_pretrained('models/smollm2-base')"
LLM_BACKEND=local NYAYA_GEN_PATH=models/smollm2-base $PY scripts/evaluate.py --output $OUT/eval_results_local_base.json
echo "[$(date +%T)] EVALUATION DONE"
