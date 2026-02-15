#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Defaults (override via env vars or CLI flags)
DEVICE_ID="${DEVICE_ID:-0}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.95}"
GENERATOR="${GENERATOR:-generate_predictions.py}" # or generate_predictions_en.py / generate_predictions_cot.py
BENCHMARK="${BENCHMARK:-open_generation}"
TASK="${TASK:-MolOpt}"
SUBTASKS="${SUBTASKS:-}" # comma-separated; if empty, inferred from TASK
OUTPUT_DIR="${OUTPUT_DIR:-./predictions/}" # NOTE: this is a PREFIX, make sure it ends with '/'
SKIP_GENERATE=0
SKIP_EVAL=0

MODEL_PATHS=()
MODELS_FILE=""

usage() {
  cat <<'EOF'
Run batch inference + evaluation for one or more checkpoints.

Usage:
  bash scripts/run_full_evaluation.sh --model_path /path/to/ckpt [--model_path /path/to/ckpt2 ...]
                                     [--models_file path/to/models.txt]
                                     [--task MolOpt|MolEdit|MolCustom]
                                     [--subtasks LogP,MR,QED]
                                     [--benchmark open_generation]
                                     [--output_dir ./predictions/]
                                     [--device_id 0]
                                     [--gpu_memory_util 0.95]
                                     [--generator generate_predictions.py]
                                     [--skip_generate] [--skip_eval]

Notes:
  - output_dir is treated as a PREFIX. We will normalize it to end with '/' to match generate_predictions.py/evaluate.py.
  - If --subtasks is not provided, defaults are inferred from --task.

Examples:
  # MolOpt evaluation (LogP/MR/QED)
  bash scripts/run_full_evaluation.sh --model_path output/grpo-component/checkpoint-100 \
    --task MolOpt --subtasks LogP,MR,QED --output_dir ./predictions/

  # MolEdit evaluation (Add/Del/Sub)
  bash scripts/run_full_evaluation.sh --model_path output/grpo-component/checkpoint-100 \
    --task MolEdit --output_dir ./predictions/

  # Use english prompt generator
  bash scripts/run_full_evaluation.sh --model_path output/grpo-component/checkpoint-100 \
    --generator generate_predictions_en.py --task MolOpt --subtasks LogP,MR,QED
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model_path) MODEL_PATHS+=("${2:?}"); shift 2 ;;
    --models_file) MODELS_FILE="${2:?}"; shift 2 ;;
    --task) TASK="${2:?}"; shift 2 ;;
    --subtasks) SUBTASKS="${2:?}"; shift 2 ;;
    --benchmark) BENCHMARK="${2:?}"; shift 2 ;;
    --output_dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    --device_id) DEVICE_ID="${2:?}"; shift 2 ;;
    --gpu_memory_util) GPU_MEMORY_UTIL="${2:?}"; shift 2 ;;
    --generator) GENERATOR="${2:?}"; shift 2 ;;
    --skip_generate) SKIP_GENERATE=1; shift ;;
    --skip_eval) SKIP_EVAL=1; shift ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -n "$MODELS_FILE" ]]; then
  if [[ ! -f "$MODELS_FILE" ]]; then
    echo "error: models_file not found: $MODELS_FILE" >&2
    exit 1
  fi
  while IFS= read -r line; do
    [[ -z "${line// /}" ]] && continue
    [[ "$line" =~ ^# ]] && continue
    MODEL_PATHS+=("$line")
  done <"$MODELS_FILE"
fi

if [[ ${#MODEL_PATHS[@]} -eq 0 ]]; then
  echo "error: at least one --model_path (or --models_file) is required" >&2
  usage
  exit 1
fi

# Normalize output prefix (important because generate_predictions.py uses string concatenation)
OUTPUT_DIR="${OUTPUT_DIR%/}/"

if [[ -z "$SUBTASKS" ]]; then
  case "$TASK" in
    MolOpt) SUBTASKS="LogP,MR,QED" ;;
    MolEdit) SUBTASKS="AddComponent,DelComponent,SubComponent" ;;
    MolCustom) SUBTASKS="AtomNum,BondNum,FunctionalGroup" ;;
    *) echo "error: unknown task '$TASK' and --subtasks not provided" >&2; exit 2 ;;
  esac
fi

IFS=',' read -r -a SUBTASK_ARR <<<"$SUBTASKS"

echo "[run_full_evaluation.sh] Models: ${#MODEL_PATHS[@]}" >&2
echo "[run_full_evaluation.sh] TASK=$TASK SUBTASKS=$SUBTASKS BENCHMARK=$BENCHMARK" >&2
echo "[run_full_evaluation.sh] GENERATOR=$GENERATOR OUTPUT_DIR=$OUTPUT_DIR" >&2
echo "[run_full_evaluation.sh] DEVICE_ID=$DEVICE_ID GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL" >&2

for model_path in "${MODEL_PATHS[@]}"; do
  if [[ ! -d "$model_path" ]]; then
    echo "error: model path not found: $model_path" >&2
    exit 1
  fi

  for subtask in "${SUBTASK_ARR[@]}"; do
    if [[ $SKIP_GENERATE -eq 0 ]]; then
      echo "[generate] $TASK-$subtask | model=$model_path" >&2
      CUDA_VISIBLE_DEVICES="$DEVICE_ID" python "$GENERATOR" \
        --model_path "$model_path" \
        --benchmark "$BENCHMARK" \
        --task "$TASK" \
        --subtask "$subtask" \
        --output_dir "$OUTPUT_DIR" \
        --gpu_memory_utilization "$GPU_MEMORY_UTIL"
    fi

    if [[ $SKIP_EVAL -eq 0 ]]; then
      echo "[evaluate] $TASK-$subtask | model=$model_path" >&2
      CUDA_VISIBLE_DEVICES="$DEVICE_ID" python evaluate.py \
        --model_path "$model_path" \
        --benchmark "$BENCHMARK" \
        --task "$TASK" \
        --subtask "$subtask" \
        --output_dir "$OUTPUT_DIR"
    fi
  done
done

echo "[run_full_evaluation.sh] Done." >&2