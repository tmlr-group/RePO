#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Optional: ensure CUDA/C++ libs from conda env are visible
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Defaults (override via env vars or CLI flags)
# export WANDB_MODE="${WANDB_MODE:-offline}"
GPUS="${GPUS:-5,6,7}"
ENTRY="${ENTRY:-src/x_r1/depo.py}"
VARIANT="${VARIANT:-}"
CONFIG="${CONFIG:-recipes/OpenMolIns_3B_config.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/zero3.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
PORT="${PORT:-29500}"
LOG_DIR="${LOG_DIR:-./logs}"
LOG_FILE="${LOG_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"

usage() {
  cat <<'EOF'
Run RL training (GRPO / DePO variants) via accelerate.

Usage:
  bash scripts/run_RL_training.sh [--gpus 0,1] [--entry src/x_r1/grpo.py] [--config recipes/XXX.yaml]
                                 [--variant default|math|mumo|pure|noisy_demo|random_mask]
                                 [--accelerate_config recipes/zero3.yaml] [--num_processes 2]
                                 [--output_dir /path/to/save_dir]
                                 [--port 29500] [--log_dir ./logs] [--log_file ./logs/your.log]

Examples:
  # Default OpenMolIns 3B GRPO training
  bash scripts/run_RL_training.sh

  # DePO math training
  bash scripts/run_RL_training.sh --entry src/x_r1/depo.py --variant math --config recipes/MATH_3B_depo.yaml

  # Multi-property training
  bash scripts/run_RL_training.sh --entry src/x_r1/grpo.py --variant mumo --config recipes/MulProp_bbbp-drd2-qed_3B_config.yaml
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --entry) ENTRY="${2:?}"; shift 2 ;;
    --variant) VARIANT="${2:?}"; shift 2 ;;
    --config) CONFIG="${2:?}"; shift 2 ;;
    --accelerate_config) ACCELERATE_CONFIG="${2:?}"; shift 2 ;;
    --num_processes) NUM_PROCESSES="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --log_dir) LOG_DIR="${2:?}"; shift 2 ;;
    --log_file) LOG_FILE="${2:?}"; shift 2 ;;
    --output_dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"

if [[ -z "${LOG_FILE}" ]]; then
  entry_base="$(basename "$ENTRY" .py)"
  cfg_base="$(basename "$CONFIG" .yaml)"
  variant_tag="${VARIANT:-default}"
  base="${LOG_DIR}/${entry_base}_${variant_tag}_${cfg_base}"
  candidate="${base}.log"
  if [[ -e "$candidate" ]]; then
    i=2
    while [[ -e "${base}_${i}.log" ]]; do
      i=$((i + 1))
    done
    candidate="${base}_${i}.log"
  fi
  LOG_FILE="$candidate"
fi

mkdir -p "$(dirname "$LOG_FILE")"

echo "[run_RL_training.sh] GPUS=$GPUS" >&2
echo "[run_RL_training.sh] ENTRY=$ENTRY" >&2
echo "[run_RL_training.sh] VARIANT=$VARIANT" >&2
echo "[run_RL_training.sh] CONFIG=$CONFIG" >&2
echo "[run_RL_training.sh] ACCELERATE_CONFIG=$ACCELERATE_CONFIG" >&2
echo "[run_RL_training.sh] NUM_PROCESSES=$NUM_PROCESSES PORT=$PORT" >&2
echo "[run_RL_training.sh] Logging to: $LOG_FILE" >&2

EXTRA_ARGS=()
if [[ -n "$VARIANT" ]]; then
  EXTRA_ARGS+=(--variant "$VARIANT")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA_ARGS+=(--output_dir "$OUTPUT_DIR")
fi

NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
CUDA_VISIBLE_DEVICES="$GPUS" ACCELERATE_LOG_LEVEL="$ACCELERATE_LOG_LEVEL" \
  accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --main_process_port "$PORT" \
    --num_processes "$NUM_PROCESSES" \
    "$ENTRY" --config "$CONFIG" "${EXTRA_ARGS[@]}" \
    >"$LOG_FILE" 2>&1

echo "[run_RL_training.sh] Done. Log: $LOG_FILE" >&2