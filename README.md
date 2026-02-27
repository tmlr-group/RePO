# Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning

## Abstract

Large language models (LLMs) benefit substantially from supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR) in reasoning tasks. However, these recipes perform poorly in instruction-based molecular optimization, where each data point typically provides only a single optimized reference molecule and no step-by-step optimization trajectory. Answer-only SFT collapses reasoning, and RLVR provides sparse feedback under similarity constraints. We introduce \emph{Reference-guided Policy Optimization} (RePO), an objective that learns from reference molecules without requiring trajectory data. At each update, RePO samples multiple candidate molecules together with their intermediate reasoning trajectories from the current policy, and scores them using verifiable rewards that measure property satisfaction under similarity constraints. It then updates the policy to prefer higher-reward candidates and avoid lower-reward ones. Meanwhile, it applies reference guidance by keeping the policy’s intermediate reasoning trajectory as context and training only the final molecule toward the provided reference. This anchors the final molecule to the reference one without constraining the reasoning trajectory, thereby preserving optimization diversity.

## Quick Start

### 1) Environment

```bash
conda create -n repo python=3.10 -y
conda activate repo
pip install -r requirements.txt
# optional (GPU-specific)
pip install flash-attn
```

### 2) RL Training

Use the unified training launcher:

```bash
bash scripts/run_RL_training.sh \
  --gpus 0,1,2 \
  --num_processes 2 \
  --entry src/x_r1/grpo.py \
  --variant mumo \
  --config recipes/MulProp_3B_config.yaml \
  --output_dir ./output/grpo_run
```

Common variants:
- `default`
- `mumo`
- `pure`
- `noisy_demo`
- `random_mask`

### 3) Generate Predictions

```bash
python generate_predictions.py \
  --model_path ./output/grpo_run/checkpoint-xxx \
  --benchmark open_generation \
  --task MolOpt \
  --subtask LogP \
  --output_dir ./predictions/ \
  --lang en
```

Notes:
- `--lang en|cn` controls prompt language.
- Outputs include CSV and JSONL artifacts under `./predictions/`.

### 4) Evaluate Predictions

```bash
bash scripts/run_full_evaluation.sh \
  --model_path ./output/grpo_run/checkpoint-xxx \
  --task MolOpt \
  --subtasks LogP,MR,QED \
  --output_dir ./predictions/ \
  --device_id 0
```

### 5) Utility Scripts

- `mumo_evaluate.py`: computes MuMo-style success/similarity metrics from generated JSON.
- `cal.py`: small CSV utilities (`--mode count` / `--mode wsr`).
- `src/x_r1/infer_vllm_math.py`: math inference + accuracy utility.
