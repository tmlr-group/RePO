# Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning

## Abstract

Large language models (LLMs) benefit substantially from supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR) in reasoning tasks. However, these recipes perform poorly in instruction-based molecular optimization, where each data point typically provides only a single optimized reference molecule and no step-by-step optimization trajectory. We reveal that answer-only SFT on the reference molecules collapses reasoning, and RLVR provides sparse feedback under similarity constraints due to the model's lack of effective exploration, which slows learning and limits optimization. To encourage the exploration of new molecules while balancing the exploitation of the reference molecules, we introduce **Re**ference-guided **P**olicy **O**ptimization (RePO), an optimization approach that learns from reference molecules without requiring trajectory data. At each update, RePO samples candidate molecules with their intermediate reasoning trajectories from the model and trains the model using verifiable rewards that measure property satisfaction under similarity constraints in an RL manner. Meanwhile, it applies reference guidance by keeping the policy’s intermediate reasoning trajectory as context and training only the answer in a supervised manner. Together, the RL term promotes exploration, while the guidance term mitigates reward sparsity and stabilizes training by grounding outputs to references when many valid molecular edits exist. Across molecular optimization benchmarks, RePO consistently outperforms SFT and RLVR baselines (e.g., GRPO), achieving improvements on the optimization metric (Success Rate $\times$ Similarity), improving balance across competing objectives, and generalizing better to unseen instruction styles.

## Quick Start

### 1) Environment

```bash
conda create -n repo python=3.10 -y
conda activate repo
pip install -r requirements.txt
pip install flash-attn
```

For MuMo multi-property training/evaluation, use a dedicated environment:

```bash
conda create -n mumo python=3.10 -y
conda activate mumo
pip install -r mumo_requirements.txt
```

### 2) MuMo Property Servers

MuMo training/evaluation depends on two local property APIs. Start both in `mumo` env (two terminals):

```bash
conda activate mumo
python multiprop_utils/admetModel_api.py
```

```bash
conda activate mumo
python multiprop_utils/drd2Model_api.py
```

Default endpoints:
- ADMET: `http://127.0.0.1:10086/predict/`
- DRD2: `http://127.0.0.1:10087/predict/`

### 3) RL Training

Use the unified training launcher:

```bash
bash scripts/run_RL_training.sh \
  --gpus 0,1,2 \
  --num_processes 2 \
  --entry src/x_r1/repo.py \
  --variant mumo \
  --config recipes/MulProp_3B_config.yaml \
  --output_dir ./output/repo_run
```

Common variants:
- `default`
- `mumo`
- `pure`
- `noisy_demo`
- `random_mask`

### 4) Single-Objective Testing

1) Generate Predictions

```bash
python generate_predictions.py \
  --model_path ./output/xxx/checkpoint-xxx \
  --benchmark open_generation \
  --task MolOpt \
  --subtask LogP \
  --output_dir ./predictions/
```

Notes:
- Outputs include CSV and JSONL artifacts under `./predictions/`.

2) Evaluate Predictions

```bash
bash scripts/run_full_evaluation.sh \
  --model_path ./output/repo_run/checkpoint-xxx \
  --task MolOpt \
  --subtasks LogP,MR,QED \
  --output_dir ./predictions/ \
  --device_id 0
```

### 5) Multi-Objective Testing

Use `mumo` env and ensure both MuMo property servers are running.

1) Inference (`inf.py`) on MuMo test data:

```bash
conda activate mumo
python inf.py \
  --model_path ./output/xxx/checkpoint-xxx \
  --data_path data/TEST_multi_prop/IND_sft_seen_bbbp+drd2+plogp_test_data.json \
  --output_dir ./output/xxx/checkpoint-xxx/bdp_repo \
  --output_name processed_IND_seen_bbbp+drd2+plogp_test_data.json \
  --gpu_memory_utilization 0.9
```

2) MuMo evaluation (`mumo_evaluate.py`):

```bash
conda activate mumo
python mumo_evaluate.py \
  --property_setting bbbp+drd2+plogp \
  --seen_setting seen \
  --IND_setting IND \
  --experiment_prefix bdp \
  --method_name _repo \
  --output_folder MuMo_performance
```

Outputs include:
- `MuMo_performance/detailed_results_*.csv`
- `MuMo_performance/avg_differences_*.csv`
- `{experiment_dir}/{IND}_sft_{seen}_{property}.json`

### 6) Utility Scripts

- `mumo_evaluate.py`: computes MuMo-style success/similarity metrics from generated JSON.
- `cal.py`: small CSV utilities (`--mode count` / `--mode wsr`).
