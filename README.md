# RePO

## DePO Quick Start

### 1) Environment

```bash
conda create -n depo python=3.10 -y
conda activate depo
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
- `math`
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
