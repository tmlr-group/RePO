#!/usr/bin/env python
# Example:
# CUDA_VISIBLE_DEVICES=1 python inf.py \
#   --model_path ./output/bbbp+plogp+qed-3B-Instruct-grpo \
#   --data_path ./data/IND_Test/IND_sft_seen_bbbp+plogp+qed_test_data.json \
#   --output_dir ./bpq \
#   --output_name processed_IND_seen_bbbp+plogp+qed_test_data.json

import json
import os
import argparse
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Run vLLM inference for each sample in the input JSON.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    args = parser.parse_args()

    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

    with open(args.data_path, "r") as f:
        data = json.load(f)

    prompts = [item.get("instruction", "") for item in data]
    outputs = llm.generate(prompts, params)

    # Build result list by attaching generated outputs to each original sample.
    result_data = []
    for item, output in zip(data, outputs):
        new_item = item.copy()
        new_item["vllm_output"] = output.outputs[0].text if output.outputs else ""
        result_data.append(new_item)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

if __name__ == "__main__":
    main()