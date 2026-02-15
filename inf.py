#!/usr/bin/env python
# Example:
# CUDA_VISIBLE_DEVICES=1 python inf.py \
#   --model_path ./output/bbbp+plogp+qed-3B-Instruct-grpo \
#   --data_path ./data/IND_Test/IND_sft_seen_bbbp+plogp+qed_test_data.json \
#   --output_dir ./bpq \
#   --output_name processed_IND_seen_bbbp+plogp+qed_test_data.json

from vllm import LLM, SamplingParams
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate repeated outputs from the first sample instruction.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    args = parser.parse_args()

    llm = LLM(model=args.model_path)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    # Use only the first sample's instruction and repeat it 100 times.
    prompts = [data[0]["instruction"]] * 100
    outputs = llm.generate(prompts, params)

    # Build a result list by cloning the first sample and attaching each generated output.
    result_data = []
    for i, output in enumerate(outputs):
        # Copy all fields from the first sample.
        new_item = data[0].copy()
        # Attach vLLM output text.
        new_item["vllm_output"] = output.outputs[0].text
        result_data.append(new_item)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)

if __name__ == "__main__":
    main()