# DATA STRUCTURE
'''json
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
'''

# COMMANDS
'''
python -m vllm.entrypoints.openai.run_batch -i ./data/benchmarks/open_generation/MolOpt/QED/test_batch_inference.jsonl -o ./data/benchmarks/open_generation/MolOpt/QED/results.jsonl --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --enable-reasoning --reasoning-parser deepseek_r1
'''
import json
import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from fire import Fire


def read_dataset(dataset_path, dataset_type="test"):
    # Load the CSV file directly
    df = pd.read_csv(dataset_path)
    
    # Print basic information about the dataframe
    print(f"==> Loaded CSV with {len(df)} rows and columns: {df.columns.tolist()}")
    
    # Create a dataset dictionary with a train split
    dataset = DatasetDict({dataset_type: Dataset.from_pandas(df)})
    
    # Map column names to match expected format
    dataset = dataset.rename_column("Instruction", "problem")
    dataset = dataset.rename_column("molecule", "solution")
    
    # Log dataset information
    print(f"==> Loaded OpenMolIns dataset with {len(dataset[dataset_type])} examples")
    print(f"==> Sample example: {dataset[dataset_type][0]}")
    return dataset

def format_batch_inference(model_name, index,query, temperature=0.6):

    message = [
        {"role": "user", "content": query}
    ]
    requests = {
        "custom_id": f"request-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": message,
            "temperature": temperature,
            }
        }
    return requests

def main(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
        dataset_path="data/benchmarks/open_generation/MolOpt/QED/test.csv", 
        dataset_type="test", 
        temperature=0.6,
    ):
    dataset = read_dataset(dataset_path, dataset_type)
    saved_path = f"{os.path.dirname(dataset_path)}/{model_name}_{dataset_type}_batch_inference.jsonl"
    os.makedirs(os.path.dirname(saved_path), exist_ok=True)
    with open(saved_path, "w") as f:
        for i, example in enumerate(dataset[dataset_type]):
            query = example["problem"]
            f.write(json.dumps(format_batch_inference(model_name, i, query, temperature)) + "\n")

    print(f"==> Saved {len(dataset[dataset_type])} requests to {saved_path}")

if __name__ == "__main__":
    Fire(main)