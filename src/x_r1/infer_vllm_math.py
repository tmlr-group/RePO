import json
import os
from typing import Iterable, List, Tuple

import pandas as pd
from vllm import LLM, SamplingParams

from fire import Fire
from math_reward import compute_score

# System prompt for all tasks
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> \\boxed{{answer here}} </answer>"
)

def _build_chat_prompt(problem_text: str, tokenizer) -> str:
    """
    Build a chat-formatted prompt using the model's chat template, including the shared system prompt.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def _chunked(iterable: List[str], chunk_size: int) -> Iterable[List[str]]:
    """
    Yield successive chunks from the given iterable.
    """
    for start in range(0, len(iterable), chunk_size):
        yield iterable[start : start + chunk_size]


def _strip_prefix_if_present(value, key: str):
    """
    Make CLI robust by stripping a leading 'key=' if Fire forwards it positionally.
    """
    if isinstance(value, str) and value.startswith(f"{key}="):
        return value.split("=", 1)[1]
    return value

def compute_predictions_accuracy(
    parquet_path: str = "./data/math/test.parquet",
    predictions_jsonl: str = "./data/math/predictions_test.jsonl",
) -> float:
    """
    Compute accuracy of predictions JSONL using ground truth from the parquet file.
    The JSONL must contain records with the 'completion' field (as written by this module).
    Returns accuracy as a float in [0, 1].
    """
    # Load ground-truth solutions
    df = pd.read_parquet(parquet_path)
    if "solution" not in df.columns:
        raise ValueError("Expected column 'solution' in the parquet file to compute accuracy.")
    solutions: List[str] = df["solution"].astype(str).tolist()

    # Load predictions in the same order they were generated
    completions: List[str] = []
    with open(predictions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            completions.append(obj.get("completion", ""))

    if len(completions) != len(solutions):
        raise ValueError(f"Length mismatch: {len(completions)} completions vs {len(solutions)} solutions.")

    # Compute score via project-provided function
    scores = compute_score(completions=completions, solution=solutions)
    # scores expected to be a list of floats (0/1)
    if not scores:
        return 0.0
    accuracy = float(sum(scores) / len(scores))
    print(f"Accuracy: {accuracy:.4f} ({sum(scores)}/{len(scores)})")
    return accuracy

def generate_math_predictions_vllm(
    model_path: str = "./output/grpo-math/MATH-3B/checkpoint-209",
    parquet_path: str = "./data/math/test.parquet",
    output_jsonl: str = "./data/math/predictions_test.jsonl",
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.65,
    batch_size: int = 128,
) -> str:
    """
    Load a vLLM model in-process, generate completions for the math test parquet, and
    write a JSONL with fields: {"prompt": <prompt>, "completion": <completion>}.

    Returns the absolute path to the written JSONL file.
    """
    # Normalize potential 'key=value' positional strings from CLI
    model_path = _strip_prefix_if_present(model_path, "model_path")
    parquet_path = _strip_prefix_if_present(parquet_path, "parquet_path")
    output_jsonl = _strip_prefix_if_present(output_jsonl, "output_jsonl")
    max_new_tokens = int(_strip_prefix_if_present(max_new_tokens, "max_new_tokens"))
    temperature = float(_strip_prefix_if_present(temperature, "temperature"))
    top_p = float(_strip_prefix_if_present(top_p, "top_p"))
    tensor_parallel_size = int(_strip_prefix_if_present(tensor_parallel_size, "tensor_parallel_size"))
    gpu_memory_utilization = float(_strip_prefix_if_present(gpu_memory_utilization, "gpu_memory_utilization"))
    batch_size = int(_strip_prefix_if_present(batch_size, "batch_size"))

    # Initialize LLM
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        n=1,
    )

    # Load data
    df = pd.read_parquet(parquet_path)
    if "problem" not in df.columns:
        raise ValueError("Expected column 'problem' in the parquet file.")

    # Build prompts using chat template
    problems: List[str] = df["problem"].astype(str).tolist()
    prompts: List[str] = [_build_chat_prompt(p, tokenizer) for p in problems]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # Generate and write JSONL incrementally
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        offset = 0
        for prompt_chunk in _chunked(prompts, batch_size):
            outputs = llm.generate(prompt_chunk, sampling_params=sampling_params)
            for i, output in enumerate(outputs):
                # Each output may contain multiple candidates; we take the first
                completion_text = output.outputs[0].text if output.outputs else ""
                record = {
                    "prompt": prompt_chunk[i],
                    "completion": completion_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            offset += len(prompt_chunk)

    accuracy = compute_predictions_accuracy(parquet_path=parquet_path, predictions_jsonl=output_jsonl)
    print(f"===> Accuracy: {accuracy:.4f}")
    return os.path.abspath(output_jsonl)


if __name__ == "__main__":
    Fire(generate_math_predictions_vllm)
    # Example:
    # CUDA_VISIBLE_DEVICES=0 python src/x_r1/infer_vllm_math.py \
    #   model_path="./output/grpo-math/MATH-3B/checkpoint-209" \
    #   parquet_path="./data/math/test.parquet" \
    #   output_jsonl="./data/math/predictions_test.jsonl" \
    #   max_new_tokens=1024 temperature=0.7 top_p=0.9 tensor_parallel_size=1 gpu_memory_utilization=0.8 batch_size=128