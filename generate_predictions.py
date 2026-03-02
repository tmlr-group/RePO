#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate predictions with selectable prompt language (en/cn)."""

import argparse
import os
import re
import json

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


def extract_smiles(text):
    """Extract a SMILES string from model output text."""
    # Try extracting from <answer> tags first.
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    if answer_match:
        return answer_match.group(1).strip()
    
    # If no <answer> tags are found, fall back to a broad SMILES-like pattern.
    smiles_pattern = r'([A-Za-z0-9@\+\-\[\]\(\)\\\/%=#$\.]+)'
    smiles_matches = re.findall(smiles_pattern, text)
    
    if smiles_matches:
        # Return the longest candidate string.
        return max(smiles_matches, key=len)
    
    return ""

def generate_predictions(model_path, benchmark, task, subtask, output_dir, device="cuda", gpu_memory_utilization=0.8, lang="en"):
    """Generate predictions using the model."""
    try:
        # Build test data path.
        test_file = f"./data/benchmarks/{benchmark}/{task}/{subtask}/test.csv"
        
        # Ensure output directory exists.
        model_name = os.path.basename(model_path.rstrip('/'))
        # Sanitize model name.
        model_name = model_name.replace('--', '_').replace('/', '_')
        print(f"Using model: {model_path}")
        print(f"Model name: {model_name}")
        
        output_path = f"{output_dir}{model_name}/{benchmark}/{task}/"
        os.makedirs(output_path, exist_ok=True)
        print(f"Output directory: {output_path}")
        
        # Check whether test data file exists.
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test data file does not exist: {test_file}")
        
        # Load test data.
        print(f"Loading test data: {test_file}")
        test_data = pd.read_csv(test_file)
        print(f"Test data loaded successfully, total {len(test_data)} records")
        
        # Load model with vLLM.
        print(f"Loading model with vLLM: {model_path}")
        try:
            llm = LLM(
                model=model_path,
                dtype="bfloat16",
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=1,  # single GPU; adjust if needed
                trust_remote_code=True
            )
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            # Try alternative dtypes.
            try:
                print("Trying float16...")
                llm = LLM(
                    model=model_path,
                    dtype="float16",
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=1,
                    trust_remote_code=True
                )
                print("Model loaded successfully with float16")
            except Exception as e2:
                print(f"Failed to load model with float16: {e2}")
                # Try float32 as last fallback.
                try:
                    print("Trying float32...")
                    llm = LLM(
                        model=model_path,
                        dtype="float32",
                        gpu_memory_utilization=gpu_memory_utilization,
                        tensor_parallel_size=1,
                        trust_remote_code=True
                    )
                    print("Model loaded successfully with float32")
                except Exception as e3:
                    print(f"Failed to load model with float32: {e3}")
                    raise
        
        # Prepare generation parameters.
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )
        
        # Generate predictions.
        outputs = []
        prompts = []
        raw_outputs = []
        
        if lang == "cn":
            atom_header = "请设计一个分子，包含以下原子数量：\n"
            group_header = "请设计一个分子，包含以下官能团数量：\n"
            bond_header = "请设计一个分子，包含以下化学键数量：\n"
            smiles_suffix = "\n请以SMILES格式给出分子结构。"
            add_template = "请修改以下分子，添加一个{group}官能团：\n{molecule}\n\n请以SMILES格式给出修改后的分子结构。"
            del_template = "请修改以下分子，移除一个{group}官能团：\n{molecule}\n\n请以SMILES格式给出修改后的分子结构。"
            sub_template = "请修改以下分子，将一个{removed_group}官能团替换为一个{added_group}官能团：\n{molecule}\n\n请以SMILES格式给出修改后的分子结构。"
            molopt_template = "{instruction}\n分子SMILES: {molecule}\n\n请以SMILES格式给出优化后的分子结构。"
            prompt_desc = f"准备 {subtask} 提示"
        else:
            atom_header = "Please design a molecule with the following atom counts:\n"
            group_header = "Please design a molecule with the following functional group counts:\n"
            bond_header = "Please design a molecule with the following bond counts:\n"
            smiles_suffix = "\nPlease provide the molecule structure in SMILES format."
            add_template = "Please modify the following molecule by adding a {group} functional group:\n{molecule}\n\nPlease provide the modified molecule in SMILES format."
            del_template = "Please modify the following molecule by removing a {group} functional group:\n{molecule}\n\nPlease provide the modified molecule in SMILES format."
            sub_template = "Please modify the following molecule by replacing a {removed_group} functional group with a {added_group} functional group:\n{molecule}\n\nPlease provide the modified molecule in SMILES format."
            molopt_template = "{instruction}\nMolecule SMILES: {molecule}\n\nPlease provide the optimized molecule in SMILES format."
            prompt_desc = f"Prepare {subtask} prompts"

        # Build prompts.
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc=prompt_desc):
            # Build prompt by task/subtask.
            if task == "MolCustom":
                if subtask == "AtomNum":
                    prompt = atom_header
                    for atom in ['carbon', 'oxygen', 'nitrogen', 'sulfur', 'fluorine', 'chlorine', 'bromine', 'iodine', 'phosphorus', 'boron', 'silicon', 'selenium', 'tellurium', 'arsenic', 'antimony', 'bismuth', 'polonium']:
                        if atom in row and int(row[atom]) > 0:
                            prompt += f"- {atom}: {int(row[atom])}\n"
                    prompt += smiles_suffix
                
                elif subtask == "FunctionalGroup":
                    prompt = group_header
                    for group in ['benzene rings', 'hydroxyl', 'anhydride', 'aldehyde', 'ketone', 'carboxyl', 'ester', 'amide', 'amine', 'nitro', 'halo', 'nitrile', 'thiol', 'sulfide', 'disulfide', 'sulfoxide', 'sulfone', 'borane']:
                        if group in row and int(row[group]) > 0:
                            prompt += f"- {group}: {int(row[group])}\n"
                    prompt += smiles_suffix
                
                elif subtask == "BondNum":
                    prompt = bond_header
                    for bond in ['single', 'double', 'triple', 'rotatable', 'aromatic']:
                        if bond in row and int(row[bond]) > 0:
                            prompt += f"- {bond} bonds: {int(row[bond])}\n"
                    prompt += smiles_suffix
            
            elif task == "MolEdit":
                molecule = row["molecule"]
                
                if subtask == "AddComponent":
                    group = row["added_group"]
                    prompt = add_template.format(group=group, molecule=molecule)
                
                elif subtask == "DelComponent":
                    group = row["removed_group"]
                    prompt = del_template.format(group=group, molecule=molecule)
                
                elif subtask == "SubComponent":
                    added_group = row["added_group"]
                    removed_group = row["removed_group"]
                    prompt = sub_template.format(
                        removed_group=removed_group,
                        added_group=added_group,
                        molecule=molecule,
                    )
            
            elif task == "MolOpt":
                molecule = row["molecule"]
                instruction = row["Instruction"]
                prompt = molopt_template.format(instruction=instruction, molecule=molecule)
            
            prompts.append(prompt)
        
        print(f"Built {len(prompts)} prompts")
        if len(prompts) > 0:
            print(f"Example prompt: {prompts[0]}")
        
        # Batch generation.
        print(f"Generating {subtask} predictions in batch with vLLM...")
        try:
            outputs_batch = llm.generate(prompts, sampling_params)
            print(f"Generation finished, total {len(outputs_batch)} results")
            
            # Extract SMILES strings.
            for idx, output in enumerate(tqdm(outputs_batch, desc="Extract SMILES")):
                generated_text = output.outputs[0].text
                raw_outputs.append(generated_text)
                smiles = extract_smiles(generated_text)
                outputs.append(smiles)
            
            # Save predictions.
            output_file = f"{output_path}{subtask}.csv"
            pd.DataFrame({"outputs": outputs}).to_csv(output_file, index=False)
            print(f"Predictions saved to: {output_file}")
            
            # Save detailed results for traceability/debugging.
            detailed_file = f"{output_path}{subtask}_detailed.csv"
            pd.DataFrame(
                {
                    "prompt": prompts,
                    "raw_output": raw_outputs,
                    "smiles": outputs,
                }
            ).to_csv(detailed_file, index=False)
            print(f"Detailed results saved to: {detailed_file}")
            
            # Also save JSONL to avoid CSV issues with multiline outputs.
            jsonl_file = f"{output_path}{subtask}_raw.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for p, r, s in zip(prompts, raw_outputs, outputs):
                    f.write(json.dumps({"prompt": p, "raw_output": r, "smiles": s}, ensure_ascii=False) + "\n")
            print(f"Raw output JSONL saved to: {jsonl_file}")
            
            print(f"Generated {len(outputs)} SMILES")
            if len(outputs) > 0:
                print(f"Example SMILES: {outputs[0]}")
        except Exception as e:
            print(f"Error during generation: {e}")
            raise
    
    except Exception as e:
        print(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate predictions with CN/EN prompts")
    parser.add_argument("--model_path", type=str, default="output/OpenMolIns-3B-v1", 
                        help="Model path")
    parser.add_argument("--benchmark", type=str, default="open_generation", 
                        help="Benchmark type")
    parser.add_argument("--task", type=str, default="MolCustom", 
                        help="Task type (MolCustom, MolEdit, MolOpt)")
    parser.add_argument("--subtask", type=str, default="AtomNum", 
                        help="Subtask name")
    parser.add_argument("--output_dir", type=str, default="./predictions/", 
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device (cuda, cpu)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--lang", type=str, default="cn", choices=["cn", "en"], help="Prompt language")
    
    args = parser.parse_args()
    
    generate_predictions(
        args.model_path,
        args.benchmark,
        args.task,
        args.subtask,
        args.output_dir,
        args.device,
        args.gpu_memory_utilization,
        args.lang,
    )

if __name__ == "__main__":
    main()