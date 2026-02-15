import os

import fire
import pandas as pd
from dataset_utils.evaluation import (calculate_novelty, calculate_similarity,
                                      mol_prop)
from tqdm import tqdm


def evaluate(
    # Model name or path
    name="qwen2.5-3b-instruct",
    # Full model path (optional, overrides name if provided)
    model_path=None,
    # Dataset settings
    benchmark="open_generation",
    task="MolCustom",
    subtask="AtomNum",
    # Output directory
    output_dir="./new_predictions/",
    # Whether to calculate novelty
    calc_novelty=False
):
    """
    Evaluate molecule generation models on various tasks.
    
    Args:
        name: Model name (used if model_path is not provided)
        model_path: Full path to the model (overrides name if provided)
        benchmark: Benchmark type (open_generation, targeted_generation)
        task: Task type (MolCustom, MolEdit, MolOpt)
        subtask: Specific subtask
        output_dir: Directory containing model predictions
        calc_novelty: Whether to calculate novelty metrics
    """
    # Use model_path if provided, otherwise use name
    model_identifier = model_path if model_path else name
    
    # Extract model name from path for file naming if full path is provided
    if model_path:
        # Extract the last part of the path as the model name for file naming
        raw_model_name = os.path.basename(model_path.rstrip('/'))
        # Sanitize the model name for file paths - replace problematic characters
        # Replace double hyphens with single underscore to avoid path issues
        model_name = raw_model_name.replace('--', '_').replace('/', '_')
        # For clarity in logs, show what name we're using
        print(f"Using model path: {model_path}")
        print(f"Model name for file organization: {model_name}")
    else:
        model_name = name
    
    # Construct file paths for test data and model predictions
    raw_file = f"./data/benchmarks/{benchmark}/{task}/{subtask}/test.csv"
    target_file = f"{output_dir}{model_name}/{benchmark}/{task}/{subtask}.csv"
    
    print(f"Loading test data from: {raw_file}")
    print(f"Looking for predictions at: {target_file}")
    
    # Check if the test data file exists
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Test data file not found: {raw_file}")
    
    # Check if the predictions directory exists, if not create it
    predictions_dir = os.path.dirname(target_file)
    if not os.path.exists(predictions_dir):
        print(f"Creating predictions directory: {predictions_dir}")
        os.makedirs(predictions_dir, exist_ok=True)
    
    # Check if the predictions file exists
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Predictions file not found: {target_file}. Please ensure predictions are generated before evaluation.")
    
    # Load test data and model predictions
    data = pd.read_csv(raw_file)
    try:
        # Try standard CSV parsing first
        target = pd.read_csv(target_file)
    except:
        # Fall back to more flexible parsing if standard fails
        target = pd.read_csv(target_file, engine='python')

    # Evaluate based on benchmark and task type
    if benchmark == "open_generation":
        if task == "MolCustom":
            if subtask == "AtomNum":
                # Evaluate atom number accuracy
                atom_type = ['carbon', 'oxygen', 'nitrogen', 'sulfur', 'fluorine', 'chlorine', 'bromine', 'iodine', 'phosphorus', 'boron', 'silicon', 'selenium', 'tellurium', 'arsenic', 'antimony', 'bismuth', 'polonium']
                flags = []
                valid_molecules = []
                
                # Process each molecule with progress bar
                for idx in tqdm(range(len(data))):
                    if mol_prop(target["outputs"][idx], "validity"):
                        valid_molecules.append(target["outputs"][idx])
                        flag = 1
                        for atom in atom_type:
                            if mol_prop(target["outputs"][idx], "num_" + atom) != int(data[atom][idx]):
                                flag = 0
                                break
                        flags.append(flag)
                    else:
                        flags.append(0)
                
                # Print evaluation metrics
                success_rate = sum(flags) / len(flags)
                validity_rate = len(valid_molecules) / len(flags)
                if calc_novelty:
                    novelties = calculate_novelty(valid_molecules)
                    novelty_rate = sum(novelties) / len(novelties)
                
                print("Success Rate:", success_rate)
                print("Validity:", validity_rate)
                print("novelty: ", novelty_rate)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "validity": validity_rate,
                    "novelty": novelty_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(flags)
                }
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Summary metrics saved to: {summary_file}")
                    
            elif subtask == "FunctionalGroup":
                # Evaluate functional group accuracy
                functional_groups = ['benzene rings', 'hydroxyl', 'anhydride', 'aldehyde', 'ketone', 'carboxyl', 'ester', 'amide', 'amine', 'nitro', 'halo', 'nitrile', 'thiol', 'sulfide', 'disulfide', 'sulfoxide', 'sulfone', 'borane']
                flags = []
                valid_molecules = []
                for idx in tqdm(range(len(data))):
                    if mol_prop(target["outputs"][idx], "validity"):
                        valid_molecules.append(target["outputs"][idx])
                        flag = 1
                        for group in functional_groups:
                            if group == "benzene rings":
                                if mol_prop(target["outputs"][idx], "num_benzene_ring") != int(data[group][idx]):
                                    flag = 0
                                    break
                            else:
                                if mol_prop(target["outputs"][idx], "num_" + group) != int(data[group][idx]):
                                    flag = 0
                                    break
                        flags.append(flag)
                    else:
                        flags.append(0)
                    
                # Print evaluation metrics
                success_rate = sum(flags) / len(flags)
                validity_rate = len(valid_molecules) / len(flags)
                if calc_novelty:
                    novelties = calculate_novelty(valid_molecules)
                    novelty_rate = sum(novelties) / len(novelties)
                
                print("Success Rate:", success_rate)
                print("Validity:", validity_rate)
                print("Novelty: ", novelty_rate)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "validity": validity_rate,
                    "novelty": novelty_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(flags)
                }
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Summary metrics saved to: {summary_file}")

            elif subtask == "BondNum":
                # Evaluate bond number accuracy
                bonds_type = ['single', 'double', 'triple', 'rotatable', 'aromatic']
                flags = []
                valid_molecules = []
                for idx in tqdm(range(len(data))):
                    if mol_prop(target["outputs"][idx], "validity"):
                        valid_molecules.append(target["outputs"][idx])
                        flag = 1
                        for bond in bonds_type:
                            if bond == "rotatable":
                                if int(data[bond][idx]) == 0:
                                    continue
                                elif mol_prop(target["outputs"][idx], "rot_bonds") != int(data[bond][idx]):
                                    flag = 0
                                    break
                            else:
                                if int(data[bond][idx]) == 0:
                                    continue
                                elif mol_prop(target["outputs"][idx], "num_" + bond + "_bonds") != int(data[bond][idx]):
                                    flag = 0
                                    break
                        flags.append(flag)
                    else:
                        flags.append(0)

                # Print evaluation metrics
                success_rate = sum(flags) / len(flags)
                validity_rate = len(valid_molecules) / len(flags)
                if calc_novelty:
                    novelties = calculate_novelty(valid_molecules)
                    novelty_rate = sum(novelties) / len(novelties)
                
                print("Success Rate:", success_rate)
                print("Validity:", validity_rate)
                print("Novelty: ", novelty_rate)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "validity": validity_rate,
                    "novelty": novelty_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(flags)
                }
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Summary metrics saved to: {summary_file}")

        elif task == "MolEdit":
            if subtask == "AddComponent":
                # Evaluate adding component to molecules
                valid_molecules = []
                successed = []
                similarities = []
                for idx in tqdm(range(len(data))):
                    raw = data["molecule"][idx]
                    group = data["added_group"][idx]
                    if group == "benzene ring":
                        group = "benzene_ring"
                    target_mol = target["outputs"][idx]
                    if mol_prop(target_mol, "validity"):
                        valid_molecules.append(target_mol)

                        if mol_prop(target_mol, "num_" + group) == mol_prop(raw, "num_" + group) + 1:
                            successed.append(1)
                        else:
                            successed.append(0)

                        similarities.append(calculate_similarity(raw, target_mol))
                    else:
                        successed.append(0)

                # Print evaluation metrics
                success_rate = sum(successed) / len(successed)
                similarity_avg = sum(similarities) / len(similarities) if similarities else 0
                validity_rate = len(valid_molecules) / len(data)
                
                print("Success Rate:", success_rate)
                print("Similarity:", similarity_avg)
                print("Validity:", validity_rate)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "similarity": similarity_avg,
                    "validity": validity_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(successed)
                }
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Summary metrics saved to: {summary_file}")

                
            elif subtask == "DelComponent":
                # Evaluate deleting component from molecules
                valid_molecules = []
                successed = []
                similarities = []
                for idx in tqdm(range(len(data))):
                    raw = data["molecule"][idx]
                    group = data["removed_group"][idx]
                    if group == "benzene ring":
                        group = "benzene_ring"
                    target_mol = target["outputs"][idx]
                    if mol_prop(target_mol, "validity"):
                        valid_molecules.append(target_mol)

                        if mol_prop(target_mol, "num_" + group) == mol_prop(raw, "num_" + group) - 1:
                            successed.append(1)
                        else:
                            successed.append(0)

                        similarities.append(calculate_similarity(raw, target_mol))
                    else:
                        successed.append(0)

                # Print evaluation metrics
                success_rate = sum(successed) / len(successed)
                similarity_avg = sum(similarities) / len(similarities) if similarities else 0
                validity_rate = len(valid_molecules) / len(data)
                
                print("Success Rate:", success_rate)
                print("Similarity:", similarity_avg)
                print("Validity:", validity_rate)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "similarity": similarity_avg,
                    "validity": validity_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(successed)
                }
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Summary metrics saved to: {summary_file}")

                
            elif subtask == "SubComponent":
                # Evaluate substituting component in molecules
                valid_molecules = []
                successed = []
                similarities = []
                for idx in tqdm(range(len(data))):
                    raw = data["molecule"][idx]
                    added_group = data["added_group"][idx]
                    removed_group = data["removed_group"][idx]
                    if added_group == "benzene ring":
                        added_group = "benzene_ring"
                    if removed_group == "benzene ring":
                        removed_group = "benzene_ring"

                    target_mol = target["outputs"][idx]

                    if mol_prop(target_mol, "validity"):
                        valid_molecules.append(target_mol)

                        if mol_prop(target_mol, "num_" + removed_group) == mol_prop(raw, "num_" + removed_group) - 1 and mol_prop(target_mol, "num_" + added_group) == mol_prop(raw, "num_" + added_group) + 1:
                            successed.append(1)
                        else:
                            successed.append(0)

                        similarities.append(calculate_similarity(raw, target_mol))
                    else:
                        successed.append(0)

                # Print evaluation metrics
                success_rate = sum(successed) / len(successed)
                similarity_avg = sum(similarities) / len(similarities) if similarities else 0
                validity_rate = len(valid_molecules) / len(data)
                
                print("Success Rate:", success_rate)
                print("Similarity:", similarity_avg)
                print("Validity:", validity_rate)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "similarity": similarity_avg,
                    "validity": validity_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(successed)
                }
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Summary metrics saved to: {summary_file}")
                

        elif task == "MolOpt":
            if subtask == "LogP":
                # Evaluate LogP optimization
                valid_molecules = []
                successed = []
                similarities = []
                
                # Create a detailed results dictionary for each molecule
                detailed_results = []
                
                for idx in tqdm(range(len(data))):
                    raw = data["molecule"][idx]
                    target_mol = target["outputs"][idx]
                    instruction = data["Instruction"][idx]
                    
                    # Initialize result dictionary for this molecule
                    result = {
                        "index": idx,
                        "original_molecule": raw,
                        "generated_molecule": target_mol,
                        "instruction": instruction,
                        "validity": False,
                        "success": 0,
                        "similarity": 0.0,
                        "original_logP": 0.0,
                        "generated_logP": 0.0,
                        "logP_change": 0.0
                    }
                    
                    if mol_prop(target_mol, "validity"):
                        valid_molecules.append(target_mol)
                        result["validity"] = True
                        
                        # Calculate similarity
                        sim = calculate_similarity(raw, target_mol)
                        similarities.append(sim)
                        result["similarity"] = sim
                        
                        # Get LogP values
                        original_logP = mol_prop(raw, "logP")
                        generated_logP = mol_prop(target_mol, "logP")
                        result["original_logP"] = original_logP
                        result["generated_logP"] = generated_logP
                        result["logP_change"] = generated_logP - original_logP
                        
                        # Check if optimization was successful
                        if "lower" in instruction or "decrease" in instruction:
                            if generated_logP < original_logP:
                                successed.append(1)
                                result["success"] = 1
                            else:
                                successed.append(0)
                        else:
                            if generated_logP > original_logP:
                                successed.append(1)
                                result["success"] = 1
                            else:
                                successed.append(0)
                    else:
                        successed.append(0)
                    
                    # Add to detailed results
                    detailed_results.append(result)
                
                # Print evaluation metrics
                success_rate = sum(successed) / len(successed)
                similarity_avg = sum(similarities) / len(similarities) if similarities else 0
                validity_rate = len(valid_molecules) / len(data)
                
                print("Success Rate:", success_rate)
                print("Similarity:", similarity_avg)
                print("Validity:", validity_rate)
                
                # Save detailed results to CSV
                results_df = pd.DataFrame(detailed_results)
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                results_file = f"{output_dir}{subtask}_detailed_results.csv"
                results_df.to_csv(results_file, index=False)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "similarity": similarity_avg,
                    "validity": validity_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(successed)
                }
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Detailed results saved to: {results_file}")
                print(f"Summary metrics saved to: {summary_file}")

            elif subtask == "MR":
                # Evaluate MR optimization
                valid_molecules = []
                successed = []
                similarities = []
                
                # Create a detailed results dictionary for each molecule
                detailed_results = []
                
                for idx in tqdm(range(len(data))):
                    raw = data["molecule"][idx]
                    target_mol = target["outputs"][idx]
                    instruction = data["Instruction"][idx]
                    
                    # Initialize result dictionary for this molecule
                    result = {
                        "index": idx,
                        "original_molecule": raw,
                        "generated_molecule": target_mol,
                        "instruction": instruction,
                        "validity": False,
                        "success": 0,
                        "similarity": 0.0,
                        "original_MR": 0.0,
                        "generated_MR": 0.0,
                        "MR_change": 0.0
                    }
                    
                    if mol_prop(target_mol, "validity"):
                        valid_molecules.append(target_mol)
                        result["validity"] = True
                        
                        # Calculate similarity
                        sim = calculate_similarity(raw, target_mol)
                        similarities.append(sim)
                        result["similarity"] = sim
                        
                        # Get MR values
                        original_MR = mol_prop(raw, "MR")
                        generated_MR = mol_prop(target_mol, "MR")
                        result["original_MR"] = original_MR
                        result["generated_MR"] = generated_MR
                        result["MR_change"] = generated_MR - original_MR
                        
                        if "lower" in instruction or "decrease" in instruction:
                            if generated_MR < original_MR:
                                successed.append(1)
                                result["success"] = 1
                            else:
                                successed.append(0)
                        else:
                            if generated_MR > original_MR:
                                successed.append(1)
                                result["success"] = 1
                            else:
                                successed.append(0)
                    else:
                        successed.append(0)
                    
                    # Add to detailed results
                    detailed_results.append(result)
                
                # Print evaluation metrics
                success_rate = sum(successed) / len(successed)
                similarity_avg = sum(similarities) / len(similarities) if similarities else 0
                validity_rate = len(valid_molecules) / len(data)
                
                print("Success Rate:", success_rate)
                print("Similarity:", similarity_avg)
                print("Validity:", validity_rate)
                
                # Save detailed results to CSV
                results_df = pd.DataFrame(detailed_results)
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                results_file = f"{output_dir}{subtask}_detailed_results.csv"
                results_df.to_csv(results_file, index=False)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "similarity": similarity_avg,
                    "validity": validity_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(successed)
                }
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Detailed results saved to: {results_file}")
                print(f"Summary metrics saved to: {summary_file}")
                
            elif subtask == "QED":
                # Evaluate QED optimization
                valid_molecules = []
                successed = []
                similarities = []
                
                # Create a detailed results dictionary for each molecule
                detailed_results = []
                
                for idx in tqdm(range(len(data))):
                    raw = data["molecule"][idx]
                    target_mol = target["outputs"][idx]
                    instruction = data["Instruction"][idx]
                    
                    # Initialize result dictionary for this molecule
                    result = {
                        "index": idx,
                        "original_molecule": raw,
                        "generated_molecule": target_mol,
                        "instruction": instruction,
                        "validity": False,
                        "success": 0,
                        "similarity": 0.0,
                        "original_qed": 0.0,
                        "generated_qed": 0.0,
                        "qed_change": 0.0
                    }
                    
                    if mol_prop(target_mol, "validity"):
                        valid_molecules.append(target_mol)
                        result["validity"] = True
                        
                        # Calculate similarity
                        sim = calculate_similarity(raw, target_mol)
                        similarities.append(sim)
                        result["similarity"] = sim
                        
                        # Get QED values
                        original_qed = mol_prop(raw, "qed")
                        generated_qed = mol_prop(target_mol, "qed")
                        result["original_qed"] = original_qed
                        result["generated_qed"] = generated_qed
                        result["qed_change"] = generated_qed - original_qed
                        
                        if "lower" in instruction or "decrease" in instruction:
                            if generated_qed < original_qed:
                                successed.append(1)
                                result["success"] = 1
                            else:
                                successed.append(0)
                        else:
                            if generated_qed > original_qed:
                                successed.append(1)
                                result["success"] = 1
                            else:
                                successed.append(0)
                    else:
                        successed.append(0)
                    
                    # Add to detailed results
                    detailed_results.append(result)
                
                # Print evaluation metrics
                success_rate = sum(successed) / len(successed)
                similarity_avg = sum(similarities) / len(similarities) if similarities else 0
                validity_rate = len(valid_molecules) / len(data)
                
                print("Success Rate:", success_rate)
                print("Similarity:", similarity_avg)
                print("Validity:", validity_rate)
                
                # Save detailed results to CSV
                results_df = pd.DataFrame(detailed_results)
                output_dir = f"{output_dir}{model_name}/{benchmark}/{task}/"
                os.makedirs(output_dir, exist_ok=True)
                results_file = f"{output_dir}{subtask}_detailed_results.csv"
                results_df.to_csv(results_file, index=False)
                
                # Save summary metrics
                summary = {
                    "success_rate": success_rate,
                    "similarity": similarity_avg,
                    "validity": validity_rate,
                    "total_molecules": len(data),
                    "valid_molecules": len(valid_molecules),
                    "successful_optimizations": sum(successed)
                }
                summary_df = pd.DataFrame([summary])
                summary_file = f"{output_dir}{subtask}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                
                print(f"Detailed results saved to: {results_file}")
                print(f"Summary metrics saved to: {summary_file}")

    elif benchmark == "targeted_generation":
        # Placeholder for targeted generation evaluation
        pass
    else:
        raise ValueError("Invalid Benchmark Type")


if __name__ == "__main__":
    # Use Fire to automatically generate command-line interface
    fire.Fire(evaluate)
    
    # Example commands:
    
    # 1. To evaluate the Qwen2.5-3B-Instruct model on LogP optimization:
    # conda activate post-training
    # python evaluate.py name="Qwen2.5-3B-Instruct" benchmark="open_generation" task="MolOpt" subtask="LogP" output_dir="./predictions/"
    
    # 2. To evaluate with novelty calculation:
    # conda activate post-training
    # python evaluate.py name="Qwen2.5-3B-Instruct" benchmark="open_generation" task="MolCustom" subtask="AtomNum" calc_novelty=True output_dir="./predictions/"