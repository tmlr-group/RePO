from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import requests
import json 
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from fire import Fire
import os
import re
from tqdm import tqdm
import pandas as pd
# taken from https://github.com/ziqi92/Modof/blob/main/model/properties.py#L34

def extract_smile(text):
        """Extract SMILE string from text using regex pattern."""
        if text is None:
            return ""
            
        # First try the provided pattern
        extract_pattern: str = r"<smile>(.*?)</smile>"
        match = re.search(extract_pattern, text, re.DOTALL)
        if match:
            smile = match.group(1).strip()
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    return smile
            except:
                pass
        
        # Try backup patterns if the main pattern fails
        backup_patterns = [
            r"<smile>(.*?)</smile>",
            r"SMILES?:\s*([A-Za-z0-9@\[\]\(\)\\.=#\-\+]+)"
        ]
        
        for pattern in backup_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                smile = match.group(1).strip()
                try:
                    mol = Chem.MolFromSmiles(smile)
                    if mol is not None:
                        return smile
                except:
                    continue
        
        # Try heuristic approach
        common_words = ["think", "answer", "molecule", "modify", "optimize", "increase", "decrease", 
                        "value", "please", "smiles", "structure", "similar", "drug-like", "maintaining"]
        
        smiles_pattern = r'([A-Za-z0-9@\[\]\(\)\\.=#\-\+]{5,})'
        matches = re.findall(smiles_pattern, text)
        
        for potential_smile in matches:
            if potential_smile.lower() in common_words:
                continue
                
            if not re.search(r'[CNOPS]|Cl|Br|Fe|Si|Al|Ca|Mg|Na|Li|He|Ne|Ar|Kr|Xe|Rn', potential_smile):
                continue
                
            try:
                mol = Chem.MolFromSmiles(potential_smile)
                if mol is not None:
                    return potential_smile
            except:
                continue
        
        return ""

def pair_similarity(amol, bmol):
    if amol is None or bmol is None: 
        return 0.0

    if isinstance(amol, str):
        amol = Chem.MolFromSmiles(amol)
    if isinstance(bmol, str):
        bmol = Chem.MolFromSmiles(bmol)
    if amol is None or bmol is None:
        return 0.0
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

    fp1 = mfpgen.GetFingerprint(amol)
    fp2 = mfpgen.GetFingerprint(bmol)
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return sim

def get_smiles_properties(smiles, require_drd2=True):
    # Define the base URLs of your running FastAPI servers
    ADMET_API_URL = "http://127.0.0.1:10086"  # ADMET Model API running on port 8000
    DRD2_API_URL = "http://127.0.0.1:10087"   # DRD2 Model API running on port 8001

    try:
        # Get ADMET results (just use POST for combined results)
        admet_response = requests.post(f"{ADMET_API_URL}/predict/", json={"smiles": smiles})
        admet_response.raise_for_status()
        combined_results = admet_response.json()
        
        if require_drd2:
            # Get DRD2 results (just use POST for combined results)
            drd2_response = requests.post(f"{DRD2_API_URL}/predict/", json={"smiles": smiles})
            drd2_response.raise_for_status()
            combined_results.update(drd2_response.json())
        
        return combined_results
    except requests.exceptions.RequestException as e:
        print(f"Error getting combined results: {e}")
        return {"mutagenicity": 0,"bbbp": 0,"qed": 0,"plogp": 0,"drd2": 0}

def get_success_rate_similarity(predictions, require_drd2=False, output_folder="./MuMo_performance_distill", 
                               property_setting="", seen_setting="", IND_setting="", method_name=""):
    success_counter = 0
    total_counter = len(predictions)
    similarity_list = []
    detailed_results = []
    
    for prediction in tqdm(predictions):
        original_smiles = prediction["meta-data"]["source_smiles"]
        target_smiles = prediction["output"]
        model_response = prediction["vllm_output"]
        response_smile = extract_smile(model_response).replace(".", "")

        mol = Chem.MolFromSmiles(response_smile)
        if mol is None:
            print("==> Invalid SMILES: ", response_smile)
            continue
        
        print("==> Processing: ", original_smiles, "->", response_smile)

        similarity = pair_similarity(original_smiles, response_smile)
        similarity_list.append(similarity)

        original_property = {k: v['source'] for k, v in prediction["meta-data"]["properties"].items()}
        response_property = get_smiles_properties(response_smile, require_drd2=require_drd2)

        # Calculate property differences
        property_differences = {}
        for prop in original_property.keys():
            if prop in response_property:
                property_differences[prop] = response_property[prop] - original_property[prop]

        # Store detailed results
        detailed_result = {
            'original_smiles': original_smiles,
            'response_smiles': response_smile,
            'similarity': similarity
        }
        
        # Add original properties
        for prop, val in original_property.items():
            detailed_result[f'original_{prop}'] = val
            
        # Add response properties
        for prop, val in response_property.items():
            detailed_result[f'response_{prop}'] = val
            
        # Add property differences
        for prop, diff in property_differences.items():
            detailed_result[f'diff_{prop}'] = diff
            
        # Calculate average difference for this sample
        if property_differences:
            detailed_result['avg_diff'] = np.mean(list(property_differences.values()))
        else:
            detailed_result['avg_diff'] = 0.0
            
        detailed_results.append(detailed_result)

        wrong_direction = False
        for prop, input_val in original_property.items():
            if prop == "mutagenicity":
                continue
            output_val = response_property[prop]
            # if prop == "mutagenicity":
            #     if output_val >= input_val:  # Mutagenicity must decrease
            #         wrong_direction = True
            #         break
            # else:
            if output_val <= input_val:  # Other properties must increase
                wrong_direction = True
                break

        if not wrong_direction:
            success_counter += 1
    
    # Save detailed results to CSV
    if detailed_results:
        df = pd.DataFrame(detailed_results)
        csv_path = os.path.join(output_folder, f"detailed_results_{IND_setting}_{seen_setting}_{property_setting}{method_name}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")
        
        # Calculate and save average differences
        diff_columns = [col for col in df.columns if col.startswith('diff_')]
        if diff_columns:
            avg_diffs = {}
            for col in diff_columns:
                avg_diffs[col] = df[col].mean()
            
            # Add overall statistics
            avg_diffs['avg_similarity'] = df['similarity'].mean()
            avg_diffs['success_rate'] = success_counter / total_counter
            
            # Save average differences to a separate CSV
            avg_df = pd.DataFrame([avg_diffs])
            avg_csv_path = os.path.join(output_folder, f"avg_differences_{IND_setting}_{seen_setting}_{property_setting}{method_name}.csv")
            avg_df.to_csv(avg_csv_path, index=False)
            print(f"Average differences saved to: {avg_csv_path}")
            
            # Print summary
            print("\n=== Average Property Differences ===")
            for prop, avg_diff in avg_diffs.items():
                if prop.startswith('diff_'):
                    print(f"{prop}: {avg_diff:.4f}")
            print(f"Average similarity: {avg_diffs['avg_similarity']:.4f}")
            print(f"Success rate: {avg_diffs['success_rate']:.4f}")
    
    return success_counter / total_counter, np.mean(similarity_list)


def main(
    property_setting = "bbbp+plogp+qed",
    seen_setting = "seen",
    IND_setting = "IND",
    output_folder = "./MuMo_performance_distill",
    method_name = "",
    base_dir = ".",
    experiment_prefix = "bpq"
):
    require_drd2 = True if "drd2" in property_setting else False

    experiment_dir = os.path.join(base_dir, f"{experiment_prefix}{method_name}")
    json_path = os.path.join(
        experiment_dir,
        f"processed_{IND_setting}_{seen_setting}_{property_setting}_test_data.json",
    )

    with open(json_path, "r") as f:
        predictions = json.load(f)

    success_rate, similarity = get_success_rate_similarity(
        predictions, 
        require_drd2=require_drd2, 
        output_folder=output_folder,
        property_setting=property_setting,
        seen_setting=seen_setting,
        IND_setting=IND_setting,
        method_name=method_name
    )
    path = os.path.join(
        experiment_dir,
        f"{IND_setting}_sft_{seen_setting}_{property_setting}.json",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "success_rate": success_rate,
            "similarity": similarity
        }, f)

if __name__ == "__main__":
    Fire(main)