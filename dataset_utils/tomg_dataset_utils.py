import json

import pandas as pd
from datasets import Dataset, DatasetDict

data_scale = 'light' # "small", "medium", "large", "xlarge", "light"
sub_task = "all" # "AtomNum", "BondNum", "FunctionalGroup", "AddComponent", "SubComponent", "DelComponent", "LogP", "MR", "QED"
# Construct dataset path based on data_scale
dataset_path = f"../data/OpenMolIns/{data_scale}/train.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# Filter by subtask if specified
if sub_task != "all":
    df = df[df["SubTask"] == sub_task]
    print(f"Filtered dataset to subtask: {sub_task}")
    print(f"Filtered dataset size: {len(df)} examples")

# Convert to the required JSON format
formatted_data = []
for _, row in df.iterrows():
    formatted_entry = {
        "instruction": row["Instruction"],  # Using the original "Instruction" column
        "output": row["molecule"],          # Using the original "molecule" column as output
        "system": "You are working as an assistant of a chemist user. Please follow the instruction of the chemist and generate a molecule that satisfies the requirements of the chemist user. You could think step by step, but your final response should be a SMILES string. For example, 'Molecule: [SMILES STRING]'."                        # Empty system prompt as it's optional
    }
    formatted_data.append(formatted_entry)