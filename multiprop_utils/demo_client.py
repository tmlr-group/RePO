import requests
import json


# Define the SMILES string you want to get predictions for
smiles_to_predict = "O=C(Nc1ccc2c(c1)OCO2)[C@H](c1ccc(Cl)cc1)N(Cc1ccc(F)cc1)C1CCN(Cc2ccccc2)CC1" # Example: Diclofenac

# --- Combined Results ---
print("\n=== Combined Results ===\n")

def get_smiles_properties(smiles, require_drd2=True):
    # Define the base URLs of your running FastAPI servers
    ADMET_API_URL = "http://127.0.0.1:10086"  # ADMET Model API running on port 10086
    DRD2_API_URL = "http://127.0.0.1:10087"   # DRD2 Model API running on port 10087

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

print(get_smiles_properties(smiles_to_predict))