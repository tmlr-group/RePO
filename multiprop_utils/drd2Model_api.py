import pickle
import sys
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from rdkit import Chem, rdBase

from proper_utils import fingerprints_from_mol

rdBase.DisableLog('rdApp.error')

# Initialize the model once when the application starts
drd2_model_name = "./multiprop_utils/clf_py36.pkl"
# Backward-compat for old sklearn pickles that reference `sklearn.svm.classes`
# (newer sklearn moved it to `sklearn.svm._classes`).
import sklearn.svm._classes as _sk_svm_classes  # type: ignore

sys.modules.setdefault("sklearn.svm.classes", _sk_svm_classes)
drd2_model = pickle.load(open(drd2_model_name, "rb"))

if hasattr(drd2_model, "__dict__"):
    _d = drd2_model.__dict__
    if ("_n_support" not in _d) and ("n_support_" in _d):
        _d["_n_support"] = _d["n_support_"]
        # Avoid confusion with the property name in newer sklearn
        del _d["n_support_"]
    if ("_probA" not in _d) and ("probA_" in _d):
        _d["_probA"] = _d["probA_"]
        del _d["probA_"]
    if ("_probB" not in _d) and ("probB_" in _d):
        _d["_probB"] = _d["probB_"]
        del _d["probB_"]
    # Best-effort: newer sklearn may have `n_features_in_`; infer from support vectors if missing.
    if ("n_features_in_" not in _d) and ("support_vectors_" in _d):
        _d["n_features_in_"] = int(_d["support_vectors_"].shape[1])

app = FastAPI(title="DRD2 Prediction API", description="API for DRD2 predictions")

# Define the request body model for POST requests
class SmilesInput(BaseModel):
    smiles: str

def get_drd2_score(s):
    # Placeholder implementation that doesn't require the pickle file
    # In a real implementation, this would use the model
    if s is None:
        return 0.0
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return 0.0
    fp = fingerprints_from_mol(mol)
    score = drd2_model.predict_proba(fp)[:, 1]
    return float(score)

@app.post("/predict/")
async def predict_post(smiles_input: SmilesInput):
    """
    Accepts a SMILES string via POST request body and returns DRD2 predictions.
    """
    drd2 = get_drd2_score(smiles_input.smiles)
    return {"drd2": drd2}

@app.get("/predict/")
async def predict_get(smiles: str):
    """
    Accepts a SMILES string via GET query parameter and returns DRD2 predictions.
    """
    drd2 = get_drd2_score(smiles)
    return {"drd2": drd2}

if __name__ == "__main__":
    # Run on port 8001 instead of 10086 to avoid conflict with admetModel_api.py
    uvicorn.run(app, host="0.0.0.0", port=10087)