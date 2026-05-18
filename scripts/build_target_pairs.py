# scripts/build_target_pairs.py
from src.ingestion.chembl_client import ChEMBLClient

client = ChEMBLClient()
drugs  = client.get_approved_drugs_universe(oral_only=True, limit=200)

diseases = [
    {"disease_id": "ORPHA:422",   "disease_name": "Pulmonary arterial hypertension"},
    {"disease_id": "ORPHA:77",    "disease_name": "Gaucher disease type 1"},
    {"disease_id": "ORPHA:33069", "disease_name": "Dravet syndrome"},
    {"disease_id": "ORPHA:566",   "disease_name": "Pompe disease"},
]

pairs = [
    {"drug_id": d["chembl_id"], "drug_name": d["name"],
     "disease_id": dis["disease_id"], "disease_name": dis["disease_name"]}
    for d in drugs for dis in diseases
    if d["chembl_id"] and d["name"]
]

import json
with open("data/target_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)
print(f"{len(pairs)} pairs written")