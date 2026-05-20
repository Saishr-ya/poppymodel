# scripts/build_target_pairs.py
# Disease list loaded from config/geo_datasets.json — no hardcoded IDs.
# To add a new disease, add it to config/geo_datasets.json and rerun this script.
import json
from src.ingestion.chembl_client import ChEMBLClient

with open("config/geo_datasets.json") as f:
    cfg = json.load(f)

diseases = [
    {"disease_id": d["disease_id"], "disease_name": d["disease_name"]}
    for d in cfg["datasets"]
]

client = ChEMBLClient()
drugs  = client.get_approved_drugs_universe(oral_only=True, limit=200)

pairs = [
    {"drug_id": d["chembl_id"], "drug_name": d["name"],
     "disease_id": dis["disease_id"], "disease_name": dis["disease_name"]}
    for d in drugs for dis in diseases
    if d["chembl_id"] and d["name"]
]

with open("data/target_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)
print(f"{len(pairs)} pairs written")
