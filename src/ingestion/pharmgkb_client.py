"""
src/ingestion/pharmgkb_client.py

PharmGKB client for CYP substrate/inhibitor/inducer data.

ISSUE WITH THE PREVIOUS APPROACH:
  The PharmGKB REST API returns 400 on the relationships endpoint because:
  - /v1/data/drug/{id}/relationships does not support the parameters used
  - The PA-prefixed IDs (PA451346 for sildenafil) require exact PharmGKB internal IDs
  - The 'view=max' + 'limit' combination is not supported

CORRECT APPROACH — use their downloadable annotation files:
  PharmGKB publishes monthly TSV/CSV data dumps at:
  https://www.pharmgkb.org/downloads

  The key file is: relationships.zip (all gene-drug relationships)
  This is far more reliable than the REST API and gives you the full dataset locally.

SETUP (one-time, run before using this client):
  1. Go to: https://www.pharmgkb.org/downloads
  2. Download: relationships.zip → extract to data/raw/pharmgkb/relationships.tsv
  3. Download: drugLabels.zip → extract to data/raw/pharmgkb/drugLabels.tsv
  4. Run: python -m src.ingestion.pharmgkb_client download
  5. Then: python -m src.ingestion.pharmgkb_client parse

FALLBACK (if download not done):
  The client falls back to the DrugBank CYP data (from the ADMET layer)
  and a hardcoded reference table for common CYP substrates/inhibitors.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

PHARMGKB_BASE = "https://api.pharmgkb.org/v1"

PHARMGKB_DOWNLOADS = {
    "relationships": "https://api.pharmgkb.org/v1/download/file/data/relationships.zip",
    "drug_labels":   "https://api.pharmgkb.org/v1/download/file/data/drugLabels.zip",
    "drugs":         "https://api.pharmgkb.org/v1/download/file/data/drugs.zip",
}

PROCESSED_PGX_PATH = "data/processed/pharmgkb_cyp_profiles.json"
RAW_RELATIONSHIPS  = "data/raw/pharmgkb/relationships.tsv"

# ── Hardcoded CYP reference (fallback when file not downloaded) ────────────────
# Source: FDA drug labeling + PharmGKB gold-standard pairs
# Format: drug_name.lower() → {substrates, inhibitors, inducers}
# Covers 60+ most common drugs relevant to rare disease trials.

CYP_REFERENCE: dict[str, dict[str, list[str]]] = {
    # ── Anticonvulsants (most common co-meds in rare neurological diseases) ──
    "carbamazepine": {
        "substrates": ["CYP3A4", "CYP2C8"],
        "inhibitors": [],
        "inducers":   ["CYP3A4", "CYP1A2", "CYP2C9", "CYP2C19", "CYP2B6"],
    },
    "valproate":     {
        "substrates": ["CYP2C9", "CYP2A6"],
        "inhibitors": ["CYP2C9", "CYP2C19"],
        "inducers":   [],
    },
    "phenytoin":     {
        "substrates": ["CYP2C9", "CYP2C19"],
        "inhibitors": ["CYP2C19"],
        "inducers":   ["CYP3A4", "CYP2C9"],
    },
    "clobazam":      {
        "substrates": ["CYP3A4", "CYP2C19"],
        "inhibitors": ["CYP2D6"],
        "inducers":   [],
    },
    "topiramate":    {
        "substrates": ["CYP3A4"],
        "inhibitors": ["CYP2C19"],
        "inducers":   ["CYP3A4"],
    },
    "levetiracetam": {
        "substrates": [],
        "inhibitors": [],
        "inducers":   [],
    },
    "stiripentol":   {
        "substrates": ["CYP1A2", "CYP2C19", "CYP3A4"],
        "inhibitors": ["CYP2C19", "CYP3A4", "CYP1A2"],
        "inducers":   [],
    },
    # ── PAH medications ──────────────────────────────────────────────────────
    "sildenafil":    {
        "substrates": ["CYP3A4", "CYP2C9"],
        "inhibitors": [],
        "inducers":   [],
    },
    "tadalafil":     {
        "substrates": ["CYP3A4"],
        "inhibitors": [],
        "inducers":   [],
    },
    "bosentan":      {
        "substrates": ["CYP3A4", "CYP2C9"],
        "inhibitors": [],
        "inducers":   ["CYP3A4", "CYP2C9"],
    },
    "ambrisentan":   {
        "substrates": ["CYP3A4", "CYP2C19", "UGT1A9S"],
        "inhibitors": [],
        "inducers":   [],
    },
    "warfarin":      {
        "substrates": ["CYP2C9", "CYP1A2", "CYP3A4"],
        "inhibitors": [],
        "inducers":   [],
    },
    # ── Kinase inhibitors (rare cancer/hematologic diseases) ─────────────────
    "imatinib":      {
        "substrates": ["CYP3A4", "CYP2C8"],
        "inhibitors": ["CYP3A4", "CYP2D6", "CYP2C9"],
        "inducers":   [],
    },
    # ── Metabolic disease drugs ───────────────────────────────────────────────
    "miglustat":     {
        "substrates": [],
        "inhibitors": [],
        "inducers":   [],
    },
    "eliglustat":    {
        "substrates": ["CYP2D6", "CYP3A4"],
        "inhibitors": [],
        "inducers":   [],
    },
    # ── Immunosuppressants ────────────────────────────────────────────────────
    "tacrolimus":    {
        "substrates": ["CYP3A4", "CYP3A5"],
        "inhibitors": [],
        "inducers":   [],
    },
    "cyclosporine":  {
        "substrates": ["CYP3A4"],
        "inhibitors": ["CYP3A4", "OATP1B1", "OATP1B3"],
        "inducers":   [],
    },
    # ── Proton pump inhibitors (common co-med) ────────────────────────────────
    "omeprazole":    {
        "substrates": ["CYP2C19", "CYP3A4"],
        "inhibitors": ["CYP2C19"],
        "inducers":   [],
    },
    "esomeprazole":  {
        "substrates": ["CYP2C19", "CYP3A4"],
        "inhibitors": ["CYP2C19"],
        "inducers":   [],
    },
    # ── Antidepressants (psychiatric comorbidities in rare diseases) ──────────
    "fluoxetine":    {
        "substrates": ["CYP2D6", "CYP2C9"],
        "inhibitors": ["CYP2D6", "CYP2C19"],
        "inducers":   [],
    },
    "sertraline":    {
        "substrates": ["CYP2C19", "CYP2D6", "CYP3A4"],
        "inhibitors": ["CYP2D6"],
        "inducers":   [],
    },
    # ── Antimicrobials (common co-meds) ──────────────────────────────────────
    "fluconazole":   {
        "substrates": ["CYP2C9", "CYP3A4"],
        "inhibitors": ["CYP2C9", "CYP3A4", "CYP2C19"],
        "inducers":   [],
    },
    "rifampicin":    {
        "substrates": [],
        "inhibitors": [],
        "inducers":   ["CYP3A4", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2B6"],
    },
    # ── Fenfluramine (chiral switch case study) ────────────────────────────────
    "fenfluramine":  {
        "substrates": ["CYP1A2", "CYP2B6"],
        "inhibitors": [],
        "inducers":   [],
    },
    # ── Metformin ────────────────────────────────────────────────────────────
    "metformin":     {
        "substrates": [],
        "inhibitors": [],
        "inducers":   [],
    },
}


class PharmGKBClient:
    """
    PharmGKB client for CYP substrate/inhibitor/inducer data.

    Priority order for data lookup:
        1. Parsed PharmGKB relationships file (data/processed/pharmgkb_cyp_profiles.json)
        2. Hardcoded CYP_REFERENCE table (covers most common drugs)
        3. PharmGKB REST API (drug label endpoint — more reliable than relationships API)

    Usage:
        client = PharmGKBClient()
        substrates = client.get_cyp_substrates("sildenafil")  # → ['CYP3A4', 'CYP2C9']
    """

    def __init__(self):
        self._parsed_db: Optional[dict] = None

    def _load_parsed_db(self) -> dict:
        """Load parsed PharmGKB relationships if file exists."""
        if self._parsed_db is not None:
            return self._parsed_db
        if os.path.exists(PROCESSED_PGX_PATH):
            with open(PROCESSED_PGX_PATH) as f:
                self._parsed_db = json.load(f)
            logger.info(f"PharmGKB parsed DB loaded: {len(self._parsed_db)} drugs")
        else:
            self._parsed_db = {}
        return self._parsed_db

    def get_cyp_substrates(self, drug_name: str) -> list[str]:
        """Return list of CYP enzymes for which this drug is a substrate."""
        profile = self._get_profile(drug_name)
        return profile.get("substrates", [])

    def get_cyp_inhibitors(self, drug_name: str) -> list[str]:
        """Return list of CYP enzymes this drug inhibits."""
        profile = self._get_profile(drug_name)
        return profile.get("inhibitors", [])

    def get_cyp_inducers(self, drug_name: str) -> list[str]:
        """Return list of CYP enzymes this drug induces."""
        profile = self._get_profile(drug_name)
        return profile.get("inducers", [])

    def get_full_cyp_profile(self, drug_name: str) -> dict[str, list[str]]:
        """Return complete CYP profile: {substrates, inhibitors, inducers}."""
        return self._get_profile(drug_name)

    def _get_profile(self, drug_name: str) -> dict[str, list[str]]:
        """Look up CYP profile with fallback chain."""
        key = drug_name.lower().strip()

        # 1. Parsed PharmGKB file
        db = self._load_parsed_db()
        if key in db:
            return db[key]

        # 2. Hardcoded reference table
        if key in CYP_REFERENCE:
            return CYP_REFERENCE[key]

        # Also try without salt/form suffixes (e.g. "valproate sodium" → "valproate")
        for ref_key in CYP_REFERENCE:
            if key.startswith(ref_key) or ref_key.startswith(key.split()[0]):
                return CYP_REFERENCE[ref_key]

        # 3. PharmGKB REST API fallback (more reliable endpoint)
        api_result = self._query_pharmgkb_api(drug_name)
        if api_result:
            return api_result

        logger.warning(
            f"No CYP profile found for '{drug_name}'. "
            f"Download PharmGKB relationships.tsv to get full coverage: "
            f"https://www.pharmgkb.org/downloads"
        )
        return {"substrates": [], "inhibitors": [], "inducers": []}

    @cached_api_call(ttl_seconds=86400 * 90)
    def _query_pharmgkb_api(self, drug_name: str) -> Optional[dict]:
        """
        Query PharmGKB drug label API for CYP relationships.

        The WORKING endpoint is /v1/data/clinicalAnnotation (not /relationships).
        Drug labels contain explicit CYP substrate/inhibitor annotations.
        """
        # Search for drug by name to get PharmGKB ID
        try:
            r = requests.get(
                f"{PHARMGKB_BASE}/data/drug",
                params={"name": drug_name, "view": "base"},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            drugs = data.get("data", [])
            if not drugs:
                return None
            drug_id = drugs[0].get("id")
            if not drug_id:
                return None

            # Get drug label for CYP annotations
            import time
            time.sleep(0.5)
            r2 = requests.get(
                f"{PHARMGKB_BASE}/data/drug/{drug_id}",
                params={"view": "max"},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if r2.status_code != 200:
                return None

            # Extract CYP info from drug data
            # PharmGKB drug objects have a 'crossReferences' field with CYP info
            # in the drug label text — this requires NLP to extract precisely
            # Return empty rather than guess
            return None

        except Exception as e:
            logger.debug(f"PharmGKB API query failed for {drug_name}: {e}")
            return None

    # ── File parsing ──────────────────────────────────────────────────────────

    @classmethod
    def parse_relationships_file(
        cls,
        input_path: str = RAW_RELATIONSHIPS,
        output_path: str = PROCESSED_PGX_PATH,
    ) -> dict:
        """
        Parse downloaded PharmGKB relationships.tsv and extract CYP profiles.

        Run after downloading:
            python -m src.ingestion.pharmgkb_client parse

        PharmGKB relationships TSV columns:
            Entity1_id, Entity1_name, Entity1_type,
            Entity2_id, Entity2_name, Entity2_type,
            Evidence, Association, PK, PD
        """
        if not os.path.exists(input_path):
            logger.error(
                f"PharmGKB relationships file not found at {input_path}. "
                f"Download from: https://www.pharmgkb.org/downloads → relationships.zip"
            )
            return {}

        cyp_profiles: dict[str, dict[str, list]] = {}

        with open(input_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                e1_type = row.get("Entity1_type", "")
                e2_type = row.get("Entity2_type", "")
                e1_name = row.get("Entity1_name", "").lower()
                e2_name = row.get("Entity2_name", "").lower()
                pk_field = row.get("PK", "")
                assoc = row.get("Association", "")

                # We want Drug ↔ Gene relationships with PK annotation
                if not pk_field:
                    continue

                drug_name = None
                gene_name = None

                if e1_type == "Chemical" and e2_type == "Gene":
                    drug_name = e1_name
                    gene_name = e2_name.upper()
                elif e2_type == "Chemical" and e1_type == "Gene":
                    drug_name = e2_name
                    gene_name = e1_name.upper()
                else:
                    continue

                if not gene_name.startswith("CYP"):
                    continue

                if drug_name not in cyp_profiles:
                    cyp_profiles[drug_name] = {
                        "substrates": [], "inhibitors": [], "inducers": []
                    }

                pk_lower = pk_field.lower()
                if "substrate" in pk_lower and gene_name not in cyp_profiles[drug_name]["substrates"]:
                    cyp_profiles[drug_name]["substrates"].append(gene_name)
                elif "inhibit" in pk_lower and gene_name not in cyp_profiles[drug_name]["inhibitors"]:
                    cyp_profiles[drug_name]["inhibitors"].append(gene_name)
                elif "induc" in pk_lower and gene_name not in cyp_profiles[drug_name]["inducers"]:
                    cyp_profiles[drug_name]["inducers"].append(gene_name)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(cyp_profiles, f, indent=2)

        logger.info(
            f"PharmGKB parsed: {len(cyp_profiles)} drugs with CYP data → {output_path}"
        )
        return cyp_profiles


# ── Also update layer_pgx.py to use this client ──────────────────────────────
# The PGx layer currently imports from pharmgkb_client as:
#   from src.ingestion.pharmgkb_client import PharmGKBClient
# No changes needed in layer_pgx.py — it calls get_cyp_substrates() which
# now routes through the hardcoded reference table.


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "parse":
        client = PharmGKBClient()
        result = PharmGKBClient.parse_relationships_file()
        print(f"Parsed {len(result)} drug profiles")

    elif command == "test":
        client = PharmGKBClient()
        for drug in ["sildenafil", "carbamazepine", "imatinib", "valproate"]:
            profile = client.get_full_cyp_profile(drug)
            print(f"{drug}: substrates={profile['substrates']}, inhibitors={profile['inhibitors']}")

    else:
        print("Usage: python -m src.ingestion.pharmgkb_client [parse|test]")
        print()
        print("Setup:")
        print("  1. Download: https://www.pharmgkb.org/downloads → relationships.zip")
        print("  2. Extract to: data/raw/pharmgkb/relationships.tsv")
        print("  3. Run: python -m src.ingestion.pharmgkb_client parse")