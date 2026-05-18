"""
src/ingestion/pharmgkb_client.py

PharmGKB client for CYP substrate/inhibitor/inducer data.

Fix #6: Added _ensure_data_downloaded() called on __init__ so the
relationships file is fetched automatically on first use. Previously
the engine silently fell through to the hardcoded CYP_REFERENCE table
and never downloaded the full dataset.

The hardcoded CYP_REFERENCE table is kept as the last-resort fallback
(covers the ~30 most common drugs in rare-disease trials) so the engine
still works offline or if the download fails, but the full PharmGKB file
is now the primary source when available.

Setup:
  The client auto-downloads on first use. If the network is not available,
  download manually:
    https://www.pharmgkb.org/downloads → relationships.zip
    Extract to: data/raw/pharmgkb/relationships.tsv
  Then run:
    python -m src.ingestion.pharmgkb_client parse
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import zipfile
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

PHARMGKB_BASE = "https://api.pharmgkb.org/v1"
PHARMGKB_RELATIONSHIPS_URL = "https://api.pharmgkb.org/v1/download/file/data/relationships.zip"

PROCESSED_PGX_PATH = "data/processed/pharmgkb_cyp_profiles.json"
RAW_RELATIONSHIPS  = "data/raw/pharmgkb/relationships.tsv"

# ── Hardcoded CYP reference — fallback only ────────────────────────────────────
CYP_REFERENCE: dict[str, dict[str, list[str]]] = {
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
    "imatinib":      {
        "substrates": ["CYP3A4", "CYP2C8"],
        "inhibitors": ["CYP3A4", "CYP2D6", "CYP2C9"],
        "inducers":   [],
    },
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
    "fenfluramine":  {
        "substrates": ["CYP1A2", "CYP2B6"],
        "inhibitors": [],
        "inducers":   [],
    },
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
        3. PharmGKB REST API drug label endpoint

    Fix #6: _ensure_data_downloaded() is called on __init__ so the full
    PharmGKB dataset is fetched automatically on first use.
    """

    def __init__(self):
        self._parsed_db: Optional[dict] = None
        self._ensure_data_downloaded()

    # ── Fix #6: auto-download ──────────────────────────────────────────────────

    def _ensure_data_downloaded(self):
        """
        Auto-download PharmGKB relationships file if not present.

        Fix #6: Previously the engine silently fell through to the small
        hardcoded reference table on every run because no download was
        ever triggered. This method is called once at __init__ and ensures
        the full dataset is present before any scoring starts.
        """
        # Already processed — nothing to do
        if os.path.exists(PROCESSED_PGX_PATH):
            return

        # Raw file already present — just parse it
        if os.path.exists(RAW_RELATIONSHIPS):
            logger.info("PharmGKB raw relationships file found — parsing...")
            self._parsed_db = self.parse_relationships_file()
            return

        # Neither exists — download now
        logger.info(
            "PharmGKB relationships file not found — downloading from PharmGKB..."
        )
        try:
            r = requests.get(
                PHARMGKB_RELATIONSHIPS_URL,
                timeout=120,
                stream=True,
                headers={"User-Agent": "PoppyRepurposingEngine/1.0 (research)"},
            )
            r.raise_for_status()

            content = b"".join(r.iter_content(chunk_size=65536))
            z = zipfile.ZipFile(io.BytesIO(content))
            os.makedirs("data/raw/pharmgkb", exist_ok=True)
            z.extractall("data/raw/pharmgkb")
            logger.info("PharmGKB downloaded successfully. Parsing...")
            self._parsed_db = self.parse_relationships_file()
        except Exception as e:
            logger.error(
                f"PharmGKB auto-download failed: {e}. "
                f"Download manually from https://www.pharmgkb.org/downloads "
                f"and extract to data/raw/pharmgkb/relationships.tsv"
            )

    # ── Data access ────────────────────────────────────────────────────────────

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
        return self._get_profile(drug_name).get("substrates", [])

    def get_cyp_inhibitors(self, drug_name: str) -> list[str]:
        """Return list of CYP enzymes this drug inhibits."""
        return self._get_profile(drug_name).get("inhibitors", [])

    def get_cyp_inducers(self, drug_name: str) -> list[str]:
        """Return list of CYP enzymes this drug induces."""
        return self._get_profile(drug_name).get("inducers", [])

    def get_full_cyp_profile(self, drug_name: str) -> dict[str, list[str]]:
        """Return complete CYP profile: {substrates, inhibitors, inducers}."""
        return self._get_profile(drug_name)

    def _get_profile(self, drug_name: str) -> dict[str, list[str]]:
        """Look up CYP profile with fallback chain."""
        key = drug_name.lower().strip()

        # 1. Parsed PharmGKB file (full dataset)
        db = self._load_parsed_db()
        if key in db:
            return db[key]

        # 2. Hardcoded reference table (fallback for common drugs)
        if key in CYP_REFERENCE:
            return CYP_REFERENCE[key]

        # Also try without salt/form suffixes (e.g. "valproate sodium" → "valproate")
        for ref_key in CYP_REFERENCE:
            if key.startswith(ref_key) or ref_key.startswith(key.split()[0]):
                return CYP_REFERENCE[ref_key]

        # 3. PharmGKB REST API fallback
        api_result = self._query_pharmgkb_api(drug_name)
        if api_result:
            return api_result

        logger.warning(
            f"No CYP profile found for '{drug_name}'. "
            f"Download PharmGKB relationships.tsv for full coverage: "
            f"https://www.pharmgkb.org/downloads"
        )
        return {"substrates": [], "inhibitors": [], "inducers": []}

    @cached_api_call(ttl_seconds=86400 * 90)
    def _query_pharmgkb_api(self, drug_name: str) -> Optional[dict]:
        """Query PharmGKB drug label API for CYP relationships (last resort)."""
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
            # The REST API doesn't expose structured CYP data reliably;
            # return None here and let the caller fall through to the reference table.
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


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "parse":
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
        print("Auto-download runs on first import. To force a re-download:")
        print("  rm data/raw/pharmgkb/relationships.tsv data/processed/pharmgkb_cyp_profiles.json")
        print("  python -m src.ingestion.pharmgkb_client parse")