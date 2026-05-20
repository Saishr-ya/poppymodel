"""
src/ingestion/pharmgkb_client.py

PharmGKB client for CYP substrate/inhibitor/inducer data.

Fix #11: Rewrote parser to use drugLabels.byGene.tsv (the file PharmGKB
actually ships with useful data) instead of relationships.tsv whose PK
column is now always empty in the current PharmGKB export.

Strategy:
  1. Download drugLabels.zip (contains drugLabels.byGene.tsv)
  2. Parse gene→drug mappings from label name strings
  3. For each (drug, CYP) pair, call DailyMed Clinical Pharmacology section
     to classify as substrate / inhibitor / inducer from label text
  4. Fall back to CYP_REFERENCE for drugs not in DailyMed

The hardcoded CYP_REFERENCE table is kept as last-resort fallback
(covers ~30 most common drugs in rare-disease trials).

Setup:
  Auto-downloads on first use. To force a re-parse:
    rm data/processed/pharmgkb_cyp_profiles.json
    python -m src.ingestion.pharmgkb_client parse
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import time
import zipfile
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

PHARMGKB_DRUG_LABELS_URL = "https://api.pharmgkb.org/v1/download/file/data/drugLabels.zip"

PROCESSED_PGX_PATH   = "data/processed/pharmgkb_cyp_profiles.json"
RAW_PHARMGKB_DIR     = "data/raw/pharmgkb"
RAW_BY_GENE_PATH     = "data/raw/pharmgkb/drugLabels.byGene.tsv"

DAILYMED_SEARCH = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_SPL    = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}/sections.json"

_SUBSTRATE_RE = re.compile(r'\bsubstrate\b', re.I)
_INHIBITOR_RE = re.compile(r'\binhibit', re.I)
_INDUCER_RE   = re.compile(r'\binduc', re.I)

# Label name pattern: "Annotation of FDA Label for sildenafil and CYP3A4, CYP2C9"
_LABEL_NAME_RE = re.compile(
    r'Annotation of (?:FDA|EMA|HCSC|PMDA|Swissmedic) Label for (.+?) and (CYP[\w,\s]+)',
    re.I,
)

# ── Hardcoded CYP reference — fallback only ────────────────────────────────────
CYP_REFERENCE: dict[str, dict[str, list[str]]] = {
    "carbamazepine": {
        "substrates": ["CYP3A4", "CYP2C8"],
        "inhibitors": [],
        "inducers":   ["CYP3A4", "CYP1A2", "CYP2C9", "CYP2C19", "CYP2B6"],
    },
    "valproate": {
        "substrates": ["CYP2C9", "CYP2A6"],
        "inhibitors": ["CYP2C9", "CYP2C19"],
        "inducers":   [],
    },
    "phenytoin": {
        "substrates": ["CYP2C9", "CYP2C19"],
        "inhibitors": ["CYP2C19"],
        "inducers":   ["CYP3A4", "CYP2C9"],
    },
    "clobazam": {
        "substrates": ["CYP3A4", "CYP2C19"],
        "inhibitors": ["CYP2D6"],
        "inducers":   [],
    },
    "topiramate": {
        "substrates": ["CYP3A4"],
        "inhibitors": ["CYP2C19"],
        "inducers":   ["CYP3A4"],
    },
    "levetiracetam": {
        "substrates": [],
        "inhibitors": [],
        "inducers":   [],
    },
    "stiripentol": {
        "substrates": ["CYP1A2", "CYP2C19", "CYP3A4"],
        "inhibitors": ["CYP2C19", "CYP3A4", "CYP1A2"],
        "inducers":   [],
    },
    "sildenafil": {
        "substrates": ["CYP3A4", "CYP2C9"],
        "inhibitors": [],
        "inducers":   [],
    },
    "tadalafil": {
        "substrates": ["CYP3A4"],
        "inhibitors": [],
        "inducers":   [],
    },
    "bosentan": {
        "substrates": ["CYP3A4", "CYP2C9"],
        "inhibitors": [],
        "inducers":   ["CYP3A4", "CYP2C9"],
    },
    "ambrisentan": {
        "substrates": ["CYP3A4", "CYP2C19", "UGT1A9S"],
        "inhibitors": [],
        "inducers":   [],
    },
    "warfarin": {
        "substrates": ["CYP2C9", "CYP1A2", "CYP3A4"],
        "inhibitors": [],
        "inducers":   [],
    },
    "imatinib": {
        "substrates": ["CYP3A4", "CYP2C8"],
        "inhibitors": ["CYP3A4", "CYP2D6", "CYP2C9"],
        "inducers":   [],
    },
    "miglustat": {
        "substrates": [],
        "inhibitors": [],
        "inducers":   [],
    },
    "eliglustat": {
        "substrates": ["CYP2D6", "CYP3A4"],
        "inhibitors": [],
        "inducers":   [],
    },
    "tacrolimus": {
        "substrates": ["CYP3A4", "CYP3A5"],
        "inhibitors": [],
        "inducers":   [],
    },
    "cyclosporine": {
        "substrates": ["CYP3A4"],
        "inhibitors": ["CYP3A4", "OATP1B1", "OATP1B3"],
        "inducers":   [],
    },
    "omeprazole": {
        "substrates": ["CYP2C19", "CYP3A4"],
        "inhibitors": ["CYP2C19"],
        "inducers":   [],
    },
    "esomeprazole": {
        "substrates": ["CYP2C19", "CYP3A4"],
        "inhibitors": ["CYP2C19"],
        "inducers":   [],
    },
    "fluoxetine": {
        "substrates": ["CYP2D6", "CYP2C9"],
        "inhibitors": ["CYP2D6", "CYP2C19"],
        "inducers":   [],
    },
    "sertraline": {
        "substrates": ["CYP2C19", "CYP2D6", "CYP3A4"],
        "inhibitors": ["CYP2D6"],
        "inducers":   [],
    },
    "fluconazole": {
        "substrates": ["CYP2C9", "CYP3A4"],
        "inhibitors": ["CYP2C9", "CYP3A4", "CYP2C19"],
        "inducers":   [],
    },
    "rifampicin": {
        "substrates": [],
        "inhibitors": [],
        "inducers":   ["CYP3A4", "CYP2C9", "CYP2C19", "CYP1A2", "CYP2B6"],
    },
    "fenfluramine": {
        "substrates": ["CYP1A2", "CYP2B6"],
        "inhibitors": [],
        "inducers":   [],
    },
    "metformin": {
        "substrates": [],
        "inhibitors": [],
        "inducers":   [],
    },
}


# ── Module-level DailyMed helpers ─────────────────────────────────────────────

def _fetch_dailymed_cyp_section(drug_name: str) -> str:
    """
    Fetch the Clinical Pharmacology section from DailyMed for a drug.
    Returns concatenated section text, or empty string on failure.
    """
    try:
        r = requests.get(
            DAILYMED_SEARCH,
            params={"drug_name": drug_name, "pagesize": 1},
            timeout=15,
            headers={"User-Agent": "PoppyRepurposingEngine/1.0 (research)"},
        )
        if r.status_code != 200:
            return ""
        spls = r.json().get("data", [])
        if not spls:
            return ""
        setid = spls[0].get("setid", "")
        if not setid:
            return ""

        r2 = requests.get(
            DAILYMED_SPL.format(setid=setid),
            params={"tocsection": "clinical-pharmacology"},
            timeout=15,
            headers={"User-Agent": "PoppyRepurposingEngine/1.0 (research)"},
        )
        if r2.status_code != 200:
            return ""

        sections = r2.json().get("data", {}).get("sections", [])
        parts = []
        for sec in sections:
            content = sec.get("sectionText", "") or sec.get("text", "")
            if content:
                parts.append(re.sub(r'<[^>]+>', ' ', content))
        return " ".join(parts)

    except Exception as e:
        logger.debug(f"DailyMed fetch failed for {drug_name}: {e}")
        return ""


def _extract_gene_context(text: str, gene: str) -> str:
    """
    Extract sentences from label text that mention the given CYP gene.
    Returns up to 3 surrounding sentences concatenated.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant = [s for s in sentences if gene in s]
    return " ".join(relevant[:3])


# ── PharmGKBClient ────────────────────────────────────────────────────────────

class PharmGKBClient:
    """
    PharmGKB client for CYP substrate/inhibitor/inducer data.

    Priority order for data lookup:
        1. Parsed PharmGKB drugLabels file (data/processed/pharmgkb_cyp_profiles.json)
        2. Hardcoded CYP_REFERENCE table (covers most common drugs)
        3. PharmGKB REST API drug label endpoint (last resort, rarely useful)
    """

    def __init__(self):
        self._parsed_db: Optional[dict] = None
        self._ensure_data_downloaded()

    # ── Auto-download ──────────────────────────────────────────────────────────

    def _ensure_data_downloaded(self):
        """
        Auto-download PharmGKB drugLabels.zip if not present and parse it.
        Called once at __init__.
        """
        # Already processed — nothing to do
        if os.path.exists(PROCESSED_PGX_PATH):
            return

        # Raw file already present — just parse it
        if os.path.exists(RAW_BY_GENE_PATH):
            logger.info("PharmGKB drugLabels.byGene.tsv found — parsing...")
            self._parsed_db = self.parse_drug_labels()
            return

        # Neither exists — download drugLabels.zip
        logger.info("PharmGKB drugLabels not found — downloading...")
        try:
            r = requests.get(
                PHARMGKB_DRUG_LABELS_URL,
                timeout=120,
                stream=True,
                headers={"User-Agent": "PoppyRepurposingEngine/1.0 (research)"},
            )
            r.raise_for_status()
            content = b"".join(r.iter_content(chunk_size=65536))
            z = zipfile.ZipFile(io.BytesIO(content))
            os.makedirs(RAW_PHARMGKB_DIR, exist_ok=True)
            z.extractall(RAW_PHARMGKB_DIR)
            logger.info("PharmGKB drugLabels downloaded. Parsing...")
            self._parsed_db = self.parse_drug_labels()
        except Exception as e:
            logger.error(
                f"PharmGKB auto-download failed: {e}. "
                f"Download manually: curl -L '{PHARMGKB_DRUG_LABELS_URL}' "
                f"-o data/raw/pharmgkb/drugLabels.zip && "
                f"cd data/raw/pharmgkb && unzip drugLabels.zip"
            )

    # ── Data access ────────────────────────────────────────────────────────────

    def _load_parsed_db(self) -> dict:
        """Load parsed PharmGKB CYP profiles if file exists."""
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

        # 2. Hardcoded reference table
        if key in CYP_REFERENCE:
            return CYP_REFERENCE[key]

        # Try without salt/form suffixes (e.g. "valproate sodium" → "valproate")
        for ref_key in CYP_REFERENCE:
            if key.startswith(ref_key) or ref_key.startswith(key.split()[0]):
                return CYP_REFERENCE[ref_key]

        # 3. PharmGKB REST API (last resort)
        api_result = self._query_pharmgkb_api(drug_name)
        if api_result:
            return api_result

        logger.warning(f"No CYP profile found for '{drug_name}'.")
        return {"substrates": [], "inhibitors": [], "inducers": []}

    @cached_api_call(ttl_seconds=86400 * 90)
    def _query_pharmgkb_api(self, drug_name: str) -> Optional[dict]:
        """Query PharmGKB REST API (last resort — structured CYP data not reliably available)."""
        try:
            r = requests.get(
                "https://api.pharmgkb.org/v1/data/drug",
                params={"name": drug_name, "view": "base"},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if r.status_code != 200:
                return None
            # PharmGKB REST API doesn't expose structured CYP data; always falls through
            return None
        except Exception as e:
            logger.debug(f"PharmGKB API query failed for {drug_name}: {e}")
            return None

    # ── Parser ────────────────────────────────────────────────────────────────

    @classmethod
    def parse_drug_labels(
        cls,
        bylabel_path: str = RAW_BY_GENE_PATH,
        output_path:  str = PROCESSED_PGX_PATH,
    ) -> dict:
        """
        Parse drugLabels.byGene.tsv to build drug→CYP profiles.

        drugLabels.byGene.tsv schema:
            Gene ID | Gene Symbol | Label IDs | Label Names

        Label Names is a semicolon-separated list like:
            "Annotation of FDA Label for sildenafil and CYP3A4, CYP2C9"

        For each (drug, CYP gene) pair, calls DailyMed Clinical Pharmacology
        section to classify the relationship as substrate/inhibitor/inducer.
        Falls back to CYP_REFERENCE for drugs not found in DailyMed.
        """
        if not os.path.exists(bylabel_path):
            logger.error(
                f"drugLabels.byGene.tsv not found at {bylabel_path}. "
                f"Run: curl -L '{PHARMGKB_DRUG_LABELS_URL}' -o data/raw/pharmgkb/drugLabels.zip "
                f"&& cd data/raw/pharmgkb && unzip drugLabels.zip"
            )
            return {}

        # Step 1: build drug → set of CYP genes from label name strings
        drug_cyp_map: dict[str, set] = {}

        with open(bylabel_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                gene_symbol = row.get("Gene Symbol", "").strip().upper()
                if not gene_symbol.startswith("CYP"):
                    continue
                label_names = row.get("Label Names", "")
                for match in _LABEL_NAME_RE.finditer(label_names):
                    drug_raw = match.group(1).strip().lower()
                    # Parse all CYP genes from the match
                    cyp_genes = {
                        g.strip().upper()
                        for g in re.split(r'[,;]', match.group(2))
                        if g.strip().upper().startswith("CYP")
                    }
                    cyp_genes.add(gene_symbol)  # belt-and-suspenders
                    if drug_raw not in drug_cyp_map:
                        drug_cyp_map[drug_raw] = set()
                    drug_cyp_map[drug_raw].update(cyp_genes)

        logger.info(f"drugLabels.byGene: {len(drug_cyp_map)} drug-CYP pairs found")

        # Step 2: classify each (drug, CYP) pair via DailyMed
        cyp_profiles: dict[str, dict] = {}

        for drug_name, cyp_genes in drug_cyp_map.items():
            profile: dict[str, list] = {"substrates": [], "inhibitors": [], "inducers": []}

            label_text = _fetch_dailymed_cyp_section(drug_name)

            for gene in sorted(cyp_genes):
                if not label_text:
                    # No DailyMed text — fall back to hardcoded reference
                    ref = CYP_REFERENCE.get(drug_name, {})
                    if gene in ref.get("substrates", []):
                        profile["substrates"].append(gene)
                    if gene in ref.get("inhibitors", []):
                        profile["inhibitors"].append(gene)
                    if gene in ref.get("inducers", []):
                        profile["inducers"].append(gene)
                    continue

                context = _extract_gene_context(label_text, gene)
                if not context:
                    continue

                if _SUBSTRATE_RE.search(context) and gene not in profile["substrates"]:
                    profile["substrates"].append(gene)
                if _INHIBITOR_RE.search(context) and gene not in profile["inhibitors"]:
                    profile["inhibitors"].append(gene)
                if _INDUCER_RE.search(context) and gene not in profile["inducers"]:
                    profile["inducers"].append(gene)

            if any(profile.values()):
                cyp_profiles[drug_name] = profile

            time.sleep(0.15)  # polite DailyMed delay

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(cyp_profiles, f, indent=2)

        logger.info(
            f"PharmGKB parsed: {len(cyp_profiles)} drugs with CYP profiles → {output_path}"
        )
        return cyp_profiles

    # ── Legacy alias — kept so any external callers don't break ───────────────
    @classmethod
    def parse_relationships_file(cls, *args, **kwargs) -> dict:
        """Deprecated. Calls parse_drug_labels() instead."""
        logger.warning(
            "parse_relationships_file() is deprecated; "
            "relationships.tsv PK column is empty in current PharmGKB exports. "
            "Calling parse_drug_labels() instead."
        )
        return cls.parse_drug_labels()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "parse":
        result = PharmGKBClient.parse_drug_labels()
        print(f"Parsed {len(result)} drug profiles")

    elif command == "test":
        client = PharmGKBClient()
        for drug in ["sildenafil", "carbamazepine", "imatinib", "valproate", "eliglustat"]:
            profile = client.get_full_cyp_profile(drug)
            print(
                f"{drug}: substrates={profile['substrates']}, "
                f"inhibitors={profile['inhibitors']}, "
                f"inducers={profile['inducers']}"
            )

    else:
        print("Usage: python -m src.ingestion.pharmgkb_client [parse|test]")
        print()
        print("To force a re-parse:")
        print("  rm data/processed/pharmgkb_cyp_profiles.json")
        print("  python -m src.ingestion.pharmgkb_client parse")