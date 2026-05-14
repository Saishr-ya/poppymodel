"""
src/ingestion/chembl_client.py

ChEMBL API client — the primary data source for drug data.
ChEMBL is free, no API key required.

Provides:
  - Drug target UniProt IDs (for Layer 1A target overlap)
  - Drug molecule properties: MW, logP, chirality, oral flag (for Layer 4 ADMET)
  - Lipinski Rule of 5 check
  - hERG IC50 lookup (for Layer 4)
  - Racemic drug universe for chiral switch module
  - Mechanism of action data

API docs: https://www.ebi.ac.uk/chembl/api/data/docs
Rate limit: ~10 req/sec with polite delays; cache aggressively.

All methods cache for 30 days by default. The first call per drug downloads
and stores; subsequent calls within 30 days hit Redis.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
HEADERS = {"Accept": "application/json"}

# ChEMBL target ID for hERG potassium channel (used in ADMET layer)
HERG_TARGET_CHEMBL_ID = "CHEMBL240"

# Chirality annotation values in ChEMBL
# 0 = Racemate, 1 = Single stereoisomer, 2 = Achiral, -1 = Unknown
CHIRALITY_MAP = {
    "0": "Racemic mixture",
    "1": "Single stereoisomer",
    "2": "Achiral",
    "-1": "Unknown",
}


class ChEMBLClient:
    """
    Client for the ChEMBL REST API.

    Usage:
        client = ChEMBLClient()
        targets = client.get_target_uniprot_ids("CHEMBL192")
        mol = client.get_molecule("CHEMBL192")
    """

    def __init__(self, base_url: str = CHEMBL_API):
        self.base_url = base_url

    # ── Core molecule data ─────────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_molecule(self, chembl_id: str) -> Optional[dict]:
        """
        Fetch full molecule record from ChEMBL.
        Includes: properties (MW, logP, HBD, HBA), oral flag, chirality,
        max_phase (approval status), first_approval year.

        Returns:
            dict with keys: molecule_properties, oral, chirality_description,
                           max_phase, first_approval, black_box_warning
            None if not found.
        """
        url = f"{self.base_url}/molecule/{chembl_id}.json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                logger.warning(f"ChEMBL molecule not found: {chembl_id}")
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"ChEMBL get_molecule({chembl_id}) failed: {e}")
            return None

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_drug_targets(self, chembl_id: str) -> list[dict]:
        """
        Fetch all protein targets for a drug via its mechanism of action.

        Returns list of dicts:
            [{target_chembl_id, target_name, target_type, action_type, uniprot_id}]

        Note: ChEMBL mechanisms → target_chembl_id → target components → UniProt.
        This does two API calls per drug (mechanisms + target components).
        Results are cached so the second call is free.
        """
        # Step 1: Get mechanism of action data
        url = f"{self.base_url}/mechanism.json"
        params = {"molecule_chembl_id": chembl_id, "limit": 50}
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            r.raise_for_status()
            mechanisms = r.json().get("mechanisms", [])
        except Exception as e:
            logger.error(f"ChEMBL mechanisms({chembl_id}) failed: {e}")
            return []

        targets = []
        seen_target_ids = set()

        for mech in mechanisms:
            target_chembl_id = mech.get("target_chembl_id")
            if not target_chembl_id or target_chembl_id in seen_target_ids:
                continue
            seen_target_ids.add(target_chembl_id)

            # Step 2: Get target components (UniProt IDs)
            time.sleep(0.2)   # polite delay
            components = self._get_target_components(target_chembl_id)

            targets.append({
                "target_chembl_id": target_chembl_id,
                "target_name": mech.get("target_name", ""),
                "action_type": mech.get("action_type", ""),
                "uniprot_ids": components,
            })

        return targets

    @cached_api_call(ttl_seconds=86400 * 30)
    def _get_target_components(self, target_chembl_id: str) -> list[str]:
        """
        Fetch UniProt accession IDs for a ChEMBL target.
        Returns list of UniProt IDs (e.g., ['P00533', 'Q9UHD4']).
        """
        url = f"{self.base_url}/target/{target_chembl_id}.json"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            target = r.json()
            components = target.get("target_components", [])
            uniprot_ids = []
            for comp in components:
                for xref in comp.get("target_component_xrefs", []):
                    if xref.get("xref_src_db") == "UniProt":
                        uniprot_ids.append(xref["xref_id"])
            return uniprot_ids
        except Exception as e:
            logger.debug(f"_get_target_components({target_chembl_id}) failed: {e}")
            return []

    def get_target_uniprot_ids(self, chembl_id: str) -> set[str]:
        """
        Convenience method: return flat set of all UniProt IDs for a drug's targets.
        This is what Layer 1A and 1B consume directly.

        Returns:
            set of UniProt accession strings e.g. {'P00533', 'Q9UHD4'}
        """
        targets = self.get_drug_targets(chembl_id)
        uniprot_ids = set()
        for t in targets:
            uniprot_ids.update(t.get("uniprot_ids", []))
        return uniprot_ids

    # ── Physicochemical properties ─────────────────────────────────────────────

    def lipinski_check(self, chembl_id: str) -> dict:
        """
        Check Lipinski Rule of 5 for an oral drug.

        Rules:
            MW < 500
            logP < 5
            H-bond donors < 5
            H-bond acceptors < 10

        Returns:
            {violations: int, details: {mw, logp, hbd, hba, mw_ok, logp_ok, hbd_ok, hba_ok}}
        """
        mol = self.get_molecule(chembl_id)
        if not mol:
            return {"violations": 0, "details": {}}

        props = mol.get("molecule_properties") or {}

        mw   = float(props.get("mw_freebase")  or 0)
        logp = float(props.get("alogp")          or 0)
        hbd  = int(props.get("hbd")              or 0)
        hba  = int(props.get("hba")              or 0)

        mw_ok   = mw < 500
        logp_ok = logp < 5
        hbd_ok  = hbd < 5
        hba_ok  = hba < 10

        violations = sum([not mw_ok, not logp_ok, not hbd_ok, not hba_ok])

        return {
            "violations": violations,
            "details": {
                "mw": mw,   "mw_ok": mw_ok,
                "logp": logp, "logp_ok": logp_ok,
                "hbd": hbd,  "hbd_ok": hbd_ok,
                "hba": hba,  "hba_ok": hba_ok,
            },
        }

    def get_chirality(self, chembl_id: str) -> str:
        """
        Return chirality description string.
        Used by the chiral switch module to identify racemic candidates.

        Returns:
            'Racemic mixture' | 'Single stereoisomer' | 'Achiral' | 'Unknown'
        """
        mol = self.get_molecule(chembl_id)
        if not mol:
            return "Unknown"
        chirality_val = str(mol.get("chirality", "-1"))
        return CHIRALITY_MAP.get(chirality_val, "Unknown")

    # ── Bioactivity data ───────────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_herg_ic50(self, chembl_id: str) -> Optional[float]:
        """
        Query ChEMBL bioactivity for hERG (CHEMBL240) IC50 values.
        Returns minimum IC50 in µM (most conservative/worst-case).

        hERG IC50 < 1 µM = HIGH cardiotoxicity risk (hard disqualifier).
        hERG IC50 < 10 µM = MODERATE risk (score penalty).
        """
        url = f"{self.base_url}/activity.json"
        params = {
            "molecule_chembl_id": chembl_id,
            "target_chembl_id": HERG_TARGET_CHEMBL_ID,
            "standard_type": "IC50",
            "limit": 25,
        }
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            r.raise_for_status()
            activities = r.json().get("activities", [])
            ic50_values_um = []
            for a in activities:
                val = a.get("standard_value")
                units = (a.get("standard_units") or "").upper()
                if val is None:
                    continue
                try:
                    val_float = float(val)
                    if "NM" in units:
                        ic50_values_um.append(val_float / 1000)
                    elif "UM" in units or "μM" in units:
                        ic50_values_um.append(val_float)
                    # Skip other units (mM, etc.) — too rare to matter here
                except (ValueError, TypeError):
                    continue
            return min(ic50_values_um) if ic50_values_um else None
        except Exception as e:
            logger.debug(f"hERG IC50 lookup failed for {chembl_id}: {e}")
            return None

    # ── Drug universe queries ──────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_racemic_candidates(
        self,
        limit: int = 2000,
        max_phase_min: int = 3,
        patent_expiry_before: int = 2015,
    ) -> list[dict]:
        """
        Fetch racemic oral small molecules for the chiral switch module.

        Filters:
          - chirality = 0 (racemate)
          - max_phase >= 3 (approved or Phase III — means human safety data exists)
          - molecule_type = Small molecule
          - oral = True

        Returns list of {chembl_id, name, max_phase, first_approval}.
        """
        url = f"{self.base_url}/molecule.json"
        params = {
            "chirality": 0,
            "max_phase__gte": max_phase_min,
            "molecule_type": "Small molecule",
            "oral": True,
            "limit": min(limit, 1000),   # ChEMBL max per page is 1000
            "offset": 0,
        }

        all_mols = []
        while len(all_mols) < limit:
            try:
                r = requests.get(url, headers=HEADERS, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                mols = data.get("molecules", [])
                if not mols:
                    break
                for mol in mols:
                    all_mols.append({
                        "chembl_id": mol.get("molecule_chembl_id"),
                        "name": mol.get("pref_name") or "",
                        "max_phase": mol.get("max_phase"),
                        "first_approval": mol.get("first_approval"),
                    })
                page_meta = data.get("page_meta", {})
                if not page_meta.get("next"):
                    break
                params["offset"] += params["limit"]
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"get_racemic_candidates failed at offset {params['offset']}: {e}")
                break

        logger.info(f"ChEMBL racemic candidates: {len(all_mols)} molecules retrieved")
        return all_mols

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_approved_drugs_universe(
        self,
        oral_only: bool = True,
        small_molecule_only: bool = True,
        limit: int = 8000,
    ) -> list[dict]:
        """
        Fetch the full universe of approved off-patent drugs for batch scoring.

        This is the drug side of the ~120,000 pair computation described in the spec.
        Filters to oral small molecules for practical trial feasibility.

        Returns list of {chembl_id, name, max_phase, first_approval, chirality}.
        """
        url = f"{self.base_url}/molecule.json"
        params = {
            "max_phase": 4,   # approved
            "molecule_type": "Small molecule" if small_molecule_only else None,
            "oral": True if oral_only else None,
            "limit": 1000,
            "offset": 0,
        }
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        all_mols = []
        while len(all_mols) < limit:
            try:
                r = requests.get(url, headers=HEADERS, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                mols = data.get("molecules", [])
                if not mols:
                    break
                for mol in mols:
                    all_mols.append({
                        "chembl_id": mol.get("molecule_chembl_id"),
                        "name": mol.get("pref_name") or "",
                        "max_phase": mol.get("max_phase"),
                        "first_approval": mol.get("first_approval"),
                        "chirality": CHIRALITY_MAP.get(
                            str(mol.get("chirality", -1)), "Unknown"
                        ),
                    })
                if not data.get("page_meta", {}).get("next"):
                    break
                params["offset"] += params["limit"]
                time.sleep(0.3)
            except Exception as e:
                logger.error(f"get_approved_drugs_universe failed: {e}")
                break

        logger.info(f"ChEMBL drug universe: {len(all_mols)} approved drugs retrieved")
        return all_mols

    # ── Approval and safety data ───────────────────────────────────────────────

    def is_approved_in_reference_country(self, chembl_id: str) -> dict:
        """
        Check regulatory approval status for regulatory scoring (Layer 6).

        Returns:
            {us: bool, eu: bool, max_phase: int, first_approval: int, black_box: bool}
        """
        mol = self.get_molecule(chembl_id)
        if not mol:
            return {"us": False, "eu": False, "max_phase": 0}

        max_phase = mol.get("max_phase", 0)
        first_approval = mol.get("first_approval")
        black_box = mol.get("black_box_warning", False)

        return {
            "us": max_phase == 4,
            "eu": max_phase == 4,   # Refined with EMA data in production
            "max_phase": max_phase,
            "first_approval": first_approval,
            "black_box_warning": bool(black_box),
            "years_approved": (2025 - first_approval) if first_approval else None,
        }