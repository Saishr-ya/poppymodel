"""
src/ingestion/chembl_client.py

ChEMBL API client — drug targets, chirality, ADMET properties.
Primary data source for Layer 1 (target overlap) and Layer 4 (ADMET).

Docs: https://www.ebi.ac.uk/chembl/api/data/docs
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLClient:
    """
    Wraps the ChEMBL REST API with caching and error handling.

    All methods return plain Python dicts/lists — no library-specific objects.
    """

    def __init__(self, base_url: str = CHEMBL_BASE):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    # ── Drug / Molecule ────────────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 90)   # 90-day cache — ChEMBL rarely changes
    def get_molecule(self, chembl_id: str) -> Optional[dict]:
        """
        Fetch molecule properties: MW, logP, HBD, HBA, chirality, bioavailability, etc.

        Returns:
            dict with keys: molecule_chembl_id, pref_name, chirality,
            molecule_properties (mw_freebase, alogp, hbd, hba, oral, bioavailability),
            molecule_type, first_approval, black_box_warning, withdrawn_flag
        """
        url = f"{self.base_url}/molecule/{chembl_id}.json"
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            logger.error(f"ChEMBL molecule fetch failed for {chembl_id}: {e}")
            return None

    def get_chirality(self, chembl_id: str) -> Optional[str]:
        """
        Returns chirality annotation:
            '0' = Racemic mixture
            '1' = Single stereoisomer
            '2' = Achiral molecule
        """
        mol = self.get_molecule(chembl_id)
        if mol:
            return str(mol.get("chirality", ""))
        return None

    def lipinski_check(self, chembl_id: str) -> dict:
        """
        Evaluate Lipinski Rule of 5. Returns violation count and details.

        Rule of 5:
            MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10
        """
        mol = self.get_molecule(chembl_id)
        if not mol:
            return {"violations": 0, "details": {}, "error": "molecule not found"}

        props = mol.get("molecule_properties") or {}
        mw    = float(props.get("mw_freebase") or 0)
        logp  = float(props.get("alogp") or 0)
        hbd   = int(props.get("hbd") or 0)
        hba   = int(props.get("hba") or 0)

        violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ])

        return {
            "violations": violations,
            "details": {
                "mw": mw, "mw_pass": mw <= 500,
                "logp": logp, "logp_pass": logp <= 5,
                "hbd": hbd, "hbd_pass": hbd <= 5,
                "hba": hba, "hba_pass": hba <= 10,
            }
        }

    # ── Drug Targets ──────────────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_drug_targets(self, chembl_id: str) -> list[dict]:
        """
        Fetch protein targets for a drug (mechanism of action targets).

        Returns list of dicts with: target_chembl_id, target_name,
        action_type, uniprot_accession
        """
        url = f"{self.base_url}/mechanism.json"
        params = {"molecule_chembl_id": chembl_id, "limit": 100}
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            mechanisms = data.get("mechanisms", [])
            # Resolve UniProt IDs for each target
            targets = []
            for m in mechanisms:
                target_id = m.get("target_chembl_id")
                uniprot = self._resolve_uniprot(target_id) if target_id else None
                targets.append({
                    "target_chembl_id": target_id,
                    "target_name": m.get("target_name"),
                    "action_type": m.get("action_type"),
                    "uniprot_accession": uniprot,
                })
            return targets
        except Exception as e:
            logger.error(f"ChEMBL targets fetch failed for {chembl_id}: {e}")
            return []

    @cached_api_call(ttl_seconds=86400 * 90)
    def _resolve_uniprot(self, target_chembl_id: str) -> Optional[str]:
        """Map a ChEMBL target ID to its primary UniProt accession."""
        url = f"{self.base_url}/target/{target_chembl_id}.json"
        try:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            components = data.get("target_components", [])
            for c in components:
                for xref in c.get("target_component_xrefs", []):
                    if xref.get("xref_src_db") == "UniProt":
                        return xref.get("xref_id")
            return None
        except Exception:
            return None

    def get_target_uniprot_ids(self, chembl_id: str) -> set[str]:
        """Return a set of UniProt IDs for a drug's targets (for Jaccard scoring)."""
        targets = self.get_drug_targets(chembl_id)
        return {
            t["uniprot_accession"]
            for t in targets
            if t.get("uniprot_accession")
        }

    # ── Off-patent / Oral filter ───────────────────────────────────────────────

    def get_oral_offpatent_drugs(
        self,
        approval_before: int = 2015,
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch oral small molecules with approval before cutoff year.
        Used to build the candidate drug universe.

        Args:
            approval_before: Include drugs approved before this year.
            limit: Max results per page.
        """
        url = f"{self.base_url}/molecule.json"
        params = {
            "molecule_type": "Small molecule",
            "molecule_properties__mw_freebase__lte": 500,
            "first_approval__lt": approval_before,
            "oral": True,
            "limit": limit,
            "format": "json",
        }
        results = []
        offset = 0
        while True:
            params["offset"] = offset
            try:
                r = self.session.get(url, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                page = data.get("molecules", [])
                results.extend(page)
                if len(page) < limit:
                    break
                offset += limit
            except Exception as e:
                logger.error(f"ChEMBL paginated fetch error at offset {offset}: {e}")
                break
        logger.info(f"ChEMBL oral off-patent drugs fetched: {len(results)}")
        return results
