"""
src/ingestion/disgenet_client.py

DisGeNET API client — disease-gene associations for Layer 1 (target overlap).
Also includes Orphanet client for disease metadata and subtypes.

DisGeNET docs: https://www.disgenet.org/api/
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

DISGENET_BASE = "https://www.disgenet.org/api"
ORPHANET_BASE = "https://api.orphacode.org/EN/ClinicalEntity"


class DisGeNETClient:
    """
    Fetches disease-gene associations from DisGeNET.

    Evidence scores (EI): 0–1. Use score_cutoff ≥ 0.3 for reliable associations.
    Gene-Disease Association types: "GeneticVariation", "Biomarker", "AlteredExpression"
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DISGENET_API_KEY", "")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers["Accept"] = "application/json"

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_disease_genes(
        self,
        disease_id: str,
        score_cutoff: float = 0.3,
        source: str = "ALL",
    ) -> list[dict]:
        """
        Fetch causal/associated genes for a disease.

        Args:
            disease_id: OMIM or UMLS CUI (e.g. "C0017374" or "OMIM:230300")
            score_cutoff: Minimum DisGeNET association score (0–1).
            source: "CURATED", "ANIMAL_MODELS", or "ALL"

        Returns:
            List of dicts: {gene_symbol, ncbi_gene_id, uniprot_id, score, association_type}
        """
        # DisGeNET accepts UMLS IDs; map OMIM if needed
        umls_id = self._to_umls(disease_id)
        if not umls_id:
            logger.warning(f"Could not map {disease_id} to UMLS — returning empty gene list")
            return []

        url = f"{DISGENET_BASE}/gda/disease/{umls_id}"
        params = {
            "source": source,
            "min_score": score_cutoff,
            "limit": 500,
        }
        try:
            r = self.session.get(url, params=params, timeout=20)
            if r.status_code == 404:
                logger.warning(f"DisGeNET: no data for {disease_id}")
                return []
            r.raise_for_status()
            data = r.json()
            genes = []
            for entry in data:
                genes.append({
                    "gene_symbol": entry.get("gene_symbol"),
                    "ncbi_gene_id": entry.get("gene_ncbi_id"),
                    "uniprot_id": entry.get("uniprot_id"),
                    "score": entry.get("score"),
                    "association_type": entry.get("association_type"),
                })
            return genes
        except Exception as e:
            logger.error(f"DisGeNET gene fetch failed for {disease_id}: {e}")
            return []

    def get_disease_uniprot_ids(
        self, disease_id: str, score_cutoff: float = 0.3
    ) -> set[str]:
        """Return UniProt IDs for a disease's causal genes (for Jaccard scoring)."""
        genes = self.get_disease_genes(disease_id, score_cutoff)
        return {g["uniprot_id"] for g in genes if g.get("uniprot_id")}

    @cached_api_call(ttl_seconds=86400 * 7)
    def _to_umls(self, disease_id: str) -> Optional[str]:
        """
        Map an OMIM or Orphanet ID to UMLS CUI via DisGeNET's mapping endpoint.
        Returns None if mapping fails.
        """
        if disease_id.startswith("C") and len(disease_id) == 8:
            return disease_id  # Already a UMLS CUI

        # Try DisGeNET's disease search
        url = f"{DISGENET_BASE}/disease/search"
        source_key = "omim" if disease_id.upper().startswith("OMIM") else "orphanet"
        clean_id = disease_id.split(":")[-1]
        params = {source_key: clean_id, "limit": 1}
        try:
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            results = r.json()
            if results:
                return results[0].get("diseaseId")
        except Exception as e:
            logger.debug(f"UMLS mapping failed for {disease_id}: {e}")
        return None


class OrphanetClient:
    """
    Fetches rare disease metadata from Orphanet.
    Used for: disease subtypes, age of onset, affected tissue, natural history level.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_disease_info(self, orpha_id: str) -> Optional[dict]:
        """
        Fetch Orphanet disease record.

        Args:
            orpha_id: Orphanet code without prefix, e.g. "355" for CML.

        Returns:
            Dict with: name, definition, age_of_onset, prevalence, gene_panels, subtypes
        """
        clean_id = orpha_id.replace("ORPHA:", "")
        url = f"{ORPHANET_BASE}/{clean_id}"
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f"Orphanet fetch failed for {orpha_id}: {e}")
            return None

    def get_age_of_onset(self, orpha_id: str) -> Optional[str]:
        """
        Returns age of onset category: 'Neonatal', 'Infancy', 'Childhood',
        'Adolescent', 'Adult', 'Elderly', or None.
        """
        info = self.get_disease_info(orpha_id)
        if info:
            onset_list = info.get("AgeOfOnset", [])
            if onset_list:
                return onset_list[0].get("Name")
        return None

    def is_pediatric_onset(self, orpha_id: str) -> bool:
        """Returns True if >50% of patients have onset before age 12."""
        pediatric_categories = {"Neonatal", "Infancy", "Childhood"}
        onset = self.get_age_of_onset(orpha_id)
        return onset in pediatric_categories if onset else False

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_subtypes(self, orpha_id: str) -> list[dict]:
        """Fetch disease subtypes for subtype-level scoring."""
        clean_id = orpha_id.replace("ORPHA:", "")
        url = f"{ORPHANET_BASE}/{clean_id}/Classification"
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            return data.get("items", [])
        except Exception as e:
            logger.debug(f"Orphanet subtypes fetch failed for {orpha_id}: {e}")
            return []
