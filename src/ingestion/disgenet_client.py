"""
src/ingestion/disgenet_client.py

DisGeNET / OpenTargets gene-disease association client.

STATUS (May 2025):
  DisGeNET migrated from disgenet.org to api.disgenet.com in 2023 and
  then changed their endpoint structure again. The free tier appears to
  have been discontinued. All endpoints return 404/403.

  This client now uses OpenTargets as the PRIMARY source (free, stable,
  better rare disease coverage via Orphanet integration).
  DisGeNET is kept as a secondary source if you obtain a paid API key
  and their endpoint documentation for the current version.

How to re-enable DisGeNET if they fix their API:
  1. Set DISGENET_ENABLED = True below
  2. Verify the correct base URL from: https://www.disgenet.com/api/
  3. Update _CURRENT_BASE_URL with the confirmed working URL
  4. Test with: python debug_apis.py

For now, all calls transparently route through OpenTargets.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.ingestion.cache import cached_api_call
from src.ingestion.opentargets_client import OpenTargetsClient

logger = logging.getLogger(__name__)

# Set to True and update URL if DisGeNET restores working access
DISGENET_ENABLED = False
_CURRENT_BASE_URL = "https://api.disgenet.com/api/v1"  # update when verified

DEFAULT_SCORE_THRESHOLD = 0.1


class DisGeNETClient:
    """
    Gene-disease association client.

    Transparently uses OpenTargets (primary) with DisGeNET
    as a secondary source when available.

    Drop-in compatible with all existing layer code — no changes needed
    in layer1_target_overlap.py, layer1b_network_proximity.py, etc.

    Usage (unchanged from before):
        client = DisGeNETClient(api_key=os.getenv("DISGENET_API_KEY"))
        genes = client.get_disease_genes("ORPHA:422")
        uniprot_ids = client.get_disease_uniprot_ids("ORPHA:422")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ):
        self.api_key = api_key or os.getenv("DISGENET_API_KEY", "")
        self.score_threshold = score_threshold
        self._ot = OpenTargetsClient(score_threshold=score_threshold)

        if self.api_key and DISGENET_ENABLED:
            logger.info("DisGeNET client: using DisGeNET API + OpenTargets fallback")
        else:
            logger.info(
                "DisGeNET client: routing through OpenTargets "
                "(DisGeNET API endpoints are currently broken; see client docstring)"
            )

    def get_disease_genes(
        self,
        disease_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch gene-disease associations for a given disease.

        Returns list of dicts with keys:
            gene_symbol, gene_id, uniprot_id, gda_score, assoc_type, source

        Now routes through OpenTargets. The returned dict schema is
        backward-compatible with all existing layer code.
        """
        # ── Primary: OpenTargets ───────────────────────────────────────────
        ot_genes = self._ot.get_disease_genes(disease_id, limit=limit)

        if ot_genes:
            # Convert to DisGeNET-compatible schema
            return [
                {
                    "gene_symbol": g["gene_symbol"],
                    "gene_id":     None,
                    "uniprot_id":  g["uniprot_ids"][0] if g.get("uniprot_ids") else "",
                    "uniprot_ids": g.get("uniprot_ids", []),   # all UniProt IDs
                    "gda_score":   g["ot_score"],
                    "assoc_type":  g.get("assoc_type", "AlteredExpression"),
                    "source":      "OpenTargets",
                    "evidence_types": g.get("evidence_types", []),
                }
                for g in ot_genes
            ]

        # ── Secondary: DisGeNET (if API is restored and key is set) ──────
        if DISGENET_ENABLED and self.api_key:
            disgenet_genes = self._query_disgenet(disease_id, limit)
            if disgenet_genes:
                return disgenet_genes

        logger.warning(
            f"No gene-disease data found for {disease_id} from any source. "
            f"Check the ORPHA_TO_EFO mapping in opentargets_client.py "
            f"and verify the disease ID is correct."
        )
        return []

    def get_disease_uniprot_ids(self, disease_id: str) -> set[str]:
        """
        Return flat set of UniProt IDs for disease causal genes.
        Primary interface for Layer 1A and 1B.
        """
        # Use OpenTargets directly for better UniProt ID coverage
        ot_ids = self._ot.get_disease_uniprot_ids(disease_id)
        if ot_ids:
            return ot_ids

        # Fallback: extract from get_disease_genes
        genes = self.get_disease_genes(disease_id)
        uniprot_ids = set()
        for g in genes:
            # Handle both single uniprot_id and list uniprot_ids
            if g.get("uniprot_ids"):
                uniprot_ids.update(g["uniprot_ids"])
            elif g.get("uniprot_id"):
                uid = g["uniprot_id"].strip()
                if uid and len(uid) >= 5:
                    uniprot_ids.add(uid)
        return uniprot_ids

    def get_disease_gene_symbols(self, disease_id: str) -> set[str]:
        """Return flat set of gene symbols."""
        return self._ot.get_disease_gene_symbols(disease_id)

    def get_gene_diseases(self, gene_symbol: str) -> list[dict]:
        """Reverse lookup: what diseases is this gene associated with?"""
        return self._ot.get_gene_diseases(gene_symbol)

    @cached_api_call(ttl_seconds=86400 * 90)
    def search_disease(self, query: str, limit: int = 10) -> list[dict]:
        """Search for diseases by name."""
        return []   # OpenTargets search handled via get_efo_id in OT client

    # ── Private DisGeNET query (disabled until API is fixed) ──────────────────

    def _query_disgenet(self, disease_id: str, limit: int) -> list[dict]:
        """
        Query DisGeNET API directly.
        CURRENTLY DISABLED — API endpoints are broken (all return 404).

        To re-enable:
          1. Verify the correct endpoint at https://www.disgenet.com/api/
          2. Set DISGENET_ENABLED = True
          3. Update _CURRENT_BASE_URL above
        """
        if not DISGENET_ENABLED:
            return []

        import requests
        import time

        disgenet_id = disease_id.replace(":", "_")
        url = f"{_CURRENT_BASE_URL}/gda/disease/{disgenet_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        params = {"min_score": self.score_threshold, "limit": limit}

        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 429:
                time.sleep(60)
                r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            associations = r.json()
            genes = []
            for assoc in associations:
                gene_data = assoc.get("gene") or {}
                symbol = gene_data.get("gene_symbol") or assoc.get("gene_symbol", "")
                if not symbol:
                    continue
                genes.append({
                    "gene_symbol": symbol,
                    "gene_id":     gene_data.get("gene_id"),
                    "uniprot_id":  gene_data.get("uniprot") or "",
                    "uniprot_ids": [gene_data.get("uniprot")] if gene_data.get("uniprot") else [],
                    "gda_score":   float(assoc.get("score") or 0),
                    "assoc_type":  assoc.get("assocType") or "AlteredExpression",
                    "source":      "DisGeNET",
                })
            return genes
        except Exception as e:
            logger.debug(f"DisGeNET direct query failed for {disease_id}: {e}")
            return []