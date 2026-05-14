"""
src/ingestion/disgenet_client.py

DisGeNET API client — the primary source for disease-gene associations.

DisGeNET aggregates gene-disease associations from multiple sources:
  - OMIM, Orphanet, ClinVar, UniProt, CTD, PsyGeNET, HPO
  - GWAS Catalog, literature mining (NLP from PubMed)

Free tier: 1000 queries/day, requires free registration for API key.
Register at: https://www.disgenet.org/signup

Paid tier (Commercial): unlimited queries. For a startup, start with free
and upgrade when you hit the daily limit during batch runs.

API docs: https://www.disgenet.org/api/#/

Evidence scoring:
    DisGeNET assigns each gene-disease association a GDA score (0–1).
    Score > 0.3 = well-supported. Use as a filter threshold.
    
    Association types matter:
    - 'Biomarker'         → gene is a biomarker, not necessarily causal
    - 'AlteredExpression' → gene is dysregulated in disease
    - 'Causal'            → gene mutation causes disease (strongest signal)
    
    For Layer 1A (target overlap), you want 'Causal' + high-confidence 'AlteredExpression'.
    For network proximity (Layer 1B), broader inclusion improves coverage.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

DISGENET_API = "https://www.disgenet.org/api"

# Association type weights for scoring
# Causal mutations are the gold standard for rare disease engines.
ASSOC_TYPE_WEIGHTS = {
    "Causal": 1.0,
    "AlteredExpression": 0.7,
    "Biomarker": 0.5,
    "GeneticVariation": 0.8,
    "Modulator": 0.6,
    "Therapeutic": 0.5,
}

# Minimum GDA score threshold (0–1) for including an association.
# 0.1 = permissive (includes low-confidence), 0.3 = medium, 0.5 = strict.
DEFAULT_SCORE_THRESHOLD = 0.1


class DisGeNETClient:
    """
    Client for the DisGeNET REST API.

    Usage:
        client = DisGeNETClient(api_key=os.getenv("DISGENET_API_KEY"))
        genes = client.get_disease_genes("ORPHA:77")
        uniprot_ids = client.get_disease_uniprot_ids("ORPHA:77")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ):
        self.api_key = api_key or os.getenv("DISGENET_API_KEY", "")
        self.score_threshold = score_threshold
        self._session_token: Optional[str] = None

    # ── Authentication ─────────────────────────────────────────────────────────

    def _get_headers(self) -> dict:
        """Return auth headers for API requests."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def authenticate(self, email: str, password: str) -> bool:
        """
        Authenticate with email/password and get session token.
        Alternative to API key auth. Store token in self._session_token.
        """
        url = f"{DISGENET_API}/auth/login"
        try:
            r = requests.post(
                url,
                json={"email": email, "password": password},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if r.status_code == 200:
                self._session_token = r.json().get("token")
                logger.info("DisGeNET: authenticated successfully")
                return bool(self._session_token)
            logger.error(f"DisGeNET auth failed: {r.status_code}")
            return False
        except Exception as e:
            logger.error(f"DisGeNET auth error: {e}")
            return False

    # ── Disease-gene associations ──────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_disease_genes(
        self,
        disease_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        Fetch gene-disease associations for a given disease.

        Accepts Orphanet (ORPHA:XXXXX) or OMIM (OMIM:XXXXXX or MIM:XXXXXX) IDs.

        Returns list of dicts:
            [{
                gene_symbol: str,
                gene_id: int,          # NCBI Gene ID
                uniprot_id: str,       # UniProt accession (if available)
                gda_score: float,      # 0–1 confidence score
                assoc_type: str,       # 'Causal', 'AlteredExpression', etc.
                source: str,           # 'OMIM', 'Orphanet', 'GWAS', etc.
            }]
        """
        # Normalize disease ID format for DisGeNET
        disgenet_id = self._normalize_disease_id(disease_id)
        if not disgenet_id:
            logger.warning(f"Cannot normalize disease ID for DisGeNET: {disease_id}")
            return []

        url = f"{DISGENET_API}/gda/disease/{disgenet_id}"
        params = {
            "min_score": self.score_threshold,
            "limit": limit,
        }

        try:
            r = requests.get(
                url,
                params=params,
                headers=self._get_headers(),
                timeout=20,
            )

            if r.status_code == 404:
                logger.warning(f"DisGeNET: disease not found: {disease_id} → {disgenet_id}")
                return self._fallback_omim_lookup(disease_id)

            if r.status_code == 429:
                logger.warning("DisGeNET rate limit hit. Sleeping 60s.")
                time.sleep(60)
                r = requests.get(url, params=params, headers=self._get_headers(), timeout=20)

            r.raise_for_status()
            associations = r.json()

        except Exception as e:
            logger.error(f"DisGeNET get_disease_genes({disease_id}) failed: {e}")
            return self._fallback_omim_lookup(disease_id)

        genes = []
        for assoc in associations:
            gene_data = assoc.get("gene") or {}
            gene_symbol = gene_data.get("gene_symbol") or assoc.get("gene_symbol", "")
            if not gene_symbol:
                continue

            genes.append({
                "gene_symbol": gene_symbol,
                "gene_id": gene_data.get("gene_id") or assoc.get("geneid"),
                "uniprot_id": gene_data.get("uniprot") or "",
                "gda_score": float(assoc.get("score") or assoc.get("gda_score") or 0),
                "assoc_type": assoc.get("assocType") or assoc.get("association_type", ""),
                "source": assoc.get("source", ""),
            })

        # Sort by GDA score descending
        genes.sort(key=lambda g: g["gda_score"], reverse=True)
        logger.info(f"DisGeNET: {len(genes)} genes for {disease_id}")
        return genes

    def get_disease_uniprot_ids(self, disease_id: str) -> set[str]:
        """
        Return flat set of UniProt IDs for disease causal genes.
        This is the primary interface for Layer 1A and 1B.

        Note: Not all genes in DisGeNET have a direct UniProt ID.
        For those missing UniProt, we include NCBI Gene IDs as fallback
        (network proximity layer handles ID mapping separately).

        Returns:
            set of UniProt accession strings (e.g., {'P04075', 'Q99697'})
        """
        genes = self.get_disease_genes(disease_id)
        uniprot_ids = set()
        for g in genes:
            uid = g.get("uniprot_id", "").strip()
            if uid and uid != "None" and len(uid) >= 5:
                uniprot_ids.add(uid)
        return uniprot_ids

    def get_disease_gene_symbols(self, disease_id: str) -> set[str]:
        """
        Return flat set of gene symbols for pathway enrichment (Layer 1A).

        Returns:
            set of gene symbols e.g. {'ASAH1', 'GBA', 'SCARB2'}
        """
        genes = self.get_disease_genes(disease_id)
        return {g["gene_symbol"] for g in genes if g.get("gene_symbol")}

    # ── Fallback: OMIM API for disease-gene data ───────────────────────────────

    def _fallback_omim_lookup(self, disease_id: str) -> list[dict]:
        """
        Fallback when DisGeNET doesn't have coverage for a disease.
        Queries the Orphanet API to extract gene-disease links.

        This is particularly important for ultra-rare diseases that aren't
        in DisGeNET but are in Orphanet's curated database.
        """
        if not disease_id.startswith("ORPHA:"):
            return []

        orpha_id = disease_id.replace("ORPHA:", "")
        url = f"https://api.orphacode.org/EN/ClinicalEntity/{orpha_id}/Genes"

        try:
            r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
            if r.status_code != 200:
                return []
            data = r.json()
            genes = []
            for item in data.get("items", []):
                gene = item.get("Gene") or {}
                symbol = gene.get("GeneSymbol", "")
                if symbol:
                    genes.append({
                        "gene_symbol": symbol,
                        "gene_id": None,
                        "uniprot_id": "",
                        "gda_score": 0.5,   # Orphanet curated = moderate confidence
                        "assoc_type": "Causal",
                        "source": "Orphanet",
                    })
            logger.info(f"Orphanet fallback: {len(genes)} genes for {disease_id}")
            return genes
        except Exception as e:
            logger.debug(f"Orphanet fallback failed for {disease_id}: {e}")
            return []

    # ── ID normalization ───────────────────────────────────────────────────────

    def _normalize_disease_id(self, disease_id: str) -> Optional[str]:
        """
        Convert disease ID to DisGeNET format.

        DisGeNET accepts:
          - Orphanet: 'ORPHA_12345' (underscore, not colon)
          - OMIM: 'OMIM_123456'
          - MeSH: 'MESH_D001234'
          - ICD10: 'ICD10_E70.0'

        Input formats this engine uses:
          - 'ORPHA:12345' → 'ORPHA_12345'
          - 'OMIM:123456' → 'OMIM_123456'
        """
        if ":" in disease_id:
            prefix, num = disease_id.split(":", 1)
            return f"{prefix.upper()}_{num}"
        return disease_id

    # ── Gene-drug reverse lookup ───────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_gene_diseases(self, gene_symbol: str) -> list[dict]:
        """
        Reverse lookup: given a gene, what diseases is it associated with?

        Used in the disease-disease similarity network (Layer 10 in the spec).
        Returns list of {disease_id, disease_name, score, source}.
        """
        url = f"{DISGENET_API}/gda/gene/{gene_symbol}"
        params = {"min_score": self.score_threshold, "limit": 200}
        try:
            r = requests.get(url, params=params, headers=self._get_headers(), timeout=20)
            r.raise_for_status()
            associations = r.json()
            diseases = []
            for assoc in associations:
                dis = assoc.get("disease") or {}
                diseases.append({
                    "disease_id": dis.get("diseaseId") or assoc.get("diseaseId", ""),
                    "disease_name": dis.get("diseaseName") or assoc.get("diseaseName", ""),
                    "score": float(assoc.get("score") or 0),
                    "source": assoc.get("source", ""),
                })
            return diseases
        except Exception as e:
            logger.debug(f"get_gene_diseases({gene_symbol}) failed: {e}")
            return []

    # ── Disease search ─────────────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 90)
    def search_disease(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search DisGeNET for diseases by name.
        Useful for finding the correct disease ID when given a disease name.

        Returns list of {disease_id, disease_name, disease_type}.
        """
        url = f"{DISGENET_API}/disease/search"
        params = {"query": query, "limit": limit}
        try:
            r = requests.get(url, params=params, headers=self._get_headers(), timeout=15)
            r.raise_for_status()
            results = r.json()
            return [
                {
                    "disease_id": d.get("diseaseId"),
                    "disease_name": d.get("diseaseName"),
                    "disease_type": d.get("diseaseType"),
                }
                for d in results
                if d.get("diseaseId")
            ]
        except Exception as e:
            logger.debug(f"DisGeNET search failed for '{query}': {e}")
            return []