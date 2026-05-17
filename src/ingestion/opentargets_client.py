"""
src/ingestion/opentargets_client.py

OpenTargets Platform API client — replaces DisGeNET as the primary
gene-disease association source.

Why OpenTargets over DisGeNET:
  - Free, no API key required
  - Stable GraphQL API (disgenet.com has repeatedly changed endpoints)
  - Native Orphanet integration — every Orphanet disease has an EFO ID
  - Returns association scores 0–1 with evidence type breakdown
  - Returns UniProt IDs directly via proteinAnnotations field
  - Better rare disease coverage than DisGeNET for Orphanet diseases

API: https://api.platform.opentargets.org/api/v4/graphql
GraphQL browser: https://api.platform.opentargets.org/api/v4/graphql/browser
Docs: https://platform-docs.opentargets.org/data-access/graphql-api

ID mapping:
  Your engine uses ORPHA:422 format.
  OpenTargets uses EFO IDs like EFO_0000222.
  This client handles the mapping automatically via a search step.
  Results are cached 90 days — the mapping is stable.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
REST_URL    = "https://api.platform.opentargets.org/api/v4"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    # OpenTargets asks for a contact header in high-volume usage
    "User-Agent": "PoppyRepurposingEngine/1.0 (research; contact your@email.com)",
}

# Minimum association score (0–1) to include a gene.
# 0.1 = permissive, 0.3 = medium confidence, 0.5 = strict.
DEFAULT_SCORE_THRESHOLD = 0.1

# Curated ORPHA → EFO ID map for your target diseases.
# Populate this as you add diseases — avoids a search API call per run.
# Find EFO IDs at: https://www.ebi.ac.uk/ols4/ontologies/efo
# or via the OpenTargets search below.
ORPHA_TO_EFO: dict[str, str] = {
    "ORPHA:422":   "EFO_0000222",   # Pulmonary arterial hypertension
    "ORPHA:77":    "EFO_0000249",   # Gaucher disease type 1
    "ORPHA:33069": "EFO_0005271",   # Dravet syndrome
    "ORPHA:566":   "EFO_0009620",   # Pompe disease
    "ORPHA:355":   "EFO_0000339",   # CML (imatinib ground truth)
    "ORPHA:101435":"EFO_0000354",   # Microcephaly (negative control)
    "ORPHA:586":   "EFO_0004259",   # Polycystic ovary syndrome
    # Add more as you expand — run get_efo_id() once per new disease
}

# Curated OMIM → EFO fallback
OMIM_TO_EFO: dict[str, str] = {}


# ── GraphQL queries ────────────────────────────────────────────────────────────

_DISEASE_TARGETS_QUERY = """
query DiseaseTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    dbXRefs
    associatedTargets(page: {index: 0, size: $size}) {
      rows {
        target {
          id
          approvedSymbol
          proteinAnnotations {
            id
          }
        }
        score
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""

_DISEASE_SEARCH_QUERY = """
query DiseaseSearch($query: String!) {
  diseases(q: $query, page: {index: 0, size: 5}) {
    rows {
      id
      name
      dbXRefs
    }
  }
}
"""


class OpenTargetsClient:
    """
    Client for the OpenTargets Platform GraphQL API.

    Usage:
        client = OpenTargetsClient()
        uniprot_ids = client.get_disease_uniprot_ids("ORPHA:422")
        genes = client.get_disease_genes("ORPHA:422")
    """

    def __init__(self, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
        self.score_threshold = score_threshold

    # ── ID mapping ─────────────────────────────────────────────────────────────

    def get_efo_id(self, disease_id: str) -> Optional[str]:
        """
        Map ORPHA or OMIM ID to OpenTargets EFO ID.

        Checks the curated ORPHA_TO_EFO dict first (fast, no API call).
        Falls back to the OpenTargets search API.

        To add a new disease:
            1. Call client.get_efo_id("ORPHA:XXXXX")
            2. Copy the returned EFO ID into ORPHA_TO_EFO above
        """
        # Curated map first
        if disease_id in ORPHA_TO_EFO:
            return ORPHA_TO_EFO[disease_id]
        if disease_id in OMIM_TO_EFO:
            return OMIM_TO_EFO[disease_id]

        # Search API fallback
        return self._search_efo_id(disease_id)

    @cached_api_call(ttl_seconds=86400 * 90)
    def _search_efo_id(self, disease_id: str) -> Optional[str]:
        """
        Search OpenTargets for the EFO ID corresponding to an ORPHA/OMIM ID.
        Cached for 90 days.

        Tips:
          - Search uses the disease name from Orphanet as the query
          - If this returns None, look up the EFO ID manually and add to ORPHA_TO_EFO
        """
        # Build a search query from the ID
        if disease_id.startswith("ORPHA:"):
            orpha_num = disease_id.replace("ORPHA:", "")
            # OpenTargets dbXRefs use "Orphanet_XXXXX" format
            search_term = f"Orphanet_{orpha_num}"
        elif disease_id.startswith("OMIM:"):
            omim_num = disease_id.replace("OMIM:", "")
            search_term = f"OMIM_{omim_num}"
        else:
            search_term = disease_id

        try:
            r = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={"query": _DISEASE_SEARCH_QUERY, "variables": {"query": search_term}},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            rows = data.get("data", {}).get("diseases", {}).get("rows", [])

            for row in rows:
                xrefs = row.get("dbXRefs", [])
                # Check for exact Orphanet/OMIM match in xrefs
                if any(search_term in str(xref) for xref in xrefs):
                    efo_id = row.get("id")
                    logger.info(f"Mapped {disease_id} → {efo_id} ({row.get('name')})")
                    return efo_id

            # Take first result if no exact xref match
            if rows:
                efo_id = rows[0].get("id")
                logger.warning(
                    f"No exact xref match for {disease_id}. "
                    f"Using first result: {efo_id} ({rows[0].get('name')}). "
                    f"Verify this is correct and add to ORPHA_TO_EFO."
                )
                return efo_id

            logger.warning(
                f"OpenTargets: no disease found for {disease_id}. "
                f"Try searching manually at https://platform.opentargets.org/disease"
                f" and add the EFO ID to ORPHA_TO_EFO in opentargets_client.py"
            )
            return None

        except Exception as e:
            logger.error(f"OpenTargets EFO ID search failed for {disease_id}: {e}")
            return None

    # ── Gene-disease associations ──────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_disease_genes(
        self,
        disease_id: str,
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch gene-disease associations from OpenTargets.

        Args:
            disease_id: "ORPHA:422" or "OMIM:123456" format
            limit:      Max associations to return

        Returns:
            List of dicts:
            [{
                gene_symbol: str,
                uniprot_ids: list[str],  # may be empty for some targets
                ensembl_id: str,
                ot_score: float,         # 0–1 overall association score
                evidence_types: list[str], # e.g. ['genetic_association', 'somatic_mutation']
                source: str,
            }]
        """
        efo_id = self.get_efo_id(disease_id)
        if not efo_id:
            logger.error(
                f"Cannot query OpenTargets: no EFO ID found for {disease_id}. "
                f"Add the mapping to ORPHA_TO_EFO in opentargets_client.py"
            )
            return []

        try:
            r = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={
                    "query": _DISEASE_TARGETS_QUERY,
                    "variables": {"efoId": efo_id, "size": limit},
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()

            if "errors" in data:
                logger.error(f"OpenTargets GraphQL errors for {disease_id}: {data['errors']}")
                return []

            disease_data = data.get("data", {}).get("disease")
            if not disease_data:
                logger.warning(f"OpenTargets: no data for EFO ID {efo_id} ({disease_id})")
                return []

            rows = disease_data.get("associatedTargets", {}).get("rows", [])
            genes = []

            for row in rows:
                score = float(row.get("score") or 0)
                if score < self.score_threshold:
                    continue

                target = row.get("target", {})
                ensembl_id = target.get("id", "")
                symbol = target.get("approvedSymbol", "")

                # Extract UniProt IDs from proteinAnnotations
                protein_ann = target.get("proteinAnnotations") or {}
                uniprot_ids = protein_ann.get("id") or []
                if isinstance(uniprot_ids, str):
                    uniprot_ids = [uniprot_ids]

                # Extract evidence type scores
                evidence_types = [
                    ds["id"]
                    for ds in row.get("datatypeScores", [])
                    if ds.get("score", 0) > 0
                ]

                genes.append({
                    "gene_symbol":    symbol,
                    "ensembl_id":     ensembl_id,
                    "uniprot_ids":    uniprot_ids,
                    "ot_score":       score,
                    "evidence_types": evidence_types,
                    "source":         "OpenTargets",
                    # DisGeNET-compatible aliases for existing layer code:
                    "gda_score":      score,
                    "assoc_type":     "Causal" if "genetic_association" in evidence_types else "AlteredExpression",
                })

            # Sort by score descending
            genes.sort(key=lambda g: g["ot_score"], reverse=True)
            logger.info(f"OpenTargets: {len(genes)} genes for {disease_id} (EFO: {efo_id})")
            return genes

        except Exception as e:
            logger.error(f"OpenTargets get_disease_genes({disease_id}) failed: {e}")
            return []

    def get_disease_uniprot_ids(self, disease_id: str) -> set[str]:
        """
        Return flat set of UniProt IDs for disease-associated genes.
        Drop-in replacement for DisGeNETClient.get_disease_uniprot_ids().
        """
        genes = self.get_disease_genes(disease_id)
        uniprot_ids = set()
        for g in genes:
            for uid in g.get("uniprot_ids", []):
                uid = uid.strip()
                if uid and len(uid) >= 5:
                    uniprot_ids.add(uid)
        return uniprot_ids

    def get_disease_gene_symbols(self, disease_id: str) -> set[str]:
        """Return flat set of gene symbols. Drop-in replacement for DisGeNET equivalent."""
        genes = self.get_disease_genes(disease_id)
        return {g["gene_symbol"] for g in genes if g.get("gene_symbol")}

    # ── Reverse lookup ─────────────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_gene_diseases(self, gene_symbol: str, limit: int = 50) -> list[dict]:
        """
        Reverse lookup: what diseases is this gene associated with?
        Used for disease-disease similarity layer (Layer 10).
        """
        query = """
        query GeneDisease($symbol: String!, $size: Int!) {
          target(ensemblId: $symbol) {
            id
            approvedSymbol
            associatedDiseases(page: {index: 0, size: $size}) {
              rows {
                disease {
                  id
                  name
                }
                score
              }
            }
          }
        }
        """
        # Note: this query uses Ensembl ID not gene symbol — resolve first
        # For simplicity, use the search to find the gene
        try:
            # Search for gene by symbol to get Ensembl ID
            search_q = """
            query($q: String!) {
              targets(q: $q, page: {index: 0, size: 1}) {
                rows { id approvedSymbol }
              }
            }
            """
            r = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={"query": search_q, "variables": {"q": gene_symbol}},
                timeout=15,
            )
            r.raise_for_status()
            targets = r.json().get("data", {}).get("targets", {}).get("rows", [])
            if not targets:
                return []

            ensembl_id = targets[0]["id"]
            time.sleep(0.3)

            # Now get disease associations
            r2 = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={"query": query, "variables": {"symbol": ensembl_id, "size": limit}},
                timeout=20,
            )
            r2.raise_for_status()
            data = r2.json()
            target_data = data.get("data", {}).get("target") or {}
            rows = target_data.get("associatedDiseases", {}).get("rows", [])

            return [
                {
                    "disease_id":   row["disease"]["id"],
                    "disease_name": row["disease"]["name"],
                    "score":        row.get("score", 0),
                    "source":       "OpenTargets",
                }
                for row in rows
            ]

        except Exception as e:
            logger.debug(f"get_gene_diseases({gene_symbol}) failed: {e}")
            return []