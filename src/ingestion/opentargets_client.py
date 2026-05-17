"""
src/ingestion/opentargets_client.py

OpenTargets Platform API client — replaces DisGeNET as the primary
gene-disease association source.

FIXES IN THIS VERSION:
  1. Removed proteinAnnotations field (removed from OpenTargets API)
  2. UniProt IDs now fetched via separate proteinIds query per target
  3. Removed time.sleep() from inside cached function body
  4. import time moved to module level
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "PoppyRepurposingEngine/1.0 (research)",
}

DEFAULT_SCORE_THRESHOLD = 0.1

ORPHA_TO_EFO: dict[str, str] = {
    "ORPHA:422":    "EFO_0000222",   # Pulmonary arterial hypertension
    "ORPHA:77":     "EFO_0000249",   # Gaucher disease type 1
    "ORPHA:33069":  "EFO_0005271",   # Dravet syndrome
    "ORPHA:566":    "EFO_0009620",   # Pompe disease
    "ORPHA:355":    "EFO_0000339",   # CML (imatinib ground truth)
    "ORPHA:101435": "EFO_0000354",   # Microcephaly (negative control)
    "ORPHA:586":    "EFO_0004259",   # Polycystic ovary syndrome
}

OMIM_TO_EFO: dict[str, str] = {}

# ── GraphQL queries ────────────────────────────────────────────────────────────

# proteinAnnotations removed from OT API — UniProt now via separate proteinIds query
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

_TARGET_PROTEIN_IDS_QUERY = """
query TargetProteinIds($id: String!) {
  target(ensemblId: $id) {
    id
    proteinIds {
      id
      source
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
        if disease_id in ORPHA_TO_EFO:
            return ORPHA_TO_EFO[disease_id]
        if disease_id in OMIM_TO_EFO:
            return OMIM_TO_EFO[disease_id]
        return self._search_efo_id(disease_id)

    @cached_api_call(ttl_seconds=86400 * 90)
    def _search_efo_id(self, disease_id: str) -> Optional[str]:
        if disease_id.startswith("ORPHA:"):
            search_term = f"Orphanet_{disease_id.replace('ORPHA:', '')}"
        elif disease_id.startswith("OMIM:"):
            search_term = f"OMIM_{disease_id.replace('OMIM:', '')}"
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
                if any(search_term in str(xref) for xref in xrefs):
                    efo_id = row.get("id")
                    logger.info(f"Mapped {disease_id} → {efo_id} ({row.get('name')})")
                    return efo_id

            if rows:
                efo_id = rows[0].get("id")
                logger.warning(
                    f"No exact xref match for {disease_id}. "
                    f"Using first result: {efo_id} ({rows[0].get('name')}). "
                    f"Verify and add to ORPHA_TO_EFO."
                )
                return efo_id

            logger.warning(f"OpenTargets: no disease found for {disease_id}.")
            return None

        except Exception as e:
            logger.error(f"OpenTargets EFO ID search failed for {disease_id}: {e}")
            return None

    # ── UniProt lookup (separate from disease query) ───────────────────────────

    @cached_api_call(ttl_seconds=86400 * 90)
    def _get_uniprot_for_ensembl(self, ensembl_id: str) -> list[str]:
        """
        Fetch UniProt accessions for an Ensembl gene ID.
        Uses the proteinIds field which is stable in the current OT API.
        Cached 90 days — protein-gene mapping is highly stable.
        """
        try:
            r = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={
                    "query": _TARGET_PROTEIN_IDS_QUERY,
                    "variables": {"id": ensembl_id},
                },
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()

            if "errors" in data:
                logger.debug(f"GraphQL errors for {ensembl_id}: {data['errors']}")
                return []

            target = (data.get("data") or {}).get("target") or {}
            protein_ids = target.get("proteinIds") or []

            # Prefer SwissProt (reviewed) over TrEMBL (unreviewed)
            swissprot = [
                p["id"] for p in protein_ids
                if p.get("source") == "uniprot_swissprot"
            ]
            trembl = [
                p["id"] for p in protein_ids
                if p.get("source") == "uniprot_trembl"
            ]

            return swissprot if swissprot else trembl

        except Exception as e:
            logger.debug(f"UniProt lookup failed for {ensembl_id}: {e}")
            return []

    # ── Gene-disease associations ──────────────────────────────────────────────

    def get_disease_genes(
        self,
        disease_id: str,
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch gene-disease associations from OpenTargets.

        NOTE: Not decorated with @cached_api_call because it makes per-target
        UniProt sub-calls. The sub-calls (_get_uniprot_for_ensembl) are each
        individually cached for 90 days, so repeated runs are fast.

        Returns list of dicts with keys:
            gene_symbol, ensembl_id, uniprot_ids, ot_score,
            evidence_types, source, gda_score, assoc_type
        """
        efo_id = self.get_efo_id(disease_id)
        if not efo_id:
            logger.error(
                f"Cannot query OpenTargets: no EFO ID for {disease_id}. "
                f"Add to ORPHA_TO_EFO in opentargets_client.py."
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
                logger.error(
                    f"OpenTargets GraphQL errors for {disease_id}: {data['errors']}"
                )
                return []

            disease_data = data.get("data", {}).get("disease")
            if not disease_data:
                logger.warning(
                    f"OpenTargets: no data for EFO {efo_id} ({disease_id})"
                )
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

                # Fetch UniProt IDs via separate cached call (no sleep needed —
                # cache decorator handles rate limiting via its own 0.5s delay)
                uniprot_ids = (
                    self._get_uniprot_for_ensembl(ensembl_id)
                    if ensembl_id else []
                )

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
                    # DisGeNET-compatible aliases
                    "gda_score":      score,
                    "assoc_type": (
                        "Causal"
                        if "genetic_association" in evidence_types
                        else "AlteredExpression"
                    ),
                })

            genes.sort(key=lambda g: g["ot_score"], reverse=True)
            logger.info(
                f"OpenTargets: {len(genes)} genes for {disease_id} (EFO: {efo_id})"
            )
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
        """Return flat set of gene symbols."""
        genes = self.get_disease_genes(disease_id)
        return {g["gene_symbol"] for g in genes if g.get("gene_symbol")}

    # ── Reverse lookup ─────────────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_gene_diseases(self, gene_symbol: str, limit: int = 50) -> list[dict]:
        """Reverse lookup: what diseases is this gene associated with?"""
        search_q = """
        query($q: String!) {
          targets(q: $q, page: {index: 0, size: 1}) {
            rows { id approvedSymbol }
          }
        }
        """
        gene_disease_q = """
        query GeneDisease($id: String!, $size: Int!) {
          target(ensemblId: $id) {
            id
            approvedSymbol
            associatedDiseases(page: {index: 0, size: $size}) {
              rows {
                disease { id name }
                score
              }
            }
          }
        }
        """
        try:
            r = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={"query": search_q, "variables": {"q": gene_symbol}},
                timeout=15,
            )
            r.raise_for_status()
            targets = (
                r.json().get("data", {}).get("targets", {}).get("rows", [])
            )
            if not targets:
                return []

            ensembl_id = targets[0]["id"]
            time.sleep(0.3)

            r2 = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={
                    "query": gene_disease_q,
                    "variables": {"id": ensembl_id, "size": limit},
                },
                timeout=20,
            )
            r2.raise_for_status()
            target_data = r2.json().get("data", {}).get("target") or {}
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