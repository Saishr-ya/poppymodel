"""
src/ingestion/opentargets_client.py

OpenTargets Platform API client.

Fix #13: Removed hardcoded _SEED_EFO_MAP entirely. The map was a source of
         silent bugs (e.g. ORPHA:422 mapped to EFO_0000222 = AML, not PAH).
         All disease IDs now go through _dynamic_efo_lookup(), which is
         cached for 365 days — so after the first run it's as fast as the
         seed map was, with no maintenance burden.

Previous fixes:
  ORPHA:33069 (Dravet)        → MONDO_0100135 (was EFO_0009897)
  ORPHA:101435 (Microcephaly) → MONDO_0015469 (was EFO_0000354)
  ORPHA:422 (PAH)             → EFO_0001361   (was EFO_0000222 = AML)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

GRAPHQL_URL  = "https://api.platform.opentargets.org/api/v4/graphql"
ORPHANET_API = "https://api.orphacode.org/EN/ClinicalEntity"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "PoppyRepurposingEngine/1.0 (research)",
}

DEFAULT_SCORE_THRESHOLD = 0.1

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

_DISEASE_NAME_SEARCH_QUERY = """
query DiseaseSearch($query: String!) {
  diseases(q: $query, page: {index: 0, size: 10}) {
    rows {
      id
      name
      dbXRefs
    }
  }
}
"""

_BATCH_PROTEIN_IDS_QUERY = """
query BatchProteinIds($ids: [String!]!) {
  targets(ensemblIds: $ids) {
    id
    proteinIds {
      id
      source
    }
  }
}
"""


class OpenTargetsClient:

    def __init__(self, score_threshold: float = DEFAULT_SCORE_THRESHOLD):
        self.score_threshold = score_threshold

    # ── EFO ID resolution ──────────────────────────────────────────────────────

    def get_efo_id(self, disease_id: str) -> Optional[str]:
        """
        Resolve any Orphanet/OMIM ID to an OpenTargets EFO/MONDO ID.

        Fix #13: No longer uses a hardcoded seed map. All IDs go through
        _dynamic_efo_lookup(), which is cached for 365 days after first call.
        """
        return self._dynamic_efo_lookup(disease_id)

    @cached_api_call(ttl_seconds=86400 * 365)
    def _dynamic_efo_lookup(self, disease_id: str) -> Optional[str]:
        """
        Resolve a disease ID dynamically:
          1. Fetch disease name from Orphanet API
          2. Search OpenTargets by name + cross-reference match
          3. Fall back to OLS4/MONDO xref lookup
        Result is cached for 1 year.
        """
        disease_name = self._get_disease_name(disease_id)
        if disease_name:
            result = self._search_ot_by_name(disease_id, disease_name)
            if result:
                logger.info(
                    f"Resolved {disease_id} → {result} (via name '{disease_name}')"
                )
                return result

        if disease_id.startswith("ORPHA:"):
            result = self._search_via_mondo(disease_id)
            if result:
                logger.info(f"Resolved {disease_id} → {result} (via MONDO xref)")
                return result

        logger.warning(f"OpenTargets: could not resolve EFO ID for {disease_id}")
        return None

    def _get_disease_name(self, disease_id: str) -> Optional[str]:
        if disease_id.startswith("ORPHA:"):
            orpha_num = disease_id.replace("ORPHA:", "")
            try:
                r = requests.get(f"{ORPHANET_API}/{orpha_num}/Name/en", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    return data.get("Name") or data.get("name")
            except Exception as e:
                logger.debug(f"Orphanet name lookup failed for {disease_id}: {e}")
        return None

    def _search_ot_by_name(self, disease_id: str, disease_name: str) -> Optional[str]:
        orpha_num = disease_id.replace("ORPHA:", "").replace("OMIM:", "")
        xref_patterns = [f"Orphanet:{orpha_num}", f"Orphanet_{orpha_num}", orpha_num]
        try:
            r = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={
                    "query": _DISEASE_NAME_SEARCH_QUERY,
                    "variables": {"query": disease_name},
                },
                timeout=20,
            )
            r.raise_for_status()
            rows = r.json().get("data", {}).get("diseases", {}).get("rows", [])
            # Priority 1: exact xref match
            for row in rows:
                if any(pat in str(x) for x in row.get("dbXRefs", []) for pat in xref_patterns):
                    return row["id"]
            # Priority 2: name substring match
            name_lower = disease_name.lower()
            for row in rows:
                if name_lower in row.get("name", "").lower():
                    return row["id"]
            # Priority 3: first result
            if rows:
                return rows[0]["id"]
        except Exception as e:
            logger.debug(f"OT name search failed for '{disease_name}': {e}")
        return None

    def _search_via_mondo(self, disease_id: str) -> Optional[str]:
        orpha_num = disease_id.replace("ORPHA:", "")
        try:
            r = requests.get(
                f"https://www.ebi.ac.uk/ols4/api/terms?obo_id=Orphanet:{orpha_num}",
                timeout=15,
            )
            r.raise_for_status()
            for term in r.json().get("_embedded", {}).get("terms", []):
                for xref in term.get("annotation", {}).get("database_cross_reference", []):
                    if xref.startswith("EFO:"):
                        return xref.replace("EFO:", "EFO_")
        except Exception as e:
            logger.debug(f"MONDO lookup failed for {disease_id}: {e}")
        return None

    # ── Batch UniProt lookup ───────────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 90)
    def _batch_uniprot_lookup(self, ensembl_ids: tuple) -> dict:
        """Single GraphQL call for all Ensembl IDs → UniProt IDs."""
        if not ensembl_ids:
            return {}
        try:
            r = requests.post(
                GRAPHQL_URL,
                headers=HEADERS,
                json={
                    "query": _BATCH_PROTEIN_IDS_QUERY,
                    "variables": {"ids": list(ensembl_ids)},
                },
                timeout=30,
            )
            r.raise_for_status()
            result = {}
            for target in r.json().get("data", {}).get("targets", []):
                swissprot = [
                    p["id"] for p in target.get("proteinIds", [])
                    if p.get("source") == "uniprot_swissprot"
                ]
                trembl = [
                    p["id"] for p in target.get("proteinIds", [])
                    if p.get("source") == "uniprot_trembl"
                ]
                result[target["id"]] = swissprot if swissprot else trembl
            return result
        except Exception as e:
            logger.debug(f"Batch UniProt lookup failed: {e}")
            return {}

    # ── Gene-disease associations ──────────────────────────────────────────────

    def get_disease_genes(self, disease_id: str, limit: int = 200) -> list[dict]:
        efo_id = self.get_efo_id(disease_id)
        if not efo_id:
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
                logger.warning(f"OpenTargets: no data for EFO {efo_id} ({disease_id})")
                return []

            rows = disease_data.get("associatedTargets", {}).get("rows", [])
            ensembl_ids = tuple(
                row["target"]["id"]
                for row in rows
                if row.get("target", {}).get("id")
            )
            uniprot_map = self._batch_uniprot_lookup(ensembl_ids)

            genes = []
            for row in rows:
                score = float(row.get("score") or 0)
                if score < self.score_threshold:
                    continue
                target     = row.get("target", {})
                ensembl_id = target.get("id", "")
                evidence_types = [
                    ds["id"] for ds in row.get("datatypeScores", [])
                    if ds.get("score", 0) > 0
                ]
                genes.append({
                    "gene_symbol":    target.get("approvedSymbol", ""),
                    "ensembl_id":     ensembl_id,
                    "uniprot_ids":    uniprot_map.get(ensembl_id, []),
                    "ot_score":       score,
                    "evidence_types": evidence_types,
                    "source":         "OpenTargets",
                    "gda_score":      score,
                    "assoc_type": (
                        "Causal" if "genetic_association" in evidence_types
                        else "AlteredExpression"
                    ),
                })

            genes.sort(key=lambda g: g["ot_score"], reverse=True)
            logger.info(f"OpenTargets: {len(genes)} genes for {disease_id} (EFO: {efo_id})")
            return genes

        except Exception as e:
            logger.error(f"OpenTargets get_disease_genes({disease_id}) failed: {e}")
            return []

    def get_disease_uniprot_ids(self, disease_id: str) -> set[str]:
        return {
            uid.strip()
            for g in self.get_disease_genes(disease_id)
            for uid in g.get("uniprot_ids", [])
            if uid.strip() and len(uid.strip()) >= 5
        }

    def get_disease_gene_symbols(self, disease_id: str) -> set[str]:
        return {
            g["gene_symbol"]
            for g in self.get_disease_genes(disease_id)
            if g.get("gene_symbol")
        }

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_gene_diseases(self, gene_symbol: str, limit: int = 50) -> list[dict]:
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
            associatedDiseases(page: {index: 0, size: $size}) {
              rows { disease { id name } score }
            }
          }
        }
        """
        try:
            r = requests.post(
                GRAPHQL_URL, headers=HEADERS,
                json={"query": search_q, "variables": {"q": gene_symbol}},
                timeout=15,
            )
            r.raise_for_status()
            targets = r.json().get("data", {}).get("targets", {}).get("rows", [])
            if not targets:
                return []
            ensembl_id = targets[0]["id"]
            time.sleep(0.3)
            r2 = requests.post(
                GRAPHQL_URL, headers=HEADERS,
                json={"query": gene_disease_q, "variables": {"id": ensembl_id, "size": limit}},
                timeout=20,
            )
            r2.raise_for_status()
            rows = (
                r2.json().get("data", {})
                .get("target", {})
                .get("associatedDiseases", {})
                .get("rows", [])
            )
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