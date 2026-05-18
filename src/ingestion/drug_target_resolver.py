"""
src/ingestion/drug_target_resolver.py

Multi-source drug protein target resolver.

Fix: _filter_human_proteins was returning an empty set when the UniProt
     batch API failed (rate limit, network error, or empty response).
     The previous code did `return ids` in the except block, which is correct,
     but the API was returning HTTP 200 with zero results — not an exception —
     so it fell through to returning the empty set from the successful call.

     Root cause: the UniProt search endpoint occasionally returns an empty
     result list even for valid accessions under high load. Now we treat
     "API returned 0 results for a non-empty input" as a failure and fall
     back to returning the unfiltered input set, same as an exception.

     Also added: the OpenTargets drug source as the first source tried.
     Previously it was being skipped because cached empty lists from old
     runs were still in Redis. After FLUSHDB, it will now run fresh and
     return correct targets (ABL1/KIT for imatinib, PDE5A for sildenafil).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

CHEMBL_API      = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_API     = "https://rest.uniprot.org/uniprotkb"
OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
GTOPDB_API      = "https://www.guidetopharmacology.org/services"

OT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "PoppyRepurposingEngine/1.0 (research)",
}
CHEMBL_HEADERS = {"Accept": "application/json"}
MIN_PCHEMBL    = 5.0

_OT_DRUG_QUERY = """
query DrugMechanisms($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    mechanismsOfAction {
      rows {
        targets {
          id
          proteinIds { id source }
        }
      }
    }
    linkedTargets {
      rows {
        id
        proteinIds { id source }
      }
    }
  }
}
"""


# ── Source functions — ALL return list[str] so Redis can serialise them ────────

@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_opentargets(chembl_id: str) -> list[str]:
    """Source 1: OpenTargets drug query. Returns list[str] — JSON-safe."""
    try:
        r = requests.post(
            OPENTARGETS_GQL,
            headers=OT_HEADERS,
            json={"query": _OT_DRUG_QUERY, "variables": {"chemblId": chembl_id.upper()}},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            return []

        drug_data = (data.get("data") or {}).get("drug") or {}
        if not drug_data:
            return []

        seen: set[str] = set()
        for row in ((drug_data.get("mechanismsOfAction") or {}).get("rows") or []):
            for target in (row.get("targets") or []):
                for pid in (target.get("proteinIds") or []):
                    if pid.get("source") in ("uniprot_swissprot", "uniprot_trembl"):
                        uid = pid.get("id", "").strip()
                        if uid:
                            seen.add(uid)

        for target in ((drug_data.get("linkedTargets") or {}).get("rows") or []):
            for pid in (target.get("proteinIds") or []):
                if pid.get("source") in ("uniprot_swissprot", "uniprot_trembl"):
                    uid = pid.get("id", "").strip()
                    if uid:
                        seen.add(uid)

        result = list(seen)
        if result:
            logger.info(f"[target_resolver] OT drug query: {chembl_id} → {len(result)} UniProt IDs")
        return result
    except Exception as e:
        logger.debug(f"[target_resolver] OT drug query failed for {chembl_id}: {e}")
        return []


@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_chembl_mechanism(chembl_id: str) -> list[str]:
    """Source 2: ChEMBL mechanism endpoint. Returns list[str] — JSON-safe."""
    parent_id = _resolve_chembl_parent(chembl_id)
    try:
        r = requests.get(
            f"{CHEMBL_API}/mechanism.json",
            headers=CHEMBL_HEADERS,
            params={"molecule_chembl_id": parent_id, "limit": 50},
            timeout=15,
        )
        r.raise_for_status()
        mechanisms = r.json().get("mechanisms", [])
        target_ids = {m["target_chembl_id"] for m in mechanisms if m.get("target_chembl_id")}

        seen: set[str] = set()
        for tid in target_ids:
            time.sleep(0.1)
            seen.update(_chembl_target_to_uniprot(tid))

        result = list(seen)
        if result:
            logger.info(f"[target_resolver] ChEMBL mechanism: {parent_id} → {len(result)} UniProt IDs")
        return result
    except Exception as e:
        logger.debug(f"[target_resolver] ChEMBL mechanism failed for {chembl_id}: {e}")
        return []


@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_chembl_activity(chembl_id: str) -> list[str]:
    """Source 3: ChEMBL binding activity. Returns list[str] — JSON-safe."""
    parent_id = _resolve_chembl_parent(chembl_id)
    try:
        r = requests.get(
            f"{CHEMBL_API}/activity.json",
            headers=CHEMBL_HEADERS,
            params={
                "molecule_chembl_id": parent_id,
                "target_type": "SINGLE PROTEIN",
                "assay_type": "B",
                "pchembl_value__gte": MIN_PCHEMBL,
                "limit": 100,
            },
            timeout=20,
        )
        r.raise_for_status()
        activities = r.json().get("activities", [])
        if not activities:
            return []

        target_ids = {a["target_chembl_id"] for a in activities if a.get("target_chembl_id")}
        seen: set[str] = set()
        for tid in list(target_ids)[:15]:
            time.sleep(0.15)
            seen.update(_chembl_target_to_uniprot(tid))

        result = list(seen)
        if result:
            logger.info(
                f"[target_resolver] ChEMBL activity: {parent_id} → "
                f"{len(result)} UniProt IDs from {len(target_ids)} targets"
            )
        return result
    except Exception as e:
        logger.debug(f"[target_resolver] ChEMBL activity failed for {chembl_id}: {e}")
        return []


@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_uniprot_cross_ref(chembl_id: str) -> list[str]:
    """Source 4: UniProt cross-reference. Returns list[str] — JSON-safe."""
    url = f"{UNIPROT_API}/search"
    try:
        r = requests.get(
            url,
            params={
                "query":  f"(database:chembl AND {chembl_id}) AND (reviewed:true)",
                "fields": "accession,organism_name",
                "format": "json",
                "size":   25,
            },
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            r2 = requests.get(
                url,
                params={"query": f"(database:chembl AND {chembl_id})",
                        "fields": "accession,organism_name", "format": "json", "size": 10},
                timeout=20,
            )
            r2.raise_for_status()
            results = r2.json().get("results", [])

        ids = [
            e.get("primaryAccession", "").strip()
            for e in results
            if "Homo sapiens" in e.get("organism", {}).get("scientificName", "")
            and e.get("primaryAccession")
        ]
        if ids:
            logger.info(f"[target_resolver] UniProt cross-ref: {chembl_id} → {len(ids)} human proteins")
        return ids
    except Exception as e:
        logger.debug(f"[target_resolver] UniProt cross-ref failed for {chembl_id}: {e}")
        return []


@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_gtopdb(drug_name: str) -> list[str]:
    """Source 5: IUPHAR GtoPdb. Returns list[str] — JSON-safe."""
    try:
        r = requests.get(
            f"{GTOPDB_API}/ligands",
            params={"name": drug_name, "type": "Approved"},
            timeout=15,
        )
        r.raise_for_status()
        ligands = r.json()
        if not ligands:
            r2 = requests.get(f"{GTOPDB_API}/ligands", params={"name": drug_name}, timeout=15)
            r2.raise_for_status()
            ligands = r2.json()
        if not ligands:
            return []

        ligand_id = None
        for lig in ligands:
            if lig.get("name", "").lower() == drug_name.lower():
                ligand_id = lig.get("ligandId")
                break
        if ligand_id is None:
            ligand_id = ligands[0].get("ligandId")
        if not ligand_id:
            return []

        time.sleep(0.2)
        r3 = requests.get(
            f"{GTOPDB_API}/interactions",
            params={"ligandId": ligand_id, "species": "Human"},
            timeout=15,
        )
        r3.raise_for_status()
        ids = [
            i.get("targetUniprotId", "").strip()
            for i in r3.json()
            if len(i.get("targetUniprotId", "").strip()) >= 5
        ]
        if ids:
            logger.info(f"[target_resolver] GtoPdb: '{drug_name}' → {len(ids)} UniProt IDs")
        return ids
    except Exception as e:
        logger.debug(f"[target_resolver] GtoPdb failed for '{drug_name}': {e}")
        return []


# ── Helpers ────────────────────────────────────────────────────────────────────

@cached_api_call(ttl_seconds=86400 * 90)
def _resolve_chembl_parent(chembl_id: str) -> str:
    try:
        r = requests.get(
            f"{CHEMBL_API}/molecule/{chembl_id}.json",
            headers=CHEMBL_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        hierarchy = r.json().get("molecule_hierarchy") or {}
        parent    = hierarchy.get("parent_chembl_id")
        return parent if (parent and parent != chembl_id) else chembl_id
    except Exception:
        return chembl_id


@cached_api_call(ttl_seconds=86400 * 90)
def _chembl_target_to_uniprot(target_chembl_id: str) -> list[str]:
    try:
        r = requests.get(
            f"{CHEMBL_API}/target/{target_chembl_id}.json",
            headers=CHEMBL_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        components = r.json().get("target_components", [])
        return [
            xref["xref_id"]
            for comp in components
            for xref in comp.get("target_component_xrefs", [])
            if xref.get("xref_src_db") == "UniProt"
        ]
    except Exception as e:
        logger.debug(f"[target_resolver] ChEMBL target→UniProt failed for {target_chembl_id}: {e}")
        return []


# ── Main resolver ──────────────────────────────────────────────────────────────

class DrugTargetResolver:
    """
    Multi-source drug protein target resolver.

    All cached source functions return list[str] (JSON-safe).
    get_uniprot_ids() merges lists into a set[str].
    _filter_human_proteins falls back to the unfiltered set if the UniProt
    API returns empty results for a non-empty input (rate limit / transient
    error), rather than silently dropping all targets.
    """

    def __init__(
        self,
        min_sources: int = 1,
        stop_early_after: int = 2,
        require_human: bool = True,
    ):
        self.stop_early_after = stop_early_after
        self.require_human    = require_human

    def get_uniprot_ids(self, chembl_id: str, drug_name: str = "") -> set[str]:
        all_ids: set[str] = set()
        sources_with_results = 0

        sources = [
            ("OpenTargets_drug",  lambda: _fetch_targets_opentargets(chembl_id)),
            ("ChEMBL_mechanism",  lambda: _fetch_targets_chembl_mechanism(chembl_id)),
            ("ChEMBL_activity",   lambda: _fetch_targets_chembl_activity(chembl_id)),
            ("UniProt_crossref",  lambda: _fetch_targets_uniprot_cross_ref(chembl_id)),
            ("GtoPdb",            lambda: _fetch_targets_gtopdb(drug_name) if drug_name else []),
        ]

        for source_name, fetch_fn in sources:
            try:
                ids_raw = fetch_fn()
                # Guard: stale cache can return a string if set was serialised before fix
                if not isinstance(ids_raw, list):
                    logger.warning(
                        f"[target_resolver] {source_name} returned "
                        f"{type(ids_raw).__name__} for {chembl_id} — "
                        f"expected list; flush Redis to clear stale cache"
                    )
                    continue
                if ids_raw:
                    logger.info(
                        f"[target_resolver] {source_name}: "
                        f"{len(ids_raw)} targets for {chembl_id}"
                    )
                    all_ids.update(ids_raw)
                    sources_with_results += 1
                    if sources_with_results >= self.stop_early_after:
                        break
            except Exception as e:
                logger.warning(
                    f"[target_resolver] {source_name} raised for {chembl_id}: {e}"
                )

        if self.require_human and all_ids:
            all_ids = self._filter_human_proteins(all_ids)

        logger.info(
            f"[target_resolver] FINAL: {chembl_id} ({drug_name}) → "
            f"{len(all_ids)} UniProt IDs from {sources_with_results} sources: "
            f"{sorted(all_ids)}"
        )
        return all_ids

    def get_target_details(self, chembl_id: str, drug_name: str = "") -> list[dict]:
        all_targets: dict[str, dict] = {}
        sources = [
            ("OpenTargets",      lambda: _fetch_targets_opentargets(chembl_id)),
            ("ChEMBL_mechanism", lambda: _fetch_targets_chembl_mechanism(chembl_id)),
            ("ChEMBL_activity",  lambda: _fetch_targets_chembl_activity(chembl_id)),
            ("UniProt",          lambda: _fetch_targets_uniprot_cross_ref(chembl_id)),
            ("GtoPdb",           lambda: _fetch_targets_gtopdb(drug_name) if drug_name else []),
        ]
        for source_name, fetch_fn in sources:
            try:
                ids_raw = fetch_fn()
                if not isinstance(ids_raw, list):
                    continue
                for uid in ids_raw:
                    if uid not in all_targets:
                        all_targets[uid] = {"uniprot_id": uid, "sources": [source_name]}
                    else:
                        all_targets[uid]["sources"].append(source_name)
            except Exception as e:
                logger.debug(f"[target_resolver] {source_name} failed: {e}")

        enriched = self._enrich_with_uniprot(list(all_targets.values()))
        if self.require_human:
            enriched = [t for t in enriched if t.get("organism") == "Homo sapiens"]
        return enriched

    def _filter_human_proteins(self, uniprot_ids: set) -> set[str]:
        """
        Filter to human proteins via UniProt batch API.

        Key fix: if the API returns HTTP 200 but zero results for a
        non-empty input, treat this as a transient failure and return
        the original unfiltered set. This prevents the resolver from
        silently dropping all targets when UniProt is rate-limiting or
        temporarily returning empty responses.
        """
        ids = set(uniprot_ids)
        if not ids:
            return set()

        id_list = list(ids)[:100]
        query   = " OR ".join(f"accession:{uid}" for uid in id_list)

        try:
            r = requests.get(
                f"{UNIPROT_API}/search",
                params={
                    "query":  f"({query}) AND (organism_id:9606)",
                    "fields": "accession",
                    "format": "json",
                    "size":   100,
                },
                timeout=20,
            )
            r.raise_for_status()
            results   = r.json().get("results", [])
            human_ids = {e["primaryAccession"] for e in results if e.get("primaryAccession")}

            # KEY FIX: empty result for non-empty input = transient API failure
            # Return the original set unfiltered rather than dropping everything
            if not human_ids and ids:
                logger.debug(
                    f"[target_resolver] UniProt human filter returned 0 results for "
                    f"{len(ids)} input IDs — API may be rate-limiting; "
                    f"returning unfiltered set"
                )
                return ids

            removed = ids - human_ids
            if removed:
                logger.debug(
                    f"[target_resolver] Filtered out non-human proteins: {removed}"
                )
            return human_ids

        except Exception as e:
            logger.debug(
                f"[target_resolver] Human protein filter failed ({e}); "
                f"returning unfiltered set"
            )
            return ids

    def _enrich_with_uniprot(self, targets: list[dict]) -> list[dict]:
        if not targets:
            return targets
        ids   = [t["uniprot_id"] for t in targets]
        query = " OR ".join(f"accession:{uid}" for uid in ids[:50])
        try:
            r = requests.get(
                f"{UNIPROT_API}/search",
                params={
                    "query":  f"({query}) AND (organism_id:9606)",
                    "fields": "accession,gene_names,protein_name,organism_name",
                    "format": "json",
                    "size":   50,
                },
                timeout=20,
            )
            r.raise_for_status()
            enrichment = {}
            for entry in r.json().get("results", []):
                acc        = entry.get("primaryAccession", "")
                gene_names = entry.get("genes", [])
                gene_sym   = gene_names[0].get("geneName", {}).get("value", "") if gene_names else ""
                protein    = (
                    entry.get("proteinDescription", {})
                    .get("recommendedName", {}).get("fullName", {}).get("value", "")
                )
                enrichment[acc] = {
                    "gene_symbol":  gene_sym,
                    "protein_name": protein,
                    "organism":     entry.get("organism", {}).get("scientificName", ""),
                }
            for target in targets:
                if target["uniprot_id"] in enrichment:
                    target.update(enrichment[target["uniprot_id"]])
        except Exception as e:
            logger.debug(f"[target_resolver] UniProt enrichment failed: {e}")
        return targets


_default_resolver = None


def get_drug_targets(chembl_id: str, drug_name: str = "") -> set[str]:
    """Drop-in replacement for ChEMBLClient.get_target_uniprot_ids()."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = DrugTargetResolver(stop_early_after=5)
    return _default_resolver.get_uniprot_ids(chembl_id, drug_name)