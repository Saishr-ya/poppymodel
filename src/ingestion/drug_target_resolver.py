"""
src/ingestion/drug_target_resolver.py

Multi-source drug protein target resolver.

Problem: ChEMBL's mechanism endpoint has coverage gaps — sildenafil, for example,
has no mechanism records despite being a well-characterized PDE5 inhibitor.
Hardcoding targets defeats the purpose of a computational engine.

Solution: Try four independent databases in order, merge results, return union.

Source priority order:
  1. OpenTargets drug query    — ChEMBL ID → known drug mechanisms (GraphQL)
  2. ChEMBL activity endpoint  — binding assays (IC50/Ki) even without mechanism records
  3. UniChem + UniProt         — InChIKey-based cross-reference to any database
  4. Guide to Pharmacology     — curated ligand-target interactions (IUPHAR)

All sources are free, no API key required, fully programmatic.
Results are cached per source. The union of all sources is returned.

Usage:
    resolver = DrugTargetResolver()
    uniprot_ids = resolver.get_uniprot_ids("CHEMBL1520", "Sildenafil")
    # Returns {'O76074'} (PDE5A) even though ChEMBL mechanism table is empty

Integration:
    Replace ChEMBLClient.get_target_uniprot_ids() calls with this resolver.
    The resolver calls ChEMBL internally as one of its sources.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

# ── API endpoints ──────────────────────────────────────────────────────────────

CHEMBL_API       = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_API      = "https://rest.uniprot.org/uniprotkb"
UNICHEM_API      = "https://www.ebi.ac.uk/unichem/rest"
OPENTARGETS_GQL  = "https://api.platform.opentargets.org/api/v4/graphql"
GTOPDB_API       = "https://www.guidetopharmacology.org/services"

OT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "PoppyRepurposingEngine/1.0 (research)",
}

CHEMBL_HEADERS = {"Accept": "application/json"}

# Confidence threshold for including activity-based targets
# pChEMBL value >= 5 = IC50/Ki <= 10 µM (meaningful binding)
MIN_PCHEMBL = 5.0


# ── Source 1: OpenTargets drug mechanism query ─────────────────────────────────

_OT_DRUG_QUERY = """
query DrugMechanisms($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id
    name
    mechanismsOfAction {
      rows {
        mechanismOfAction
        targetName
        targets {
          id
          approvedSymbol
          proteinIds {
            id
            source
          }
        }
      }
    }
    linkedTargets {
      rows {
        id
        approvedSymbol
        proteinIds {
          id
          source
        }
      }
    }
  }
}
"""


@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_opentargets(chembl_id: str) -> set[str]:
    """
    Source 1: OpenTargets drug query.

    OpenTargets has a drug-centric API that returns mechanisms of action
    and linked targets for any ChEMBL ID. This is separate from and more
    complete than the ChEMBL mechanism endpoint.

    Returns set of UniProt IDs.
    """
    # Normalise: OpenTargets wants uppercase without prefix
    ot_drug_id = chembl_id.upper()

    try:
        r = requests.post(
            OPENTARGETS_GQL,
            headers=OT_HEADERS,
            json={
                "query": _OT_DRUG_QUERY,
                "variables": {"chemblId": ot_drug_id},
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()

        if "errors" in data:
            logger.debug(f"OT drug query errors for {chembl_id}: {data['errors']}")
            return set()

        drug_data = (data.get("data") or {}).get("drug") or {}
        if not drug_data:
            logger.debug(f"OT: no drug data for {chembl_id}")
            return set()

        uniprot_ids = set()

        # From mechanismsOfAction
        moa_rows = (
            drug_data.get("mechanismsOfAction") or {}
        ).get("rows") or []
        for row in moa_rows:
            for target in row.get("targets") or []:
                for pid in target.get("proteinIds") or []:
                    if pid.get("source") in ("uniprot_swissprot", "uniprot_trembl"):
                        uid = pid.get("id", "").strip()
                        if uid:
                            uniprot_ids.add(uid)

        # From linkedTargets (broader, includes indirect)
        linked_rows = (
            drug_data.get("linkedTargets") or {}
        ).get("rows") or []
        for target in linked_rows:
            for pid in target.get("proteinIds") or []:
                if pid.get("source") in ("uniprot_swissprot", "uniprot_trembl"):
                    uid = pid.get("id", "").strip()
                    if uid:
                        uniprot_ids.add(uid)

        if uniprot_ids:
            logger.info(
                f"[target_resolver] OT drug query: {chembl_id} → "
                f"{len(uniprot_ids)} UniProt IDs"
            )

        return uniprot_ids

    except Exception as e:
        logger.debug(f"[target_resolver] OT drug query failed for {chembl_id}: {e}")
        return set()


# ── Source 2: ChEMBL activity endpoint ────────────────────────────────────────

@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_chembl_activity(chembl_id: str) -> set[str]:
    """
    Source 2: ChEMBL binding activity data.

    Queries bioactivity records (IC50, Ki, Kd) for the drug.
    Filters to single-protein targets with pChEMBL >= 5 (IC50/Ki <= 10 µM).
    This works even when the mechanism endpoint is empty.

    Returns set of UniProt IDs.
    """
    # Step 1: resolve salt to parent (mechanism data on parent)
    parent_id = _resolve_chembl_parent(chembl_id)

    # Step 2: get binding activity records
    url = f"{CHEMBL_API}/activity.json"
    params = {
        "molecule_chembl_id": parent_id,
        "target_type": "SINGLE PROTEIN",
        "assay_type": "B",          # binding assays only
        "pchembl_value__gte": MIN_PCHEMBL,
        "limit": 100,
    }

    try:
        r = requests.get(
            url, headers=CHEMBL_HEADERS, params=params, timeout=20
        )
        r.raise_for_status()
        activities = r.json().get("activities", [])

        if not activities:
            logger.debug(
                f"[target_resolver] ChEMBL activity: no binding data for {parent_id}"
            )
            return set()

        # Collect unique target ChEMBL IDs
        target_ids = {
            a["target_chembl_id"]
            for a in activities
            if a.get("target_chembl_id")
        }

        # Step 3: resolve each target to UniProt
        uniprot_ids = set()
        for tid in list(target_ids)[:15]:   # cap to avoid abuse
            time.sleep(0.15)
            uids = _chembl_target_to_uniprot(tid)
            uniprot_ids.update(uids)

        if uniprot_ids:
            logger.info(
                f"[target_resolver] ChEMBL activity: {parent_id} → "
                f"{len(uniprot_ids)} UniProt IDs from {len(target_ids)} targets"
            )

        return uniprot_ids

    except Exception as e:
        logger.debug(
            f"[target_resolver] ChEMBL activity failed for {chembl_id}: {e}"
        )
        return set()


@cached_api_call(ttl_seconds=86400 * 90)
def _resolve_chembl_parent(chembl_id: str) -> str:
    """Resolve a salt/formulation ChEMBL ID to the parent molecule."""
    url = f"{CHEMBL_API}/molecule/{chembl_id}.json"
    try:
        r = requests.get(url, headers=CHEMBL_HEADERS, timeout=15)
        r.raise_for_status()
        hierarchy = r.json().get("molecule_hierarchy") or {}
        parent = hierarchy.get("parent_chembl_id")
        if parent and parent != chembl_id:
            logger.debug(f"[target_resolver] {chembl_id} → parent {parent}")
            return parent
        return chembl_id
    except Exception:
        return chembl_id


@cached_api_call(ttl_seconds=86400 * 90)
def _chembl_target_to_uniprot(target_chembl_id: str) -> list[str]:
    """Resolve a ChEMBL target ID to UniProt accession(s)."""
    url = f"{CHEMBL_API}/target/{target_chembl_id}.json"
    try:
        r = requests.get(url, headers=CHEMBL_HEADERS, timeout=15)
        r.raise_for_status()
        components = r.json().get("target_components", [])
        ids = []
        for comp in components:
            for xref in comp.get("target_component_xrefs", []):
                if xref.get("xref_src_db") == "UniProt":
                    ids.append(xref["xref_id"])
        return ids
    except Exception as e:
        logger.debug(f"[target_resolver] ChEMBL target→UniProt failed for {target_chembl_id}: {e}")
        return []


# ── Source 3: UniChem + UniProt ────────────────────────────────────────────────

@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_uniprot_cross_ref(chembl_id: str) -> set[str]:
    """
    Source 3: UniProt cross-reference lookup.

    UniProt lets you search for all proteins that a compound (by ChEMBL ID)
    is documented to interact with in their curated database.

    Query: UniProt search for entries where chebi/chembl cross-ref matches
    and interaction type is ligand/inhibitor/substrate.

    Returns set of UniProt IDs (reviewed SwissProt entries only).
    """
    # UniProt REST API: search for reviewed entries mentioning this ChEMBL ID
    # in their ligand/binding site annotations
    url = f"{UNIPROT_API}/search"
    params = {
        "query": f"(database:chembl AND {chembl_id}) AND (reviewed:true)",
        "fields": "accession,id,gene_names,organism_name",
        "format": "json",
        "size": 25,
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])

        if not results:
            # Fallback: also try without the reviewed filter
            params["query"] = f"(database:chembl AND {chembl_id})"
            params["size"] = 10
            r2 = requests.get(url, params=params, timeout=20)
            r2.raise_for_status()
            results = r2.json().get("results", [])

        uniprot_ids = set()
        for entry in results:
            accession = entry.get("primaryAccession", "").strip()
            # Only include human proteins
            organism = (
                entry.get("organism", {}).get("scientificName", "")
            )
            if accession and "Homo sapiens" in organism:
                uniprot_ids.add(accession)

        if uniprot_ids:
            logger.info(
                f"[target_resolver] UniProt cross-ref: {chembl_id} → "
                f"{len(uniprot_ids)} human protein entries"
            )

        return uniprot_ids

    except Exception as e:
        logger.debug(
            f"[target_resolver] UniProt cross-ref failed for {chembl_id}: {e}"
        )
        return set()


# ── Source 4: Guide to Pharmacology (IUPHAR) ──────────────────────────────────

@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_gtopdb(drug_name: str) -> set[str]:
    """
    Source 4: IUPHAR Guide to Pharmacology.

    IUPHAR is the gold-standard curated ligand-target database.
    It has explicit UniProt IDs for every curated interaction.
    Query by drug name to find ligand ID, then get targets.

    Returns set of UniProt IDs.
    """
    try:
        # Step 1: search for ligand by name
        r = requests.get(
            f"{GTOPDB_API}/ligands",
            params={"name": drug_name, "type": "Approved"},
            timeout=15,
        )
        r.raise_for_status()
        ligands = r.json()

        if not ligands:
            # Try without type filter
            r2 = requests.get(
                f"{GTOPDB_API}/ligands",
                params={"name": drug_name},
                timeout=15,
            )
            r2.raise_for_status()
            ligands = r2.json()

        if not ligands:
            logger.debug(
                f"[target_resolver] GtoPdb: no ligand found for '{drug_name}'"
            )
            return set()

        # Take the best match (first result, exact name preferred)
        ligand_id = None
        drug_lower = drug_name.lower()
        for lig in ligands:
            if lig.get("name", "").lower() == drug_lower:
                ligand_id = lig.get("ligandId")
                break
        if ligand_id is None:
            ligand_id = ligands[0].get("ligandId")

        if not ligand_id:
            return set()

        # Step 2: get interactions for this ligand
        time.sleep(0.2)
        r3 = requests.get(
            f"{GTOPDB_API}/interactions",
            params={"ligandId": ligand_id, "species": "Human"},
            timeout=15,
        )
        r3.raise_for_status()
        interactions = r3.json()

        # Step 3: extract UniProt IDs from target entries
        uniprot_ids = set()
        for interaction in interactions:
            # GtoPdb returns targetUniprotId directly
            uid = interaction.get("targetUniprotId", "").strip()
            if uid and len(uid) >= 5:
                uniprot_ids.add(uid)

        if uniprot_ids:
            logger.info(
                f"[target_resolver] GtoPdb: '{drug_name}' (ligandId={ligand_id}) → "
                f"{len(uniprot_ids)} UniProt IDs"
            )

        return uniprot_ids

    except Exception as e:
        logger.debug(
            f"[target_resolver] GtoPdb failed for '{drug_name}': {e}"
        )
        return set()


# ── Source 5: ChEMBL mechanism endpoint (original) ────────────────────────────

@cached_api_call(ttl_seconds=86400 * 30)
def _fetch_targets_chembl_mechanism(chembl_id: str) -> set[str]:
    """
    Source 5: ChEMBL mechanism of action endpoint.
    The original source — kept in the pipeline but no longer the only source.
    Often empty for older approved drugs due to ChEMBL curation backlog.
    """
    parent_id = _resolve_chembl_parent(chembl_id)
    url = f"{CHEMBL_API}/mechanism.json"
    params = {"molecule_chembl_id": parent_id, "limit": 50}

    try:
        r = requests.get(url, headers=CHEMBL_HEADERS, params=params, timeout=15)
        r.raise_for_status()
        mechanisms = r.json().get("mechanisms", [])

        target_ids = {
            m["target_chembl_id"]
            for m in mechanisms
            if m.get("target_chembl_id")
        }

        uniprot_ids = set()
        for tid in target_ids:
            time.sleep(0.1)
            uids = _chembl_target_to_uniprot(tid)
            uniprot_ids.update(uids)

        if uniprot_ids:
            logger.info(
                f"[target_resolver] ChEMBL mechanism: {parent_id} → "
                f"{len(uniprot_ids)} UniProt IDs"
            )
        return uniprot_ids

    except Exception as e:
        logger.debug(
            f"[target_resolver] ChEMBL mechanism failed for {chembl_id}: {e}"
        )
        return set()


# ── Main resolver class ────────────────────────────────────────────────────────

class DrugTargetResolver:
    """
    Multi-source drug protein target resolver.

    Queries up to 5 independent databases and returns the union of all results.
    Stops early if a high-confidence source returns results (configurable).

    Sources tried in order:
        1. OpenTargets drug query       (most comprehensive, curated MOA)
        2. ChEMBL mechanism endpoint    (original source, often empty)
        3. ChEMBL binding activity      (broader, includes assay data)
        4. UniProt cross-reference      (reviewed entries only)
        5. Guide to Pharmacology        (IUPHAR curated, gold standard)

    Usage:
        resolver = DrugTargetResolver()
        ids = resolver.get_uniprot_ids("CHEMBL1520", "Sildenafil")
        # → {'O76074'} (PDE5A) from OpenTargets + GtoPdb

    Integration with ChEMBLClient:
        Inject resolver into TargetOverlapLayer and NetworkProximityLayer
        instead of calling chembl.get_target_uniprot_ids() directly.
    """

    def __init__(
        self,
        min_sources: int = 1,
        stop_early_after: int = 2,
        require_human: bool = True,
    ):
        """
        Args:
            min_sources:       Minimum number of sources to try before returning.
            stop_early_after:  Stop after this many sources return results.
                               Set to 5 to always query all sources (most complete).
                               Set to 1 to return as soon as any source succeeds (fastest).
            require_human:     If True, filter to human proteins only via UniProt validation.
        """
        self.min_sources = min_sources
        self.stop_early_after = stop_early_after
        self.require_human = require_human

    def get_uniprot_ids(
        self,
        chembl_id: str,
        drug_name: str = "",
    ) -> set[str]:
        """
        Fetch UniProt IDs for a drug's protein targets from all available sources.

        Args:
            chembl_id:  ChEMBL compound ID (CHEMBL192, CHEMBL1520, etc.)
            drug_name:  Drug name string (used for GtoPdb and UniProt queries)

        Returns:
            Set of UniProt accession strings. Empty set if no source returns data.
        """
        all_ids: set[str] = set()
        sources_with_results = 0

        sources = [
            ("OpenTargets_drug",    lambda: _fetch_targets_opentargets(chembl_id)),
            ("ChEMBL_mechanism",    lambda: _fetch_targets_chembl_mechanism(chembl_id)),
            ("ChEMBL_activity",     lambda: _fetch_targets_chembl_activity(chembl_id)),
            ("UniProt_crossref",    lambda: _fetch_targets_uniprot_cross_ref(chembl_id)),
            ("GtoPdb",              lambda: _fetch_targets_gtopdb(drug_name) if drug_name else set()),
        ]

        for source_name, fetch_fn in sources:
            try:
                ids = fetch_fn()
                if ids:
                    logger.info(
                        f"[target_resolver] {source_name}: "
                        f"{len(ids)} targets for {chembl_id}"
                    )
                    all_ids.update(ids)
                    sources_with_results += 1

                    if sources_with_results >= self.stop_early_after:
                        break
                else:
                    logger.debug(
                        f"[target_resolver] {source_name}: no results for {chembl_id}"
                    )

            except Exception as e:
                logger.warning(
                    f"[target_resolver] {source_name} raised exception "
                    f"for {chembl_id}: {e}"
                )

        if self.require_human and all_ids:
            all_ids = self._filter_human_proteins(all_ids)

        logger.info(
            f"[target_resolver] FINAL: {chembl_id} ({drug_name}) → "
            f"{len(all_ids)} UniProt IDs from {sources_with_results} sources: "
            f"{sorted(all_ids)}"
        )

        return all_ids

    def get_target_details(
        self,
        chembl_id: str,
        drug_name: str = "",
    ) -> list[dict]:
        """
        Same as get_uniprot_ids but returns full target details including
        gene symbol, protein name, and which source provided each target.

        Useful for the Layer 1A target overlap report and the candidate report.
        """
        all_targets: dict[str, dict] = {}

        sources = [
            ("OpenTargets",      lambda: _fetch_targets_opentargets(chembl_id)),
            ("ChEMBL_mechanism", lambda: _fetch_targets_chembl_mechanism(chembl_id)),
            ("ChEMBL_activity",  lambda: _fetch_targets_chembl_activity(chembl_id)),
            ("UniProt",          lambda: _fetch_targets_uniprot_cross_ref(chembl_id)),
            ("GtoPdb",           lambda: _fetch_targets_gtopdb(drug_name) if drug_name else set()),
        ]

        for source_name, fetch_fn in sources:
            try:
                ids = fetch_fn()
                for uid in ids:
                    if uid not in all_targets:
                        all_targets[uid] = {
                            "uniprot_id": uid,
                            "sources": [source_name],
                        }
                    else:
                        all_targets[uid]["sources"].append(source_name)
            except Exception as e:
                logger.debug(f"[target_resolver] {source_name} failed: {e}")

        # Enrich with UniProt gene name and protein name
        enriched = self._enrich_with_uniprot(list(all_targets.values()))

        if self.require_human:
            enriched = [t for t in enriched if t.get("organism") == "Homo sapiens"]

        return enriched

    @cached_api_call(ttl_seconds=86400 * 90)
    def _filter_human_proteins(self, uniprot_ids: frozenset) -> set[str]:
        """
        Filter a set of UniProt IDs to human proteins only.
        Uses UniProt batch lookup. Cached 90 days.
        """
        # Convert frozenset to set for input, cast back for caching compatibility
        ids = set(uniprot_ids)
        if not ids:
            return set()

        # UniProt batch query — up to 100 IDs per call
        id_list = list(ids)[:100]
        query = " OR ".join(f"accession:{uid}" for uid in id_list)

        try:
            r = requests.get(
                f"{UNIPROT_API}/search",
                params={
                    "query": f"({query}) AND (organism_id:9606)",
                    "fields": "accession",
                    "format": "json",
                    "size": 100,
                },
                timeout=20,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            human_ids = {
                e["primaryAccession"] for e in results
                if e.get("primaryAccession")
            }
            removed = ids - human_ids
            if removed:
                logger.debug(
                    f"[target_resolver] Filtered out non-human proteins: {removed}"
                )
            return human_ids

        except Exception as e:
            logger.debug(f"[target_resolver] Human protein filter failed: {e}")
            return ids   # return unfiltered on error

    def _enrich_with_uniprot(self, targets: list[dict]) -> list[dict]:
        """Add gene symbol and protein name from UniProt for each target."""
        if not targets:
            return targets

        ids = [t["uniprot_id"] for t in targets]
        query = " OR ".join(f"accession:{uid}" for uid in ids[:50])

        try:
            r = requests.get(
                f"{UNIPROT_API}/search",
                params={
                    "query": f"({query}) AND (organism_id:9606)",
                    "fields": "accession,gene_names,protein_name,organism_name",
                    "format": "json",
                    "size": 50,
                },
                timeout=20,
            )
            r.raise_for_status()
            results = r.json().get("results", [])

            enrichment = {}
            for entry in results:
                acc = entry.get("primaryAccession", "")
                gene_names = entry.get("genes", [])
                gene_symbol = (
                    gene_names[0].get("geneName", {}).get("value", "")
                    if gene_names else ""
                )
                protein_desc = (
                    entry.get("proteinDescription", {})
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value", "")
                )
                enrichment[acc] = {
                    "gene_symbol": gene_symbol,
                    "protein_name": protein_desc,
                    "organism": entry.get("organism", {}).get("scientificName", ""),
                }

            for target in targets:
                uid = target["uniprot_id"]
                if uid in enrichment:
                    target.update(enrichment[uid])

        except Exception as e:
            logger.debug(f"[target_resolver] UniProt enrichment failed: {e}")

        return targets


# ── Convenience function for drop-in use ──────────────────────────────────────

_default_resolver = None


def get_drug_targets(chembl_id: str, drug_name: str = "") -> set[str]:
    """
    Module-level convenience function.
    Drop-in replacement for ChEMBLClient.get_target_uniprot_ids().

    Creates a shared resolver instance (stop_early_after=2 for speed —
    change to 5 to always try all sources).
    """
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = DrugTargetResolver(stop_early_after=2)
    return _default_resolver.get_uniprot_ids(chembl_id, drug_name)