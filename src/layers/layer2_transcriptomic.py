"""
src/layers/layer2_transcriptomic.py

Layer 2 — Transcriptomic Signature Reversal.

Fixes applied:
  - DISEASE_GEO_DATASETS no longer hardcoded. Loaded dynamically from
    config/geo_datasets.json (same config used by compute_geo_signatures.py).
  - get_disease_signature_from_file now globs for {safe_id}_*.json so it
    finds files written by compute_geo_signatures.py (which names them
    {disease_id}_{geo_id}.json), instead of looking for a non-existent
    {disease_id}.json.
  - When multiple GEO files exist for a disease, their log2FC values are
    averaged across shared genes before computing the KS score.
  - GEOClient.get_disease_signature stub now logs clearly that it is a stub
    and that real data comes from pre-computed files.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

ILINCS_BASE = "https://www.ilincs.org/ilincs/api"
GEO_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEO_CONFIG_PATH = PROJECT_ROOT / "config" / "geo_datasets.json"
GEO_SIGNATURES_DIR = PROJECT_ROOT / "data" / "processed" / "geo_signatures"

# Threshold for flagging strong reversal signal
KS_STRONG_REVERSAL_THRESHOLD = -0.3


def _load_geo_config() -> dict:
    """Load GEO dataset configuration from config/geo_datasets.json."""
    if not GEO_CONFIG_PATH.exists():
        logger.warning(
            f"GEO config not found at {GEO_CONFIG_PATH}. "
            "No transcriptomic scoring will occur. "
            "Run: python data/scripts/compute_geo_signatures.py --list"
        )
        return {"datasets": []}
    with open(GEO_CONFIG_PATH) as f:
        return json.load(f)


def _build_disease_geo_map() -> dict[str, list[str]]:
    """
    Build {disease_id: [GSExxxxx, ...]} from config/geo_datasets.json.
    This replaces the old hardcoded DISEASE_GEO_DATASETS dict.
    """
    cfg = _load_geo_config()
    mapping: dict[str, list[str]] = {}
    for dataset in cfg.get("datasets", []):
        disease_id = dataset.get("disease_id", "")
        geo_id = dataset.get("geo_id", "")
        if disease_id and geo_id:
            mapping.setdefault(disease_id, []).append(geo_id)
    return mapping


class iLINCSClient:
    """
    Query iLINCS (integrative LINCS) for drug-induced gene expression signatures.
    """

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_drug_signature(self, drug_name: str) -> Optional[dict[str, float]]:
        """
        Fetch drug-induced gene expression signature from iLINCS.

        Returns:
            Dict mapping gene_symbol → log2 fold change
            None if signature not found.
        """
        search_url = f"{ILINCS_BASE}/SignatureMeta/findSignatures"
        params = {
            "search": drug_name,
            "sigType": "CMap_chemical",
            "limit": 5,
        }
        try:
            r = requests.get(search_url, params=params, timeout=20)
            r.raise_for_status()
            results = r.json()

            if not results:
                logger.warning(f"iLINCS: no signature found for {drug_name}")
                return None

            sig_id = results[0].get("signatureID") or results[0].get("id")
            if not sig_id:
                return None

            return self._fetch_signature_genes(sig_id)
        except Exception as e:
            logger.error(f"iLINCS signature fetch failed for {drug_name}: {e}")
            return None

    @cached_api_call(ttl_seconds=86400 * 30)
    def _fetch_signature_genes(self, sig_id: str) -> Optional[dict[str, float]]:
        url = f"{ILINCS_BASE}/SignatureMeta/{sig_id}/genes"
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            genes = r.json()
            return {
                g["name"]: float(g.get("value") or g.get("logFC") or 0)
                for g in genes
                if g.get("name")
            }
        except Exception as e:
            logger.debug(f"_fetch_signature_genes({sig_id}) failed: {e}")
            return None


class GEOClient:
    """
    Loads pre-computed disease gene expression signatures from disk.

    Files are produced by data/scripts/compute_geo_signatures.py and named:
        {disease_id_safe}_{geo_id}.json
    e.g.  ORPHA_422_GSE113439.json

    When multiple files exist for the same disease, their log2FC values
    are averaged across shared genes to produce a consensus signature.
    """

    def get_disease_signature_from_file(
        self, disease_id: str
    ) -> Optional[dict[str, float]]:
        """
        Load pre-computed disease signature(s) from disk.

        FIX: Previously looked for {disease_id}.json which never existed
        because compute_geo_signatures.py writes {disease_id}_{geo_id}.json.
        Now globs for all matching files and averages the log2FC values.
        """
        safe_id = disease_id.replace(":", "_").replace("/", "_")
        pattern = str(GEO_SIGNATURES_DIR / f"{safe_id}_*.json")
        matching_files = glob.glob(pattern)

        if not matching_files:
            logger.warning(
                f"No pre-computed GEO signature for {disease_id}. "
                f"Expected files matching: {pattern}. "
                f"Run: python data/scripts/compute_geo_signatures.py {disease_id}"
            )
            return None

        # Collect log2FC values from all matching files, average across GSEs
        gene_values: dict[str, list[float]] = {}
        loaded_count = 0

        for fpath in matching_files:
            try:
                with open(fpath) as f:
                    data = json.load(f)

                # compute_geo_signatures.py stores full_results as list of
                # {gene, log2fc, pvalue, fdr} dicts
                full_results = data.get("full_results", [])
                if full_results:
                    for entry in full_results:
                        gene = entry.get("gene", "")
                        log2fc = entry.get("log2fc")
                        if gene and log2fc is not None:
                            gene_values.setdefault(gene, []).append(float(log2fc))
                    loaded_count += 1
                    logger.info(
                        f"Loaded GEO signature from {os.path.basename(fpath)}: "
                        f"{len(full_results)} genes"
                    )
                else:
                    # Fallback: file might just be {gene: log2fc}
                    for gene, val in data.items():
                        if isinstance(val, (int, float)):
                            gene_values.setdefault(gene, []).append(float(val))
                    if gene_values:
                        loaded_count += 1

            except Exception as e:
                logger.warning(f"Failed to load GEO signature {fpath}: {e}")

        if not gene_values:
            return None

        # Average log2FC across all loaded GSE datasets
        merged = {gene: float(np.mean(vals)) for gene, vals in gene_values.items()}
        logger.info(
            f"GEO signature for {disease_id}: {len(merged)} genes "
            f"averaged from {loaded_count} dataset(s)"
        )
        return merged

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_disease_signature(self, geo_series_id: str) -> Optional[dict[str, float]]:
        """
        Stub: checks if a GEO dataset exists in NCBI.
        Real data comes from pre-computed files (get_disease_signature_from_file).
        Full DEG computation requires GEOparse — run compute_geo_signatures.py.
        """
        url = f"{GEO_BASE}/esearch.fcgi"
        params = {"db": "gds", "term": geo_series_id, "retmode": "json"}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            count = int(r.json().get("esearchresult", {}).get("count", 0))
            if count > 0:
                logger.info(
                    f"GEO dataset {geo_series_id} exists. "
                    f"Run compute_geo_signatures.py to pre-compute DEG results."
                )
        except Exception as e:
            logger.debug(f"GEO check failed for {geo_series_id}: {e}")
        return None


def compute_ks_connectivity_score(
    disease_signature: dict[str, float],
    drug_signature: dict[str, float],
    top_n: int = 150,
) -> Optional[float]:
    """
    Compute KS connectivity score between disease and drug expression signatures.

    Returns float in [-1, 1]. Strongly negative = drug reverses disease signature.
    """
    if not disease_signature or not drug_signature:
        return None

    sorted_disease = sorted(disease_signature.items(), key=lambda x: x[1], reverse=True)
    disease_up = {g for g, _ in sorted_disease[:top_n]}
    disease_down = {g for g, _ in sorted_disease[-top_n:]}

    drug_genes = list(drug_signature.keys())
    drug_ranked = sorted(drug_genes, key=lambda g: drug_signature[g], reverse=True)
    n = len(drug_ranked)

    if n < 10:
        logger.warning("Drug signature too small for KS scoring (< 10 genes)")
        return None

    rank_lookup = {gene: rank for rank, gene in enumerate(drug_ranked)}

    def ks_score(query_set: set[str]) -> float:
        query_in_drug = [g for g in query_set if g in rank_lookup]
        if not query_in_drug:
            return 0.0
        positions = sorted(rank_lookup[g] for g in query_in_drug)
        m = len(positions)
        ks_max, ks_min = 0.0, 0.0
        running_sum = 0.0
        prev_pos = 0
        for rank_pos in positions:
            running_sum -= (rank_pos - prev_pos) / (n - m)
            ks_min = min(ks_min, running_sum)
            running_sum += 1.0 / m
            ks_max = max(ks_max, running_sum)
            prev_pos = rank_pos
        return ks_max if abs(ks_max) > abs(ks_min) else ks_min

    ks_up = ks_score(disease_up)
    ks_down = ks_score(disease_down)

    if (ks_up > 0 and ks_down < 0) or (ks_up < 0 and ks_down > 0):
        connectivity = (ks_up - ks_down) / 2
    else:
        connectivity = 0.0

    return float(connectivity)


class TranscriptomicLayer(BaseLayer):
    """
    Layer 2 — Transcriptomic signature reversal.

    Disease GEO datasets are loaded from config/geo_datasets.json (not hardcoded).
    Signatures are read from data/processed/geo_signatures/{disease_id}_{geo_id}.json
    files produced by data/scripts/compute_geo_signatures.py.

    Scores:
        pair.scores.transcriptomic_reversal_ks
    """

    layer_name = "layer2_transcriptomic"
    version = "1.1"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.ilincs = iLINCSClient()
        self.geo = GEOClient()
        # Loaded lazily so config file changes are picked up at runtime
        self._disease_geo_map: Optional[dict] = None

    @property
    def disease_geo_map(self) -> dict[str, list[str]]:
        if self._disease_geo_map is None:
            self._disease_geo_map = _build_disease_geo_map()
            logger.info(
                f"[{self.layer_name}] Loaded GEO config: "
                f"{len(self._disease_geo_map)} diseases configured"
            )
        return self._disease_geo_map

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Get drug expression signature ──────────────────────────────
        drug_sig = self.ilincs.get_drug_signature(pair.drug_name)
        if not drug_sig:
            logger.warning(
                f"[{self.layer_name}] No LINCS signature for {pair.drug_name}."
            )
            return pair

        # ── 2. Get disease expression signature ───────────────────────────
        # Primary: pre-computed files (glob for {disease_id}_{geo_id}.json)
        disease_sig = self.geo.get_disease_signature_from_file(pair.disease_id)

        if not disease_sig:
            # Secondary: try GEO API stub (won't return real data,
            # but confirms dataset existence for manual follow-up)
            geo_ids = self.disease_geo_map.get(pair.disease_id, [])
            if geo_ids:
                logger.info(
                    f"[{self.layer_name}] Pre-computed signature missing for "
                    f"{pair.disease_id}. Configured GEO datasets: {geo_ids}. "
                    f"Run: python data/scripts/compute_geo_signatures.py"
                )
            else:
                logger.warning(
                    f"[{self.layer_name}] No GEO dataset configured for "
                    f"{pair.disease_id}. Add it via: "
                    f"python data/scripts/compute_geo_signatures.py discover "
                    f'"{pair.disease_name}" {pair.disease_id}'
                )
            return pair

        # ── 3. Compute KS connectivity score ─────────────────────────────
        ks = compute_ks_connectivity_score(disease_sig, drug_sig)
        pair.scores.transcriptomic_reversal_ks = ks

        if ks is not None:
            signal = (
                "STRONG REVERSAL" if ks < -0.5
                else "REVERSAL" if ks < KS_STRONG_REVERSAL_THRESHOLD
                else "WEAK" if ks < 0
                else "NO REVERSAL"
            )
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"KS={ks:.4f} [{signal}]"
            )

        return pair