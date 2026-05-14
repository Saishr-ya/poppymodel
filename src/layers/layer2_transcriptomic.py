"""
src/layers/layer2_transcriptomic.py

Layer 2 — Transcriptomic Signature Reversal.

Core idea (Sirota et al. 2011, PMID 22116928):
  A disease creates a specific gene expression signature — some genes up-regulated,
  others down-regulated. A drug that REVERSES that signature is a repurposing candidate.

Method:
  1. Pull disease gene expression from GEO (diseased vs healthy tissue)
  2. Pull drug-induced expression from LINCS L1000 (thousands of compounds)
  3. Compute connectivity score using Kolmogorov-Smirnov (KS) statistic
     - Strongly NEGATIVE score = drug reverses disease → strong repurposing signal
     - Score near zero = no relationship
     - Strongly POSITIVE = drug makes things worse

Data sources:
  - iLINCS API: https://www.ilincs.org/ilincs/api  (drug expression profiles)
  - GEO API: https://www.ncbi.nlm.nih.gov/geo/  (disease expression signatures)
  - NCBI Datasets: https://api.ncbi.nlm.nih.gov/

Used by: BenevolentAI, Recursion, and the Broad Institute (CMap project).

Bio team notes:
  - KS score < -0.3 is a meaningful reversal signal
  - KS score < -0.5 is strong
  - This layer is highly complementary to Layer 1 — a drug can have no target
    overlap but still show strong transcriptomic reversal via downstream effects
  - Validation: sildenafil × PAH should show strong negative KS score
  - GEO dataset selection is critical — bio person must curate the right datasets
    per disease (diseased tissue, same cell type, comparable conditions)
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import numpy as np
import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

ILINCS_BASE = "https://www.ilincs.org/ilincs/api"
GEO_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Manually curated disease → GEO dataset mappings.
# Bio team: add disease-specific GEO datasets here.
# Format: "ORPHA:xxx" or "OMIM:xxx" → list of GEO series IDs (GSExxxxx)
# These must be datasets comparing diseased vs healthy tissue of the same type.
DISEASE_GEO_DATASETS: dict[str, list[str]] = {
    "ORPHA:422": ["GSE113439", "GSE15197"],    # Pulmonary arterial hypertension
    "ORPHA:77":  ["GSE43955"],                  # Gaucher disease type 1
    "ORPHA:33069": ["GSE82109"],               # Dravet syndrome (SCN1A)
    # Add more disease-GSE mappings as you expand your disease list
}

# Threshold for flagging strong reversal signal
KS_STRONG_REVERSAL_THRESHOLD = -0.3


class iLINCSClient:
    """
    Query iLINCS (integrative LINCS) for drug-induced gene expression signatures.
    iLINCS provides the L1000 signatures from the LINCS project.

    L1000 measures ~1000 "landmark" genes that can infer full transcriptome.
    """

    @cached_api_call(ttl_seconds=86400 * 30)
    def get_drug_signature(self, drug_name: str) -> Optional[dict[str, float]]:
        """
        Fetch drug-induced gene expression signature from iLINCS.

        Returns:
            Dict mapping gene_symbol → log2 fold change (positive = up-regulated by drug)
            None if signature not found.
        """
        # Step 1: search for the compound
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

            # Take first (best match) signature
            sig_id = results[0].get("signatureID") or results[0].get("id")
            if not sig_id:
                return None

            return self._fetch_signature_genes(sig_id)
        except Exception as e:
            logger.error(f"iLINCS signature fetch failed for {drug_name}: {e}")
            return None

    @cached_api_call(ttl_seconds=86400 * 30)
    def _fetch_signature_genes(self, sig_id: str) -> Optional[dict[str, float]]:
        """Fetch the gene-level expression values for a specific signature ID."""
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
    Query NCBI GEO for disease gene expression signatures.
    Returns differential expression (disease vs healthy).
    """

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_disease_signature(self, geo_series_id: str) -> Optional[dict[str, float]]:
        """
        Fetch a disease gene expression signature from a GEO series.

        In production, this uses GEO2R or pre-computed DEG results.
        Returns dict: gene_symbol → log2FC (positive = up in disease).

        Note: Full GEO parsing requires downloading large files.
        This implementation uses the GEO DataSets API for summary data,
        with a pointer to the full implementation needed for production.
        """
        # Real implementation: use GEOparse library to download and parse the series
        # pip install GEOparse
        # Then: gse = GEOparse.get_GEO(geo=geo_series_id, destdir="data/raw/geo/")
        # Then run DEG analysis using scipy.stats.ttest_ind per gene

        # For now: try GEO API to check if dataset exists
        url = f"{GEO_BASE}/esearch.fcgi"
        params = {
            "db": "gds",
            "term": geo_series_id,
            "retmode": "json",
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            count = int(r.json().get("esearchresult", {}).get("count", 0))
            if count > 0:
                logger.info(f"GEO dataset {geo_series_id} exists. "
                            f"Install GEOparse and implement full DEG analysis.")
            return None   # Stub — implement with GEOparse in production
        except Exception as e:
            logger.debug(f"GEO check failed for {geo_series_id}: {e}")
            return None

    def get_disease_signature_from_file(
        self, disease_id: str
    ) -> Optional[dict[str, float]]:
        """
        Load a pre-computed disease signature from disk.
        Bio team should pre-compute DEG results and store as JSON files.

        Expected file location:
            data/processed/geo_signatures/{disease_id.replace(':', '_')}.json
        Format:
            {"GENE1": 2.3, "GENE2": -1.8, ...}  (log2FC values)
        """
        safe_id = disease_id.replace(":", "_").replace("/", "_")
        path = f"data/processed/geo_signatures/{safe_id}.json"

        if os.path.exists(path):
            import json
            with open(path) as f:
                data = json.load(f)
            logger.info(f"Loaded pre-computed GEO signature for {disease_id}: "
                        f"{len(data)} genes")
            return data

        logger.warning(
            f"No pre-computed GEO signature for {disease_id}. "
            f"Expected at: {path}. "
            f"Use GEOparse to compute DEG from raw GEO data and save there. "
            f"See: data/scripts/compute_geo_signatures.py"
        )
        return None


def compute_ks_connectivity_score(
    disease_signature: dict[str, float],
    drug_signature: dict[str, float],
    top_n: int = 150,
) -> Optional[float]:
    """
    Compute connectivity score between disease and drug expression signatures.
    Uses the Kolmogorov-Smirnov (KS) statistic — same method as Broad Institute CMap.

    Args:
        disease_signature: gene_symbol → log2FC in disease (positive = up in disease)
        drug_signature:    gene_symbol → log2FC induced by drug (positive = up-regulated)
        top_n:             Number of top/bottom genes to use from each signature.

    Returns:
        KS connectivity score in range [-1, 1].
        Strongly negative = drug reverses disease signature → repurposing signal.
        Near zero = no relationship.
        Strongly positive = drug amplifies disease signature → bad.
        None if insufficient overlapping genes.

    Reference: Lamb et al. 2006 Science (CMap); Subramanian et al. 2005 PNAS (GSEA).
    """
    if not disease_signature or not drug_signature:
        return None

    # Get top N up-regulated and bottom N down-regulated genes in disease
    sorted_disease = sorted(disease_signature.items(), key=lambda x: x[1], reverse=True)
    disease_up = {g for g, _ in sorted_disease[:top_n]}
    disease_down = {g for g, _ in sorted_disease[-top_n:]}

    # Rank drug signature genes (highest = most up-regulated by drug)
    drug_genes = list(drug_signature.keys())
    drug_ranked = sorted(drug_genes, key=lambda g: drug_signature[g], reverse=True)
    n = len(drug_ranked)

    if n < 10:
        logger.warning("Drug signature too small for KS scoring (< 10 genes)")
        return None

    rank_lookup = {gene: rank for rank, gene in enumerate(drug_ranked)}

    # KS statistic for disease-upregulated genes vs drug ranking
    def ks_score(query_set: set[str]) -> float:
        """Run KS test of query genes against ranked drug signature."""
        query_in_drug = [g for g in query_set if g in rank_lookup]
        if not query_in_drug:
            return 0.0

        # Positions of query genes in drug ranking (0 = most up-regulated)
        positions = sorted(rank_lookup[g] for g in query_in_drug)
        m = len(positions)

        # KS running sum
        ks_max, ks_min = 0.0, 0.0
        running_sum = 0.0
        prev_pos = 0

        for rank_pos in positions:
            running_sum -= (rank_pos - prev_pos) / (n - m)   # penalty for gap
            ks_min = min(ks_min, running_sum)
            running_sum += 1.0 / m                            # reward for hit
            ks_max = max(ks_max, running_sum)
            prev_pos = rank_pos

        # Final KS: use the extreme value (positive or negative)
        if abs(ks_max) > abs(ks_min):
            return ks_max
        return ks_min

    # Connectivity score:
    # If disease-UP genes rank LOW in drug → drug REVERSES up-regulation → negative score
    # If disease-DOWN genes rank HIGH in drug → drug REVERSES down-regulation → negative score
    ks_up = ks_score(disease_up)
    ks_down = ks_score(disease_down)

    # Combined connectivity score (same sign = no reversal, opposite sign = reversal)
    if (ks_up > 0 and ks_down < 0) or (ks_up < 0 and ks_down > 0):
        connectivity = (ks_up - ks_down) / 2   # opposite = reversal → negative result
    else:
        connectivity = 0.0   # same direction = no clear reversal

    return float(connectivity)


class TranscriptomicLayer(BaseLayer):
    """
    Layer 2 — Transcriptomic signature reversal.

    Scores:
        pair.scores.transcriptomic_reversal_ks  (KS statistic, negative = reversal)

    Validation:
        sildenafil × PAH should score < -0.3 (known reversal)
        imatinib × microcephaly should score near 0 (no relationship)

    Production setup needed:
        1. Install GEOparse: pip install GEOparse
        2. Run: python data/scripts/compute_geo_signatures.py
        3. This populates data/processed/geo_signatures/{disease_id}.json
        4. The layer then loads these pre-computed signatures
    """

    layer_name = "layer2_transcriptomic"
    version = "1.0"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.ilincs = iLINCSClient()
        self.geo = GEOClient()

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Get drug expression signature ─────────────────────────────
        drug_sig = self.ilincs.get_drug_signature(pair.drug_name)
        if not drug_sig:
            logger.warning(
                f"[{self.layer_name}] No LINCS signature for {pair.drug_name}. "
                f"Try alternative names or synonyms."
            )
            return pair

        # ── 2. Get disease expression signature ───────────────────────────
        # First try pre-computed file, then GEO API
        disease_sig = self.geo.get_disease_signature_from_file(pair.disease_id)

        if not disease_sig:
            # Try GEO datasets mapped to this disease
            geo_ids = DISEASE_GEO_DATASETS.get(pair.disease_id, [])
            for geo_id in geo_ids:
                disease_sig = self.geo.get_disease_signature(geo_id)
                if disease_sig:
                    break

        if not disease_sig:
            logger.warning(
                f"[{self.layer_name}] No GEO signature for {pair.disease_id}. "
                f"Add GSE dataset mapping to DISEASE_GEO_DATASETS or "
                f"pre-compute to data/processed/geo_signatures/"
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