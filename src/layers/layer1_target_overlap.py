"""
src/layers/layer1_target_overlap.py

Layer 1A — Direct target and gene overlap.

Implements:
  - Jaccard similarity between drug protein targets and disease causal genes
  - Shared pathway enrichment (hypergeometric p-value)

Based on: Gottlieb et al. 2011 (PREDICT method), PMID 21915133.

Bio team notes:
  - Jaccard = |intersection| / |union| of UniProt IDs
  - A Jaccard of 0.0 doesn't mean the drug is irrelevant — it may still score
    well on network proximity (Layer 1B) if targets are biologically nearby.
  - Score threshold for "signal": Jaccard ≥ 0.05 is meaningful for rare diseases
    with small gene sets; don't over-filter on this alone.
"""

from __future__ import annotations
import logging
from typing import Optional

from scipy import stats

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.chembl_client import ChEMBLClient
from src.ingestion.disgenet_client import DisGeNETClient
from src.ingestion.drug_target_resolver import get_drug_targets


logger = logging.getLogger(__name__)

# Reactome and KEGG pathway mappings (gene → pathway set)
# In production, load these from the Reactome API or pre-built Parquet files.
# Here we define the interface and load logic.
_PATHWAY_GENE_MAP: Optional[dict[str, set[str]]] = None


def _load_pathway_gene_map() -> dict[str, set[str]]:
    """
    Load gene → pathway mapping from disk (Reactome export).
    Returns dict: { gene_symbol: {pathway_id, ...} }

    In production, populate from:
        data/processed/reactome_gene_pathway.parquet
    """
    global _PATHWAY_GENE_MAP
    if _PATHWAY_GENE_MAP is not None:
        return _PATHWAY_GENE_MAP

    try:
        import pandas as pd
        import os
        path = "data/processed/reactome_gene_pathway.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            mapping = {}
            for _, row in df.iterrows():
                gene = row["gene_symbol"]
                if gene not in mapping:
                    mapping[gene] = set()
                mapping[gene].add(row["pathway_id"])
            _PATHWAY_GENE_MAP = mapping
            logger.info(f"Loaded pathway map: {len(mapping)} genes")
        else:
            logger.warning(
                "Reactome pathway map not found at data/processed/reactome_gene_pathway.parquet. "
                "Pathway enrichment scoring will be skipped. "
                "Download from: https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt"
            )
            _PATHWAY_GENE_MAP = {}
    except Exception as e:
        logger.error(f"Failed to load pathway map: {e}")
        _PATHWAY_GENE_MAP = {}

    return _PATHWAY_GENE_MAP


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Standard Jaccard index. Returns 0.0 if both sets are empty."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def pathway_overlap_pvalue(
    drug_gene_symbols: set[str],
    disease_gene_symbols: set[str],
    pathway_map: dict[str, set[str]],
    total_genes: int = 20_000,
) -> Optional[float]:
    """
    Hypergeometric test for shared pathway enrichment.

    Tests whether drug targets and disease genes share more pathways than
    expected by chance given the total human gene count.

    Args:
        drug_gene_symbols:    Gene symbols for drug targets.
        disease_gene_symbols: Gene symbols for disease causal genes.
        pathway_map:          Gene → set of Reactome pathway IDs.
        total_genes:          Background gene count (default: ~20K human genes).

    Returns:
        p-value (float) or None if insufficient data.
    """
    if not drug_gene_symbols or not disease_gene_symbols:
        return None

    # Flatten to pathway sets
    drug_pathways = set()
    for g in drug_gene_symbols:
        drug_pathways |= pathway_map.get(g, set())

    disease_pathways = set()
    for g in disease_gene_symbols:
        disease_pathways |= pathway_map.get(g, set())

    if not drug_pathways or not disease_pathways:
        return None

    shared = drug_pathways & disease_pathways
    all_pathways = drug_pathways | disease_pathways

    # Hypergeometric test:
    # M = total pathways, n = drug pathways, N = disease pathways, k = shared
    M = len(all_pathways)
    n = len(drug_pathways)
    N = len(disease_pathways)
    k = len(shared)

    if M == 0 or k == 0:
        return 1.0  # No enrichment

    pval = stats.hypergeom.sf(k - 1, M, n, N)
    return float(pval)


class TargetOverlapLayer(BaseLayer):
    """
    Layer 1A — Direct target overlap between drug targets and disease genes.

    Scores:
        pair.scores.target_overlap_jaccard
        pair.scores.pathway_enrichment_pvalue

    Flags:
        None directly — but near-zero Jaccard should be noted by bio reviewer.
    """

    layer_name = "layer1_target_overlap"
    version = "1.0"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.chembl = ChEMBLClient()
        self.disgenet = DisGeNETClient(
            api_key=config.get("disgenet_api_key") if config else None
        )

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── Fetch drug target UniProt IDs ──────────────────────────────────
        drug_targets = get_drug_targets(pair.drug_id, pair.drug_name)
        if not drug_targets:
            logger.warning(
                f"[{self.layer_name}] No targets found for {pair.drug_id} "
                f"({pair.drug_name}). Check ChEMBL mechanism data."
            )

        # ── Fetch disease causal gene UniProt IDs ─────────────────────────
        disease_genes = self.disgenet.get_disease_uniprot_ids(pair.disease_id)
        if not disease_genes:
            logger.warning(
                f"[{self.layer_name}] No causal genes found for {pair.disease_id} "
                f"({pair.disease_name}). Check DisGeNET coverage."
            )

        # ── Jaccard similarity ─────────────────────────────────────────────
        jaccard = jaccard_similarity(drug_targets, disease_genes)
        pair.scores.target_overlap_jaccard = jaccard

        logger.debug(
            f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
            f"targets={len(drug_targets)}, disease_genes={len(disease_genes)}, "
            f"Jaccard={jaccard:.4f}"
        )

        # ── Pathway enrichment ─────────────────────────────────────────────
        # Note: pathway map uses gene symbols; we need to re-fetch those
        # In production, DisGeNET returns gene_symbol alongside UniProt IDs
        drug_gene_symbols = self._get_drug_gene_symbols(pair.drug_id)
        disease_gene_symbols = self._get_disease_gene_symbols(pair.disease_id)

        if drug_gene_symbols and disease_gene_symbols:
            pathway_map = _load_pathway_gene_map()
            pval = pathway_overlap_pvalue(
                drug_gene_symbols, disease_gene_symbols, pathway_map
            )
            pair.scores.pathway_enrichment_pvalue = pval
            if pval is not None:
                logger.debug(
                    f"[{self.layer_name}] Pathway enrichment p={pval:.4e}"
                )

        return pair

    def _get_drug_gene_symbols(self, drug_id: str) -> set[str]:
        """Get gene symbols (not UniProt) for drug targets."""
        targets = self.chembl.get_drug_targets(drug_id)
        symbols = set()
        for t in targets:
            name = t.get("target_name", "")
            # ChEMBL target names often include gene symbol — simplified extraction
            # In production, cross-reference via UniProt → gene symbol mapping
            if name:
                symbols.add(name.split()[0].upper())
        return symbols

    def _get_disease_gene_symbols(self, disease_id: str) -> set[str]:
        """Get gene symbols for disease causal genes."""
        genes = self.disgenet.get_disease_genes(disease_id)
        return {g["gene_symbol"] for g in genes if g.get("gene_symbol")}
