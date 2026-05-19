"""
data/scripts/compute_geo_signatures.py

Pre-compute disease gene expression signatures from GEO datasets.
Run this ONCE per disease before using Layer 2 (transcriptomics).

Output: data/processed/geo_signatures/{disease_id}.json
Format: {"GENE1": 2.3, "GENE2": -1.8, ...}  (log2 fold change, disease vs healthy)

Refactored to use a format-agnostic matrix pivot approach and included an
automated platform type router to flag and separate RNA-Seq data.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Curated disease → GEO dataset mappings ────────────────────────────────────
DISEASE_GEO_CONFIG = [
    {
        "disease_id":      "ORPHA:422",
        "disease_name":    "Pulmonary arterial hypertension",
        "geo_id":          "GSE113439",
        "disease_keyword": "PAH",
        "control_keyword": "control",
        "tissue":          "lung",
    },
    {
        "disease_id":      "ORPHA:77",
        "disease_name":    "Gaucher disease type 1",
        "geo_id":          "GSE13675",
        "disease_keyword": "CBE",       
        "control_keyword": "ctrl",      
        "tissue":          "macrophage",
    },
    {
        "disease_id":      "ORPHA:33069",
        "disease_name":    "Dravet syndrome",
        "geo_id":          "GSE143272",   
        "disease_keyword": "responder",   
        "control_keyword": "healthy",     
        "tissue":          "blood",
    },
    {
        "disease_id":      "ORPHA:566",
        "disease_name":    "Pompe disease",
        "geo_id":          "GSE38680",
        "disease_keyword": "Pompe",
        "control_keyword": "Control",
        "tissue":          "muscle",
    },
]

OUTPUT_DIR = "data/processed/geo_signatures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _get_sample_meta(gse, gsm_id: str) -> str:
    """Helper function to cleanly extract and combine title and characteristics for a GSM sample."""
    gsm = gse.gsms.get(gsm_id)
    if not gsm:
        return ""
    title = gsm.metadata.get("title", [""])[0].lower()
    characteristics = " ".join(
        str(c) for cs in gsm.metadata.get("characteristics_ch1", []) for c in [cs]
    ).lower()
    return f"{title} {characteristics}"


def compute_signature_from_geo(
    geo_id: str,
    disease_keyword: str,
    control_keyword: str,
    min_samples: int = 3,
    top_n_genes: int = 500,
    destdir: str = "data/raw/geo",
) -> Optional[dict[str, float]]:
    """
    Download GEO series and compute differential expression signature.
    Uses format-agnostic matrix pivots for microarrays and routes sequencing tracks away safely.
    """
    try:
        import GEOparse
        import pandas as pd
    except ImportError as e:
        logger.error(f"Required package missing: {e}")
        return None

    os.makedirs(destdir, exist_ok=True)
    logger.info(f"Downloading/loading GEO series {geo_id}...")

    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=destdir, silent=True)
    except Exception as e:
        logger.error(f"Failed to download {geo_id}: {e}")
        return None

    # ── 1. Platform Ingestion & Technology Type Routing ───────────────────────
    try:
        gpl_id = list(gse.gpls.keys())[0]
        platform_type = gse.gpls[gpl_id].metadata.get("technology", [""])[0]
        
        if "sequencing" in platform_type.lower() or "rna-seq" in platform_type.lower():
            logger.warning(f"⚠️ {geo_id} is RNA-Seq — supplementary counts matrix download required. Handling separately.")
            return None
            
        # Microarray Path: Pull pre-compiled expression grid using optimized pivot call
        full_matrix = gse.pivot_samples("VALUE")
        
    except Exception as e:
        logger.error(f"Failed parsing platform layout for {geo_id}: {e}")
        return None

    # ── 2. Vectorized Condition Sorting via Meta Matching ────────────────────
    disease_cols = [c for c in full_matrix.columns if disease_keyword.lower() in _get_sample_meta(gse, c)]
    control_cols = [c for c in full_matrix.columns if control_keyword.lower() in _get_sample_meta(gse, c)]

    logger.info(f"{geo_id}: {len(disease_cols)} disease, {len(control_cols)} control samples verified via matrix pivot.")

    if len(disease_cols) < min_samples or len(control_cols) < min_samples:
        logger.warning(
            f"Insufficient cohorts matched inside {geo_id} for keywords: "
            f"disease='{disease_keyword}', control='{control_keyword}'."
        )
        return None

    # Slice expression profiles out of our comprehensive data frame
    disease_expr = full_matrix[disease_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    control_expr = full_matrix[control_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if disease_expr.empty or control_expr.empty:
        logger.error(f"Resulting data slices were completely empty for matrix calculation on {geo_id}")
        return None

    # ── 3. Map probe IDs to gene symbols ──────────────────────────────────────
    if hasattr(gse, "gpls") and gse.gpls:
        probe_map = _build_probe_to_gene_map(gse.gpls[gpl_id].table)
        disease_expr.index = disease_expr.index.map(lambda x: probe_map.get(str(x), str(x)))
        control_expr.index = control_expr.index.map(lambda x: probe_map.get(str(x), str(x)))

    # ── 4. Compute differential expression ────────────────────────────────────
    common_genes = disease_expr.index.intersection(control_expr.index)
    if len(common_genes) == 0:
        logger.error(f"No common genes between disease and control for {geo_id}")
        return None

    disease_vals = disease_expr.loc[common_genes]
    control_vals = control_expr.loc[common_genes]

    signature = {}
    for gene in common_genes:
        d_row = disease_vals.loc[gene].dropna().values
        c_row = control_vals.loc[gene].dropna().values

        # If duplicate mappings exist for a gene, treat arrays uniformly
        if d_row.ndim > 1: d_row = d_row.flatten()
        if c_row.ndim > 1: c_row = c_row.flatten()

        if len(d_row) < 2 or len(c_row) < 2:
            continue

        try:
            d_mean = np.mean(d_row)
            c_mean = np.mean(c_row)
            log2fc = (
                np.log2(d_mean + 1) - np.log2(c_mean + 1)
                if d_mean > 50 else d_mean - c_mean
            )
            _, pval = stats.ttest_ind(d_row, c_row)
            if pval < 0.1:
                signature[str(gene)] = float(log2fc)
        except Exception:
            continue

    logger.info(f"{geo_id}: computed signature for {len(signature)} genes")

    if len(signature) > top_n_genes:
        sorted_by_abs = sorted(
            signature.items(), key=lambda x: abs(x[1]), reverse=True
        )
        signature = dict(sorted_by_abs[:top_n_genes])

    return signature


def _build_probe_to_gene_map(platform_table) -> dict[str, str]:
    probe_map = {}
    if platform_table is None or platform_table.empty:
        return probe_map

    gene_col = None
    for col in platform_table.columns:
        col_lower = col.lower()
        if "gene_symbol" in col_lower or "gene symbol" in col_lower:
            gene_col = col
            break
        if "symbol" in col_lower:
            gene_col = col
            break

    id_col = platform_table.columns[0]

    if gene_col and id_col:
        for _, row in platform_table.iterrows():
            probe = str(row[id_col])
            gene  = str(row[gene_col]).strip()
            if gene and gene != "nan" and gene != "---":
                genes = [g.strip() for g in gene.split("///")]
                probe_map[probe] = genes[0] if genes else gene

    return probe_map


def run_all():
    results = {}

    for config in DISEASE_GEO_CONFIG:
        disease_id = config["disease_id"]
        geo_id     = config["geo_id"]
        logger.info(
            f"\n{'='*60}\n"
            f"Processing: {config['disease_name']} ({disease_id})\n"
            f"GEO series: {geo_id}\n"
            f"{'='*60}"
        )

        sig = compute_signature_from_geo(
            geo_id=geo_id,
            disease_keyword=config["disease_keyword"],
            control_keyword=config["control_keyword"],
        )

        if sig:
            geo_path = os.path.join(
                OUTPUT_DIR,
                f"{disease_id.replace(':', '_')}_{geo_id}.json"
            )
            with open(geo_path, "w") as f:
                json.dump(sig, f, indent=2)
            logger.info(f"Saved: {geo_path}")

            if disease_id not in results:
                results[disease_id] = {}
            for gene, log2fc in sig.items():
                if gene in results[disease_id]:
                    results[disease_id][gene].append(log2fc)
                else:
                    results[disease_id][gene] = [log2fc]

    # Save merged signatures
    for disease_id, gene_values in results.items():
        merged      = {gene: float(np.mean(vals)) for gene, vals in gene_values.items()}
        merged_path = os.path.join(
            OUTPUT_DIR,
            f"{disease_id.replace(':', '_')}.json"
        )
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info(
            f"Saved merged signature for {disease_id}: "
            f"{len(merged)} genes → {merged_path}"
        )

    logger.info(f"\nDone. {len(results)} disease signatures computed.")


if __name__ == "__main__":
    run_all()