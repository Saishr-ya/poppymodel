"""
data/scripts/compute_geo_signatures.py

Pre-compute disease gene expression signatures from GEO datasets.
Run this ONCE per disease before using Layer 2 (transcriptomics).

Output: data/processed/geo_signatures/{disease_id}.json
Format: {"GENE1": 2.3, "GENE2": -1.8, ...}  (log2 fold change, disease vs healthy)

Usage:
    python data/scripts/compute_geo_signatures.py

Keyword fix notes (from manual inspection of GEO sample metadata):
  GSE15197: disease samples say "IPAH" in characteristics_ch1, controls say "donor"
            (not "normal" as previously coded — that's why 0 samples were found)
  GSE43955: sample titles are all lowercase; disease samples have "gaucher" in
            characteristics, controls have "healthy" not "control"
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

# ── Curated disease → GEO dataset mappings (corrected keywords) ───────────────
DISEASE_GEO_CONFIG = [
    # PAH — GSE113439: working (14 disease, 11 control confirmed)
    {
        "disease_id":      "ORPHA:422",
        "disease_name":    "Pulmonary arterial hypertension",
        "geo_id":          "GSE113439",
        "disease_keyword": "PAH",
        "control_keyword": "control",
        "tissue":          "lung",
    },
    # PAH — GSE15197: fixed keywords (was "IPAH"/"normal", correct is "IPAH"/"donor")
    {
        "disease_id":      "ORPHA:422",
        "disease_name":    "Pulmonary arterial hypertension",
        "geo_id":          "GSE15197",
        "disease_keyword": "IPAH",
        "control_keyword": "donor",
        "tissue":          "lung",
    },
    # Gaucher — GSE43955: fixed keywords (was "Gaucher"/"control", correct is "gaucher"/"healthy")
    {
        "disease_id":      "ORPHA:77",
        "disease_name":    "Gaucher disease type 1",
        "geo_id":          "GSE43955",
        "disease_keyword": "gaucher",
        "control_keyword": "healthy",
        "tissue":          "macrophage",
    },
    # Dravet syndrome — SCN1A haploinsufficiency, iPSC-derived neurons
    {
        "disease_id":      "ORPHA:33069",
        "disease_name":    "Dravet syndrome",
        "geo_id":          "GSE82109",
        "disease_keyword": "Dravet",
        "control_keyword": "control",
        "tissue":          "neuron",
    },
    # Pompe disease — GAA deficiency in muscle
    {
        "disease_id":      "ORPHA:566",
        "disease_name":    "Pompe disease",
        "geo_id":          "GSE38680",
        "disease_keyword": "Pompe",
        "control_keyword": "normal",
        "tissue":          "muscle",
    },
]

OUTPUT_DIR = "data/processed/geo_signatures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    Returns dict: gene_symbol → log2 fold change (positive = up in disease).
    """
    try:
        import GEOparse
    except ImportError:
        logger.error("GEOparse not installed. Run: pip install GEOparse")
        return None

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed.")
        return None

    os.makedirs(destdir, exist_ok=True)
    logger.info(f"Downloading GEO series {geo_id}...")

    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=destdir, silent=True)
    except Exception as e:
        logger.error(f"Failed to download {geo_id}: {e}")
        return None

    # ── 1. Identify disease and control samples ───────────────────────────────
    disease_samples = []
    control_samples = []

    for sample_name, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0].lower()
        characteristics = " ".join(
            c for cs in gsm.metadata.get("characteristics_ch1", []) for c in [cs]
        ).lower()
        combined = f"{title} {characteristics}"

        if disease_keyword.lower() in combined:
            disease_samples.append(sample_name)
        elif control_keyword.lower() in combined:
            control_samples.append(sample_name)

    logger.info(
        f"{geo_id}: {len(disease_samples)} disease, "
        f"{len(control_samples)} control samples"
    )

    if len(disease_samples) < min_samples or len(control_samples) < min_samples:
        logger.warning(
            f"Insufficient samples in {geo_id} with keywords "
            f"disease='{disease_keyword}', control='{control_keyword}'. "
            f"Review the GEO series manually and adjust keywords."
        )
        # Print sample titles to help debug
        logger.info(f"Sample titles in {geo_id} (first 10):")
        for i, (name, gsm) in enumerate(list(gse.gsms.items())[:10]):
            title = gsm.metadata.get("title", [""])[0]
            chars = gsm.metadata.get("characteristics_ch1", [])
            logger.info(f"  {name}: '{title}' | chars: {chars}")
        return None

    # ── 2. Extract expression matrix ──────────────────────────────────────────
    try:
        disease_expr = pd.DataFrame({
            s: gse.gsms[s].table.set_index("ID_REF")["VALUE"]
            for s in disease_samples
            if s in gse.gsms and "VALUE" in gse.gsms[s].table.columns
        })
        control_expr = pd.DataFrame({
            s: gse.gsms[s].table.set_index("ID_REF")["VALUE"]
            for s in control_samples
            if s in gse.gsms and "VALUE" in gse.gsms[s].table.columns
        })

        if disease_expr.empty or control_expr.empty:
            logger.error(f"Could not extract expression matrix from {geo_id}")
            return None

        disease_expr = disease_expr.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        control_expr = control_expr.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    except Exception as e:
        logger.error(f"Expression matrix extraction failed for {geo_id}: {e}")
        return None

    # ── 3. Map probe IDs to gene symbols ──────────────────────────────────────
    if hasattr(gse, "gpls") and gse.gpls:
        platform_key = list(gse.gpls.keys())[0]
        gpl          = gse.gpls[platform_key]
        probe_map    = _build_probe_to_gene_map(gpl.table)
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
    logger.info(
        "Now run: python run_engine.py validate  "
        "to see transcriptomic layer activate"
    )


if __name__ == "__main__":
    run_all()