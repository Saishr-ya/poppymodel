#!/usr/bin/env python3
"""
data/scripts/compute_geo_signatures.py

Compute DEG signatures from GEO datasets for use in Layer 2 (Transcriptomic).

Fix #14: Removed hardcoded DISEASE_GEO_CONFIG array. Dataset configuration
         is now loaded from config/geo_datasets.json. Add new diseases there
         without touching this script.

Fix #15: Replaced fragile GPL column-name guessing with mygene-based probe
         mapping. mygene translates any stable probe/Entrez/Ensembl ID to
         a unified HGNC symbol, handling platform-specific header variations
         automatically.

Fix #16: Added 'discover' subcommand. Given a disease name and Orphanet ID,
         searches NCBI GEO via Entrez API, scores candidate datasets by sample
         count and metadata quality, auto-detects case/control keywords from
         characteristics blocks, and writes confirmed entries to
         config/geo_datasets.json.

Usage:
    python data/scripts/compute_geo_signatures.py                        # all datasets
    python data/scripts/compute_geo_signatures.py GSE113439              # one dataset
    python data/scripts/compute_geo_signatures.py --list                 # list configured
    python data/scripts/compute_geo_signatures.py discover "Gaucher disease" ORPHA:77
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import GEOparse
import mygene
import numpy as np
import pandas as pd
import requests
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "geo_datasets.json"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed" / "geo_signatures"
CACHE_DIR    = PROJECT_ROOT / "data" / "raw" / "geo_cache"

mg = mygene.MyGeneInfo()

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
GEO_QUERY_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"

# Common control/case terms for auto-detection
# Covers clinical samples AND cell/iPSC model terminology
_CONTROL_TERMS = {
    "control", "healthy", "normal", "donor", "unaffected", "non-disease",
    "ctrl", "hc", "nd",
    "wildtype", "wild type", "wild-type", "wt", "isogenic",
    "scramble", "scrambled", "empty vector", "vehicle",
    "untreated", "mock", "parental",
    "rescue", "corrected", "restored",
}
_CASE_TERMS = {
    "patient", "disease", "affected", "case", "subject", "diagnosed",
    "knock down", "knockdown", "knockout", "knock out", "kd", "ko",
    "mutant", "mutation", "deficient", "deficiency",
    "overexpression", "transfected",
    "gla", "fabry", "gaucher", "pompe", "dravet", "pah",
    "lysosomal", "enzyme deficiency",
}


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"datasets": []}
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_configured_geo_ids() -> set[str]:
    return {d["geo_id"] for d in load_config().get("datasets", [])}


# ── GEO discovery via NCBI Entrez ─────────────────────────────────────────────

def search_geo_datasets(disease_name: str, max_results: int = 20) -> list[str]:
    """
    Search NCBI GEO for expression datasets matching a disease name.
    Returns list of GEO Series UIDs.
    """
    query = (
        f'{disease_name}[Title/Abstract] '
        f'AND "Homo sapiens"[Organism]'
    )
    try:
        r = requests.get(
            NCBI_ESEARCH,
            params={
                "db": "gds",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "usehistory": "n",
            },
            timeout=20,
            headers={"User-Agent": "PoppyRepurposingEngine/1.0 (research)"},
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        logger.info(f"NCBI GEO search '{disease_name}': {len(ids)} datasets found")
        return ids
    except Exception as e:
        logger.error(f"NCBI GEO search failed: {e}")
        return []


def fetch_geo_summaries(uids: list[str]) -> list[dict]:
    """Fetch GEO dataset summaries for a list of UIDs."""
    if not uids:
        return []
    time.sleep(0.4)  # NCBI rate limit: 3 req/sec without API key
    try:
        r = requests.get(
            NCBI_ESUMMARY,
            params={"db": "gds", "id": ",".join(uids), "retmode": "json"},
            timeout=20,
            headers={"User-Agent": "PoppyRepurposingEngine/1.0 (research)"},
        )
        r.raise_for_status()
        result = r.json().get("result", {})
        summaries = [result[uid] for uid in uids if uid in result and uid != "uids"]
        logger.info(f"Fetched {len(summaries)} GEO summaries")
        return summaries
    except Exception as e:
        logger.error(f"NCBI GEO summary fetch failed: {e}")
        return []


def score_dataset(summary: dict) -> tuple[float, dict]:
    """
    Score a GEO dataset summary for suitability.
    Returns (score, info_dict).

    NCBI esummary (db=gds) field names:
      accession → GSE ID (e.g. "GSE12345")
      title     → dataset title
      n_samples → sample count (int or string)
      gpl       → platform accession
      summary   → abstract text
    """
    score = 0.0
    try:
        n_samples = int(summary.get("n_samples", 0))
    except (ValueError, TypeError):
        n_samples = 0

    info = {
        "geo_id":    summary.get("accession", ""),
        "title":     summary.get("title", ""),
        "n_samples": n_samples,
        "platform":  summary.get("gpl", ""),
        "summary":   summary.get("summary", "")[:200],
    }

    n = info["n_samples"]
    # Sample count scoring — need at least 6 (3 per group)
    if n >= 20:
        score += 3.0
    elif n >= 10:
        score += 2.0
    elif n >= 6:
        score += 1.0
    else:
        score -= 5.0  # too small, heavily penalise

    # Prefer datasets with "case" and "control" in summary
    summary_lower = info["summary"].lower()
    if "control" in summary_lower or "healthy" in summary_lower:
        score += 1.0
    if "patient" in summary_lower or "disease" in summary_lower:
        score += 1.0

    # Prefer human microarray platforms
    platform = info["platform"].upper()
    if platform.startswith("GPL"):
        score += 0.5

    return score, info


def detect_case_control_keywords(
    geo_id: str,
    auto_confirm: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Download a GEO series and extract case/control keywords from
    characteristics_ch1 metadata blocks.

    When auto-classify fails, shows the user the unique metadata values
    and prompts them to pick case/control keywords instead of silently
    returning empty (which caused every Fabry disease dataset to be skipped).

    Returns (case_keywords, control_keywords).
    """
    logger.info(f"Downloading {geo_id} to detect sample groups...")
    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(CACHE_DIR), silent=True)
    except Exception as e:
        logger.warning(f"Could not download {geo_id}: {e}")
        return [], []

    # Collect all characteristics values across all samples
    all_values: list[str] = []
    for gsm in gse.gsms.values():
        for ch in gsm.metadata.get("characteristics_ch1", []):
            parts = ch.split(":", 1)
            val = parts[-1].strip().lower()
            if val:
                all_values.append(val)

    if not all_values:
        for gsm in gse.gsms.values():
            title = " ".join(gsm.metadata.get("title", [])).lower()
            all_values.append(title)

    value_counts = Counter(all_values)
    unique_values = [v for v, _ in value_counts.most_common(30)]

    # Classify each unique value as case, control, or ambiguous
    case_kw, control_kw = [], []
    for val in unique_values:
        val_words = set(val.replace("-", " ").replace("_", " ").split())
        if val_words & _CONTROL_TERMS:
            control_kw.append(val)
        elif val_words & _CASE_TERMS:
            case_kw.append(val)

    # Auto-classify succeeded
    if case_kw or control_kw:
        return list(dict.fromkeys(case_kw))[:5], list(dict.fromkeys(control_kw))[:5]

    # Auto-classify failed — show values and prompt user (unless --yes)
    print(f"\n  Could not auto-classify sample groups for {geo_id}.")
    print(f"  Unique metadata values found ({len(unique_values)} total):")
    for i, v in enumerate(unique_values[:15], 1):
        print(f"    [{i:2d}] {v}  (n={value_counts[v]})")

    if auto_confirm:
        # In --yes mode, pick the two most frequent values as case/control
        if len(unique_values) >= 2:
            print(f"  --yes mode: auto-assigning case={unique_values[0]}, control={unique_values[1]}")
            return [unique_values[0]], [unique_values[1]]
        return [], []

    print()
    case_input = input(
        "  Enter CASE keyword(s) from the list above (comma-separated, or blank to skip): "
    ).strip()
    if not case_input:
        return [], []
    ctrl_input = input(
        "  Enter CONTROL keyword(s) from the list above (comma-separated): "
    ).strip()
    if not ctrl_input:
        return [], []

    case_kw    = [k.strip() for k in case_input.split(",")  if k.strip()]
    control_kw = [k.strip() for k in ctrl_input.split(",") if k.strip()]
    return case_kw, control_kw


# ── Discover subcommand ────────────────────────────────────────────────────────

def cmd_discover(disease_name: str, disease_id: str, auto_confirm: bool = False):
    """
    Search GEO for datasets matching a disease, score candidates,
    auto-detect case/control keywords, and offer to add to config.
    """
    already_configured = get_configured_geo_ids()

    print(f"\n🔍 Searching GEO for: {disease_name} ({disease_id})\n")

    uids = search_geo_datasets(disease_name, max_results=20)
    if not uids:
        print("No datasets found. Try a different disease name spelling.")
        return

    summaries = fetch_geo_summaries(uids)
    if not summaries:
        print("Could not fetch dataset summaries.")
        return

    # Score and sort
    scored = sorted(
        [score_dataset(s) for s in summaries],
        key=lambda x: x[0],
        reverse=True,
    )

    # Filter: skip already configured, require ≥6 samples
    candidates = [
        (score, info) for score, info in scored
        if info["geo_id"] not in already_configured
        and info["n_samples"] >= 6
    ][:5]  # top 5

    if not candidates:
        print("No new suitable datasets found (all already configured or too small).")
        return

    print(f"Top candidates for '{disease_name}':\n")
    for i, (score, info) in enumerate(candidates, 1):
        print(f"  [{i}] {info['geo_id']}  (score={score:.1f}, n={info['n_samples']} samples)")
        print(f"      {info['title']}")
        print(f"      {info['summary'][:120]}...")
        print()

    # For each candidate, detect keywords and present for confirmation
    added = []
    for score, info in candidates:
        geo_id = info["geo_id"]
        print(f"─── {geo_id} ───────────────────────────────────────────")

        case_kw, control_kw = detect_case_control_keywords(geo_id, auto_confirm=auto_confirm)

        if not case_kw or not control_kw:
            print(f"  ⚠  Could not auto-detect case/control groups for {geo_id}.")
            print(f"     You can add it manually to config/geo_datasets.json.")
            print()
            continue

        print(f"  Auto-detected case keywords:    {case_kw}")
        print(f"  Auto-detected control keywords: {control_kw}")
        print()

        if auto_confirm:
            confirm = "y"
        else:
            confirm = input(f"  Add {geo_id} to config? [y/N/edit] ").strip().lower()

        if confirm == "edit":
            case_kw    = input("  Enter case keywords (comma-separated): ").split(",")
            control_kw = input("  Enter control keywords (comma-separated): ").split(",")
            case_kw    = [k.strip() for k in case_kw if k.strip()]
            control_kw = [k.strip() for k in control_kw if k.strip()]
            confirm    = "y"

        if confirm == "y":
            entry = {
                "geo_id":           geo_id,
                "disease_id":       disease_id,
                "disease_name":     disease_name,
                "case_keywords":    case_kw,
                "control_keywords": control_kw,
                "notes":            f"Auto-discovered. Title: {info['title'][:80]}",
            }
            cfg = load_config()
            cfg.setdefault("datasets", []).append(entry)
            save_config(cfg)
            print(f"  ✓ Added {geo_id} to config/geo_datasets.json")
            added.append(geo_id)
        else:
            print(f"  Skipped {geo_id}")
        print()

    if added:
        print(f"\nAdded {len(added)} dataset(s): {added}")
        print("Run the pipeline now?")
        if auto_confirm or input("  Compute signatures now? [y/N] ").strip().lower() == "y":
            datasets = load_config()["datasets"]
            for d in datasets:
                if d["geo_id"] in added:
                    process_dataset(d)
    else:
        print("No datasets added.")


# ── Probe → gene mapping via mygene ───────────────────────────────────────────

def map_probes_to_genes(gpl) -> dict[str, str]:
    """
    Fix #15: Use mygene to map probe IDs → HGNC gene symbols.
    Tries Entrez IDs first, then Ensembl, then direct symbol column.
    """
    table = gpl.table
    if table is None or table.empty:
        return {}

    cols_lower = {c.lower(): c for c in table.columns}

    # Try Entrez ID columns
    entrez_col = next(
        (cols_lower[k] for k in cols_lower
         if k in ("entrez_gene_id", "entrez_id", "gene_id", "entrezid", "gene", "gb_acc")),
        None,
    )
    if entrez_col:
        ids = table[entrez_col].dropna().astype(str).str.strip()
        ids = ids[ids.str.match(r'^\d+$')]
        if len(ids) > 100:
            logger.info(f"Mapping {len(ids)} probes via Entrez IDs using mygene")
            try:
                result = mg.querymany(
                    ids.tolist(), scopes="entrezgene", fields="symbol",
                    species="human", returnall=False, verbose=False,
                )
                entrez_to_symbol = {
                    str(r["query"]): r["symbol"]
                    for r in result if "symbol" in r and not r.get("notfound")
                }
                probe_to_gene = {}
                for probe_id, entrez_id in zip(table.index, ids):
                    sym = entrez_to_symbol.get(str(entrez_id))
                    if sym:
                        probe_to_gene[str(probe_id)] = sym
                logger.info(f"mygene mapped {len(probe_to_gene)} probes via Entrez")
                return probe_to_gene
            except Exception as e:
                logger.warning(f"mygene Entrez mapping failed: {e}")

    # Try Ensembl ID columns
    ensembl_col = next(
        (cols_lower[k] for k in cols_lower if "ensembl" in k), None
    )
    if ensembl_col:
        ids = table[ensembl_col].dropna().astype(str).str.strip()
        ids = ids[ids.str.match(r'^ENSG\d+')]
        if len(ids) > 100:
            logger.info(f"Mapping {len(ids)} probes via Ensembl IDs using mygene")
            try:
                result = mg.querymany(
                    ids.tolist(), scopes="ensembl.gene", fields="symbol",
                    species="human", returnall=False, verbose=False,
                )
                ensembl_to_symbol = {
                    r["query"]: r["symbol"]
                    for r in result if "symbol" in r and not r.get("notfound")
                }
                probe_to_gene = {}
                for probe_id, ensembl_id in zip(table.index, ids):
                    sym = ensembl_to_symbol.get(str(ensembl_id))
                    if sym:
                        probe_to_gene[str(probe_id)] = sym
                logger.info(f"mygene mapped {len(probe_to_gene)} probes via Ensembl")
                return probe_to_gene
            except Exception as e:
                logger.warning(f"mygene Ensembl mapping failed: {e}")

    # Last resort: direct symbol column
    symbol_col = next(
        (cols_lower[k] for k in cols_lower
         if k in ("gene_symbol", "gene symbol", "symbol", "genesymbol",
                  "gene_name", "ilmn_gene", "orf")),
        None,
    )
    if symbol_col:
        logger.info(f"Using direct symbol column '{symbol_col}'")
        return {
            str(probe): str(sym)
            for probe, sym in zip(table.index, table[symbol_col])
            if pd.notna(sym) and str(sym).strip()
        }

    logger.warning("No usable gene mapping column found in GPL table")
    return {}


# ── Sample stratification ──────────────────────────────────────────────────────

def stratify_samples(
    gse,
    case_keywords: list[str],
    control_keywords: list[str],
) -> tuple[list[str], list[str]]:
    """
    Split GSM sample IDs into case and control groups.
    Searches characteristics values only (not keys) to avoid false positives.
    """
    cases, controls = [], []
    for gsm_id, gsm in gse.gsms.items():
        char_values = []
        for ch in gsm.metadata.get("characteristics_ch1", []):
            parts = ch.split(":", 1)
            char_values.append(parts[-1].lower().strip())
        title  = " ".join(gsm.metadata.get("title", [])).lower()
        source = " ".join(gsm.metadata.get("source_name_ch1", [])).lower()
        search_text = " ".join(char_values) + " " + title + " " + source

        if any(kw.lower() in search_text for kw in case_keywords):
            cases.append(gsm_id)
        elif any(kw.lower() in search_text for kw in control_keywords):
            controls.append(gsm_id)

    return cases, controls


# ── DEG computation ────────────────────────────────────────────────────────────

def compute_deg_signature(
    gse,
    case_ids: list[str],
    control_ids: list[str],
    probe_to_gene: dict[str, str],
    min_samples: int = 3,
) -> Optional[pd.DataFrame]:
    try:
        expr = gse.pivot_samples("VALUE")
    except Exception as e:
        logger.error(f"pivot_samples failed: {e}")
        return None

    case_cols    = [c for c in case_ids    if c in expr.columns]
    control_cols = [c for c in control_ids if c in expr.columns]

    if len(case_cols) < min_samples or len(control_cols) < min_samples:
        logger.error(
            f"Insufficient samples: {len(case_cols)} cases, "
            f"{len(control_cols)} controls (need ≥{min_samples} each)"
        )
        return None

    logger.info(f"Computing DEGs: {len(case_cols)} cases vs {len(control_cols)} controls")

    expr.index = expr.index.astype(str)
    expr["gene"] = expr.index.map(probe_to_gene)
    expr = expr.dropna(subset=["gene"])
    expr = expr.groupby("gene").mean()

    case_data    = expr[case_cols].values.astype(float)
    control_data = expr[control_cols].values.astype(float)

    t_stats, p_values = stats.ttest_ind(
        case_data, control_data, axis=1, equal_var=False, nan_policy="omit"
    )
    mean_case    = np.nanmean(case_data,    axis=1)
    mean_control = np.nanmean(control_data, axis=1)
    log2fc       = mean_case - mean_control

    from statsmodels.stats.multitest import multipletests
    valid_mask = ~np.isnan(p_values)
    fdr = np.full_like(p_values, np.nan)
    if valid_mask.sum() > 0:
        _, fdr_vals, _, _ = multipletests(p_values[valid_mask], method="fdr_bh")
        fdr[valid_mask] = fdr_vals

    result = pd.DataFrame({
        "gene":         expr.index,
        "log2fc":       log2fc,
        "pvalue":       p_values,
        "fdr":          fdr,
        "mean_case":    mean_case,
        "mean_control": mean_control,
    }).sort_values("fdr")

    sig = result[result["fdr"] < 0.05]
    logger.info(f"DEG results: {len(result)} genes, {len(sig)} significant (FDR<0.05)")
    return result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_dataset(cfg: dict) -> bool:
    geo_id       = cfg["geo_id"]
    disease_id   = cfg["disease_id"]
    disease_name = cfg["disease_name"]

    out_path = OUTPUT_DIR / f"{disease_id.replace(':', '_')}_{geo_id}.json"
    if out_path.exists():
        logger.info(f"Signature already exists for {geo_id} — skipping")
        return True

    logger.info(f"Processing {geo_id} → {disease_name} ({disease_id})")

    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(CACHE_DIR), silent=True)
    except Exception as e:
        logger.error(f"Failed to download {geo_id}: {e}")
        return False

    platform_type = ""
    for gpl in gse.gpls.values():
        platform_type = " ".join(gpl.metadata.get("technology", [])).lower()
        break
    if "sequencing" in platform_type or "rna-seq" in platform_type:
        logger.warning(f"{geo_id} is RNA-Seq — not yet supported. Skipping.")
        return False

    probe_to_gene = {}
    for gpl in gse.gpls.values():
        probe_to_gene.update(map_probes_to_genes(gpl))

    if not probe_to_gene:
        logger.error(f"No probe-to-gene mapping for {geo_id}")
        return False

    cases, controls = stratify_samples(gse, cfg["case_keywords"], cfg["control_keywords"])
    logger.info(f"{geo_id}: {len(cases)} cases, {len(controls)} controls")

    if not cases or not controls:
        logger.error(f"Could not stratify samples for {geo_id}. Check keywords in config.")
        return False

    deg_df = compute_deg_signature(gse, cases, controls, probe_to_gene)
    if deg_df is None:
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sig = deg_df[deg_df["fdr"] < 0.05].head(500)
    output = {
        "disease_id":   disease_id,
        "disease_name": disease_name,
        "geo_id":       geo_id,
        "n_cases":      len(cases),
        "n_controls":   len(controls),
        "n_sig_genes":  len(sig),
        "top_genes":    sig[["gene", "log2fc", "fdr"]].to_dict(orient="records"),
        "full_results": deg_df[["gene", "log2fc", "pvalue", "fdr"]].to_dict(orient="records"),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved → {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compute GEO DEG signatures",
        usage="%(prog)s [geo_id] | discover <disease_name> <disease_id> | list"
    )
    subparsers = parser.add_subparsers(dest="command")

    # discover subcommand
    disc = subparsers.add_parser("discover", help="Search GEO and add new disease datasets")
    disc.add_argument("disease_name", help="Disease name to search (e.g. 'Gaucher disease')")
    disc.add_argument("disease_id",   help="Orphanet ID (e.g. ORPHA:77)")
    disc.add_argument("--yes", action="store_true", help="Auto-confirm all prompts")

    # list subcommand
    subparsers.add_parser("list", help="List configured datasets")

    # process subcommand (optional geo_id)
    proc = subparsers.add_parser("process", help="Process all or one configured dataset")
    proc.add_argument("geo_id", nargs="?", help="Single GSE ID to process (default: all)")

    args = parser.parse_args()

    if args.command == "discover":
        cmd_discover(args.disease_name, args.disease_id, auto_confirm=args.yes)
        return

    if args.command == "list":
        cfg = load_config()
        datasets = cfg.get("datasets", [])
        logger.info(f"Loaded {len(datasets)} dataset configs from {CONFIG_PATH}")
        print(f"\nConfigured datasets ({CONFIG_PATH}):\n")
        for d in datasets:
            print(f"  {d['geo_id']}  {d['disease_id']}  {d['disease_name']}")
        print()
        return

    # Default: process (with or without geo_id)
    geo_id = getattr(args, "geo_id", None)
    cfg      = load_config()
    datasets = cfg.get("datasets", [])

    if geo_id:
        matches = [d for d in datasets if d["geo_id"] == geo_id]
        if not matches:
            logger.error(f"{geo_id} not in config. Add it via 'discover' or manually.")
            sys.exit(1)
        datasets = matches

    results = {d["geo_id"]: process_dataset(d) for d in datasets}
    passed  = sum(results.values())
    logger.info(f"Done: {passed}/{len(results)} datasets processed successfully")
    if passed < len(results):
        logger.warning(f"Failed: {[k for k, v in results.items() if not v]}")


if __name__ == "__main__":
    # Support legacy --list flag
    if "--list" in sys.argv:
        sys.argv = [sys.argv[0], "list"]
    main()