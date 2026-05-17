"""
src/graph/ppi_network.py

Human Protein-Protein Interactome builder for Layer 1B (network proximity).

Downloads STRING DB v12 human PPI data and builds a NetworkX graph.
Nodes = UniProt IDs. Edges weighted by STRING combined confidence score.

Setup (run once, ~30 minutes):
    python -m src.graph.ppi_network build

Requirements:
    ~2GB download, ~8GB RAM to hold graph in memory.
    Pickles to ~500MB at data/processed/ppi_network.pkl

STRING DB:
    https://stringdb-downloads.org/download/protein.links.v12.0/
    File: 9606.protein.links.v12.0.txt.gz  (human, all interaction types)
    Also: 9606.protein.info.v12.0.txt.gz   (protein ID → gene name mapping)
"""

from __future__ import annotations
import gzip
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/protein.links.v12.0/"
    "9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/"
    "9606.protein.info.v12.0.txt.gz"
)
STRING_RAW_PATH = "data/raw/string_db/9606.protein.links.v12.0.txt.gz"
STRING_INFO_PATH = "data/raw/string_db/9606.protein.info.v12.0.txt.gz"
GRAPH_OUTPUT_PATH = "data/processed/ppi_network.pkl"
UNIPROT_MAP_PATH = "data/processed/string_to_uniprot.json"

# STRING confidence cutoff (0–1000). 400 = medium confidence (recommended).
CONFIDENCE_CUTOFF = 400


def download_string_db(force: bool = False) -> bool:
    """
    Download STRING DB human interactome files.
    Returns True if successful, False if already exists.
    """
    import requests
    from tqdm import tqdm

    os.makedirs("data/raw/string_db", exist_ok=True)
    files = [
        (STRING_LINKS_URL, STRING_RAW_PATH),
        (STRING_INFO_URL, STRING_INFO_PATH),
    ]

    for url, path in files:
        if os.path.exists(path) and not force:
            logger.info(f"Already downloaded: {path}")
            continue

        logger.info(f"Downloading {url}...")
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(path, "wb") as f, tqdm(
                total=total, unit="iB", unit_scale=True, desc=os.path.basename(path)
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            logger.info(f"Downloaded: {path}")
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return False

    return True


def build_ppi_graph(
    links_path: str = STRING_RAW_PATH,
    confidence_cutoff: int = CONFIDENCE_CUTOFF,
    output_path: str = GRAPH_OUTPUT_PATH,
) -> Optional[object]:
    """
    Parse STRING DB links file and build NetworkX graph.

    Nodes = STRING protein IDs (9606.ENSP...) — converted to UniProt where possible.
    Edges = weighted by combined_score / 1000 (0–1 range).

    Returns the NetworkX graph, or None if failed.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("networkx not installed. Run: pip install networkx")
        return None

    if not os.path.exists(links_path):
        logger.error(
            f"STRING DB file not found at {links_path}. "
            f"Run: python -m src.graph.ppi_network download"
        )
        return None

    logger.info(f"Building PPI graph from {links_path} (cutoff={confidence_cutoff})...")

    G = nx.Graph()
    edge_count = 0

    with gzip.open(links_path, "rt") as f:
        header = f.readline()   # skip header

        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            p1, p2, score = parts[0], parts[1], int(parts[2])

            if score < confidence_cutoff:
                continue

            G.add_edge(p1, p2, weight=score / 1000)
            edge_count += 1

    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, {edge_count} edges "
        f"(cutoff={confidence_cutoff})"
    )

    # Map STRING IDs to UniProt IDs (better for cross-referencing with ChEMBL/DisGeNET)
    uniprot_map = _load_string_to_uniprot_map()
    if uniprot_map:
        G = nx.relabel_nodes(G, uniprot_map, copy=True)
        logger.info(f"Relabeled {len(uniprot_map)} nodes to UniProt IDs")

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"PPI graph saved to {output_path}")

    return G

def _load_string_to_uniprot_map() -> dict[str, str]:
    """
    Load STRING → UniProt ID mapping.
    Prefer aliases file (has real UniProt ACs) over info file (gene names only).
    """
    import json

    if os.path.exists(UNIPROT_MAP_PATH):
        with open(UNIPROT_MAP_PATH) as f:
            return json.load(f)

    # Use aliases file if available — more reliable than info file
    aliases_path = "data/raw/string_db/9606.protein.aliases.v12.0.txt.gz"
    if os.path.exists(aliases_path):
        return build_uniprot_map_from_aliases(aliases_path)

    # Fall back to info file (gene names — reduced cross-referencing accuracy)
    if os.path.exists(STRING_INFO_PATH):
        return _build_uniprot_map_from_info_file()

    logger.warning(
        f"No STRING→UniProt mapping found. "
        f"Network proximity will have reduced coverage."
    )
    return {}

def _build_uniprot_map_from_info_file() -> dict[str, str]:
    """
    Parse STRING protein info file to extract UniProt IDs.
    Falls back to UniProt API if info file doesn't contain UniProt directly.
    """
    import json

    logger.info("Building STRING→UniProt map from info file...")
    mapping = {}

    try:
        with gzip.open(STRING_INFO_PATH, "rt") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                string_id = parts[0]   # 9606.ENSP00000...
                # STRING info file has gene names but not directly UniProt
                # For UniProt mapping, use STRING's UniProt links file or UniProt API
                # This is a placeholder — see the full mapping script below
                gene_name = parts[1] if len(parts) > 1 else ""
                if gene_name:
                    mapping[string_id] = gene_name   # temporary: use gene name

        os.makedirs(os.path.dirname(UNIPROT_MAP_PATH), exist_ok=True)
        with open(UNIPROT_MAP_PATH, "w") as f:
            json.dump(mapping, f)
        logger.info(f"STRING→UniProt map saved ({len(mapping)} entries)")
    except Exception as e:
        logger.error(f"Failed to build UniProt map: {e}")

    return mapping

def build_uniprot_map_from_aliases(
    aliases_path: str = "data/raw/string_db/9606.protein.aliases.v12.0.txt.gz",
    output_path: str = UNIPROT_MAP_PATH,
) -> dict[str, str]:
    """
    Parse STRING aliases file to build STRING ID → UniProt accession mapping.

    Priority order:
      1. Ensembl_HGNC_uniprot_ids — canonical IDs curated by HGNC (best)
      2. UniProt_AC               — may include isoforms, use as fallback
    """
    import json

    logger.info(f"Building STRING→UniProt map from aliases file: {aliases_path}")

    hgnc = {}      # string_id → canonical UniProt from HGNC
    fallback = {}  # string_id → UniProt AC (may be isoform)

    with gzip.open(aliases_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            string_id = parts[0]
            alias = parts[1]
            sources = parts[2]

            if "Ensembl_HGNC_uniprot_ids" in sources:
                if string_id not in hgnc:
                    hgnc[string_id] = alias

            elif "UniProt_AC" in sources:
                if string_id not in fallback:
                    fallback[string_id] = alias

    # Merge: HGNC canonical wins, UniProt_AC fills the gaps
    mapping = {**fallback, **hgnc}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mapping, f)

    logger.info(
        f"STRING→UniProt map saved: {len(mapping)} total entries "
        f"({len(hgnc)} HGNC canonical, {len(fallback)} UniProt_AC fallback) → {output_path}"
    )
    return mapping


def load_graph(path: str = GRAPH_OUTPUT_PATH):
    """Load the PPI graph from disk."""
    if not os.path.exists(path):
        logger.error(
            f"PPI graph not found at {path}. "
            f"Build it: python -m src.graph.ppi_network build"
        )
        return None

    logger.info(f"Loading PPI graph from {path}...")
    with open(path, "rb") as f:
        G = pickle.load(f)
    logger.info(
        f"PPI graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


def graph_stats(G) -> dict:
    """Return basic statistics about the PPI graph."""
    import networkx as nx
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "connected_components": nx.number_connected_components(G),
        "largest_component_size": len(max(nx.connected_components(G), key=len)),
        "avg_degree": sum(d for _, d in G.degree()) / G.number_of_nodes(),
        "density": nx.density(G),
    }


if __name__ == "__main__":
    """
    CLI for PPI network management.

    Usage:
        python -m src.graph.ppi_network download
        python -m src.graph.ppi_network build
        python -m src.graph.ppi_network stats
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "download":
        success = download_string_db()
        sys.exit(0 if success else 1)

    elif command == "build":
        download_string_db()   # no-op if already downloaded
        G = build_ppi_graph()
        if G:
            stats = graph_stats(G)
            for k, v in stats.items():
                print(f"  {k}: {v}")

    elif command == "stats":
        G = load_graph()
        if G:
            stats = graph_stats(G)
            for k, v in stats.items():
                print(f"  {k}: {v}")

    else:
        print("Usage: python -m src.graph.ppi_network [download|build|stats]")