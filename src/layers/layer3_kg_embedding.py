"""
src/layers/layer3_kg_embedding.py

Layer 3 — Knowledge Graph Embedding.

URL fix: the Hetionet raw GitHub URLs changed. The nodes and edges files
are now served via GitHub releases and a Zenodo archive. Updated with
working URLs and a fallback download sequence.

The edges file (2.2M edges, ~250MB compressed) is the most important.
The nodes file is small (~2MB). Both are required for graph construction.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = "data/processed/kg_embeddings.json"
GRAPH_NODES_PATH = "data/processed/kg_nodes.json"

# Hetionet v1.0 — corrected URLs
# Primary: GitHub releases (direct download, no auth required)
# Fallback: Zenodo archive DOI 10.5281/zenodo.1043597
HETIONET_NODES_URLS = [
    "https://github.com/hetio/hetionet/releases/download/v1.0.0/hetionet-v1.0-nodes.json",
    "https://zenodo.org/record/1043597/files/hetionet-v1.0-nodes.json",
    # Last resort: specific commit SHA
    "https://raw.githubusercontent.com/hetio/hetionet/8a9b79b18a76ae5a1b59c6c6db5cd35f2e1d6d9e/hetnet/json/hetionet-v1.0-nodes.json",
]
HETIONET_EDGES_URLS = [
    "https://github.com/hetio/hetionet/releases/download/v1.0.0/hetionet-v1.0-edges.sif.gz",
    "https://zenodo.org/record/1043597/files/hetionet-v1.0-edges.sif.gz",
]


class KGEmbeddingLayer(BaseLayer):
    """
    Layer 3 — Knowledge graph embedding-based drug-disease scoring.

    Scores:
        pair.scores.kg_embedding_cosine  (0–1; higher = more similar embeddings)
    """

    layer_name = "layer3_kg_embedding"
    version    = "1.0"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._embeddings: Optional[dict[str, list[float]]] = None

    @property
    def embeddings(self) -> Optional[dict[str, list[float]]]:
        if self._embeddings is None:
            self._embeddings = self._load_embeddings()
        return self._embeddings

    def _load_embeddings(self) -> Optional[dict[str, list[float]]]:
        if not os.path.exists(EMBEDDINGS_PATH):
            logger.warning(
                f"KG embeddings not found at {EMBEDDINGS_PATH}. "
                "Run: python -m src.layers.layer3_kg_embedding build"
            )
            return None
        with open(EMBEDDINGS_PATH) as f:
            data = json.load(f)
        logger.info(f"KG embeddings loaded: {len(data)} nodes")
        return data

    def score(self, pair: CandidatePair) -> CandidatePair:
        if self.embeddings is None:
            return pair

        drug_emb    = self._get_embedding(pair.drug_id, "Drug")
        disease_emb = self._get_embedding(pair.disease_id, "Disease")

        if drug_emb is None or disease_emb is None:
            return pair

        cosine = self._cosine_similarity(drug_emb, disease_emb)
        pair.scores.kg_embedding_cosine = float(cosine)
        logger.debug(
            f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
            f"KG cosine={cosine:.4f}"
        )
        return pair

    def _get_embedding(
        self, entity_id: str, entity_type: str
    ) -> Optional[list[float]]:
        if self.embeddings is None:
            return None
        for key in [
            entity_id,
            f"{entity_type}::{entity_id}",
            entity_id.replace("ORPHA:", "Disease::Orphanet:"),
            entity_id.replace("OMIM:", "Disease::OMIM:"),
            entity_id.replace("CHEMBL", "Compound::ChEMBL:CHEMBL"),
        ]:
            if key in self.embeddings:
                return self.embeddings[key]
        return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        va    = np.array(a, dtype=float)
        vb    = np.array(b, dtype=float)
        na    = np.linalg.norm(va)
        nb    = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))


# ── Graph building and embedding training ─────────────────────────────────────

def _download_with_fallback(urls: list[str], dest_path: str) -> bool:
    """Try each URL in order until one succeeds. Returns True if downloaded."""
    import requests
    from tqdm import tqdm

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    for url in urls:
        logger.info(f"Trying: {url}")
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                logger.warning(f"  HTTP {r.status_code} — trying next URL")
                continue

            total = int(r.headers.get("content-length", 0))
            with open(dest_path, "wb") as f, tqdm(
                total=total, unit="iB", unit_scale=True,
                desc=os.path.basename(dest_path)
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            logger.info(f"  Downloaded to {dest_path}")
            return True

        except Exception as e:
            logger.warning(f"  Failed ({e}) — trying next URL")
            if os.path.exists(dest_path):
                os.remove(dest_path)   # remove partial download

    return False


def build_hetionet_graph(output_dir: str = "data/raw/hetionet") -> Optional[object]:
    """
    Download and load Hetionet v1.0.

    Uses a fallback URL list because the Hetionet project has moved files
    between GitHub branches and Zenodo archives over time.
    """
    try:
        import networkx as nx
        import gzip
    except ImportError as e:
        logger.error(f"Required package missing: {e}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    nodes_path = os.path.join(output_dir, "hetionet-v1.0-nodes.json")
    edges_path = os.path.join(output_dir, "hetionet-v1.0-edges.sif.gz")

    # Download nodes
    if not os.path.exists(nodes_path):
        logger.info("Downloading Hetionet nodes...")
        if not _download_with_fallback(HETIONET_NODES_URLS, nodes_path):
            logger.error(
                "Could not download Hetionet nodes from any URL.\n"
                "Manual download: https://zenodo.org/record/1043597\n"
                f"Place at: {nodes_path}"
            )
            return None

    # Download edges
    if not os.path.exists(edges_path):
        logger.info("Downloading Hetionet edges (~250MB)...")
        if not _download_with_fallback(HETIONET_EDGES_URLS, edges_path):
            logger.error(
                "Could not download Hetionet edges from any URL.\n"
                "Manual download: https://zenodo.org/record/1043597\n"
                f"Place at: {edges_path}"
            )
            return None

    # Build graph
    logger.info("Building NetworkX graph from Hetionet...")
    G = nx.Graph()

    with open(nodes_path) as f:
        nodes_data = json.load(f)
    for node in nodes_data.get("nodes", []):
        G.add_node(
            node["identifier"],
            name=node.get("name", ""),
            kind=node.get("kind", ""),
        )

    with gzip.open(edges_path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                G.add_edge(parts[0], parts[2], relation=parts[1])

    logger.info(
        f"Hetionet graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def train_node2vec_embeddings(
    graph,
    output_path: str = EMBEDDINGS_PATH,
    dimensions: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    workers: int = 4,
) -> dict[str, list[float]]:
    """Train node2vec embeddings on the KG. ~30–60 min on 8 CPU cores."""
    try:
        from node2vec import Node2Vec
    except ImportError:
        logger.error("node2vec not installed. Run: pip install node2vec")
        return {}

    logger.info(
        f"Training node2vec: dim={dimensions}, walk_length={walk_length}, "
        f"num_walks={num_walks}, workers={workers}"
    )

    n2v   = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        quiet=False,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)

    embeddings = {}
    for node in graph.nodes():
        try:
            embeddings[str(node)] = model.wv[str(node)].tolist()
        except KeyError:
            pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(embeddings, f)
    logger.info(f"Embeddings saved: {len(embeddings)} nodes → {output_path}")
    return embeddings


def evaluate_embeddings(
    embeddings: dict,
    ground_truth_positives: list[tuple[str, str]],
    ground_truth_negatives: list[tuple[str, str]],
) -> dict:
    """Evaluate embedding quality. Target: AUROC > 0.78."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {}

    layer  = KGEmbeddingLayer()
    layer._embeddings = embeddings
    scores, labels = [], []

    for drug_id, disease_id in ground_truth_positives:
        d  = layer._get_embedding(drug_id, "Drug")
        di = layer._get_embedding(disease_id, "Disease")
        if d and di:
            scores.append(layer._cosine_similarity(d, di))
            labels.append(1)

    for drug_id, disease_id in ground_truth_negatives:
        d  = layer._get_embedding(drug_id, "Drug")
        di = layer._get_embedding(disease_id, "Disease")
        if d and di:
            scores.append(layer._cosine_similarity(d, di))
            labels.append(0)

    if len(set(labels)) < 2:
        return {"error": "Need both positive and negative pairs"}

    auroc     = roc_auc_score(labels, scores)
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    return {
        "auroc":                      round(auroc, 4),
        "n_positives_evaluated":      len(pos_scores),
        "n_negatives_evaluated":      len(neg_scores),
        "avg_positive_similarity":    round(np.mean(pos_scores), 4) if pos_scores else None,
        "avg_negative_similarity":    round(np.mean(neg_scores), 4) if neg_scores else None,
        "passes_threshold":           auroc > 0.78,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "build":
        graph = build_hetionet_graph()
        if graph:
            # Install node2vec first if needed: pip install node2vec
            try:
                embeddings = train_node2vec_embeddings(graph)
                print(f"Built embeddings for {len(embeddings)} nodes")
            except ImportError:
                print("Install node2vec first: pip install node2vec")
                print("Then re-run: python -m src.layers.layer3_kg_embedding build")

    elif command == "evaluate":
        if not os.path.exists(EMBEDDINGS_PATH):
            print(f"No embeddings at {EMBEDDINGS_PATH}. Run 'build' first.")
            sys.exit(1)
        with open(EMBEDDINGS_PATH) as f:
            embeddings = json.load(f)
        positives = [
            ("CHEMBL1520", "ORPHA:422"),
            ("CHEMBL53463", "ORPHA:77"),
        ]
        negatives = [
            ("CHEMBL192", "ORPHA:101435"),
        ]
        results = evaluate_embeddings(embeddings, positives, negatives)
        for k, v in results.items():
            print(f"  {k}: {v}")

    else:
        print("Usage: python -m src.layers.layer3_kg_embedding [build|evaluate]")
        print()
        print("Prerequisites:")
        print("  pip install node2vec")
        print()
        print("The build command will:")
        print("  1. Download Hetionet (~250MB) from GitHub/Zenodo")
        print("  2. Build a NetworkX graph (47K nodes, 2.2M edges)")
        print("  3. Train node2vec embeddings (~30-60 min)")
        print("  4. Save to data/processed/kg_embeddings.json")