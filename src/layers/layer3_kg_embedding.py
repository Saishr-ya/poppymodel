"""
src/layers/layer3_kg_embedding.py

Layer 3 — Knowledge Graph Embedding (State of the Art).

Based on: Himmelstein et al. 2017 (Hetionet, eLife PMC5640425) and
the DREAMwalk paper (Nature Communications 2023, PMC10264374).

Core idea: Build a biomedical knowledge graph where every entity
(drug, disease, gene, pathway, phenotype) is embedded as a vector.
Drug-disease repurposing = predicting missing Drug→treats→Disease edges
by scoring cosine similarity of their learned embeddings.

Why this outperforms simple target overlap:
  A drug doesn't need to directly hit a disease gene. It might:
  - Target a gene that modifies a disease-causing protein (2 hops)
  - Affect a pathway that compensates for the disease mechanism (3 hops)
  - Have phenotypic effects that partially address disease symptoms
  GNNs and KG embeddings capture all of these indirect paths automatically.

Graph schema (per the spec):
  Nodes:
    Drug        (from ChEMBL)
    Disease     (from Orphanet/OMIM)
    Gene/Protein (from UniProt)
    Pathway     (from Reactome)
    Phenotype   (from HPO)

  Edges:
    Drug      → targets            → Protein
    Protein   → involved_in        → Pathway
    Gene      → causes             → Disease
    Disease   → has_phenotype      → Phenotype
    Drug      → treats             → Disease   (known — for training)
    Protein   → interacts_with     → Protein   (PPI, from STRING)

Implementation:
  Phase 1 (now):     node2vec — random walk embeddings, fast, no GPU needed
  Phase 2 (month 4+): TransE or ComplEx — translational embeddings, better accuracy
  Phase 3 (month 5+): Graph Neural Network (PyTorch Geometric) — state of the art

Setup time: 2–4 hours to build the graph, ~30 min to train node2vec.
Target: AUROC > 0.78 on held-out ground truth pairs.

References:
  - Himmelstein 2017 eLife — Hetionet (use this graph directly as your starting point)
  - DREAMwalk 2023 Nat Comms — multi-layer KG random walks, AUROC 0.938
  - node2vec: https://snap.stanford.edu/node2vec/
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

# Hetionet direct download (skip building from scratch for Phase 1)
# The spec cites this: Himmelstein et al. 2017, PMC5640425
HETIONET_NODES_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0-nodes.json"
HETIONET_EDGES_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0-edges.sif.gz"


class KGEmbeddingLayer(BaseLayer):
    """
    Layer 3 — Knowledge graph embedding-based drug-disease scoring.

    Scores:
        pair.scores.kg_embedding_cosine  (0–1; higher = more similar embeddings)

    Validation target: AUROC > 0.78 on held-out ground truth pairs.
    If below 0.78: check graph connectivity, try TransE instead of node2vec.
    """

    layer_name = "layer3_kg_embedding"
    version = "1.0"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self._embeddings: Optional[dict[str, list[float]]] = None
        self._node_map: Optional[dict[str, str]] = None  # entity_id → embedding key

    @property
    def embeddings(self) -> Optional[dict[str, list[float]]]:
        """Lazy-load embeddings from disk."""
        if self._embeddings is None:
            self._embeddings = self._load_embeddings()
        return self._embeddings

    def _load_embeddings(self) -> Optional[dict[str, list[float]]]:
        """Load pre-trained embeddings from disk."""
        if not os.path.exists(EMBEDDINGS_PATH):
            logger.warning(
                f"KG embeddings not found at {EMBEDDINGS_PATH}. "
                "Run: python -m src.layers.layer3_kg_embedding build "
                "to generate embeddings. Layer 3 will be skipped until then."
            )
            return None

        with open(EMBEDDINGS_PATH) as f:
            data = json.load(f)
        logger.info(f"KG embeddings loaded: {len(data)} nodes")
        return data

    def score(self, pair: CandidatePair) -> CandidatePair:
        if self.embeddings is None:
            logger.debug(f"[{self.layer_name}] No embeddings — skipping")
            return pair

        # Look up drug and disease embeddings
        drug_emb = self._get_embedding(pair.drug_id, "Drug")
        disease_emb = self._get_embedding(pair.disease_id, "Disease")

        if drug_emb is None:
            logger.debug(f"[{self.layer_name}] No embedding for drug {pair.drug_id}")
            return pair
        if disease_emb is None:
            logger.debug(f"[{self.layer_name}] No embedding for disease {pair.disease_id}")
            return pair

        cosine = self._cosine_similarity(drug_emb, disease_emb)
        pair.scores.kg_embedding_cosine = float(cosine)

        logger.debug(
            f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
            f"KG cosine={cosine:.4f}"
        )
        return pair

    def _get_embedding(self, entity_id: str, entity_type: str) -> Optional[list[float]]:
        """Look up embedding for an entity by ID, trying multiple key formats."""
        if self.embeddings is None:
            return None
        # Try multiple key formats (Hetionet uses different ID formats)
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
        va = np.array(a, dtype=float)
        vb = np.array(b, dtype=float)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))


# ── Graph building and embedding training ─────────────────────────────────────

def build_hetionet_graph(output_dir: str = "data/raw/hetionet") -> Optional[object]:
    """
    Download and load Hetionet — the most complete public biomedical KG.

    Himmelstein et al. 2017 built Hetionet with:
        47,031 nodes (diseases, genes, compounds, pathways, etc.)
        2,250,197 edges across 24 edge types

    This is your starting point — add your own edges on top as your
    engine matures. Using Hetionet directly saves 3–4 weeks of graph-building.

    Returns:
        NetworkX graph or None if download failed.
    """
    try:
        import networkx as nx
        import requests
        import gzip
    except ImportError as e:
        logger.error(f"Required package missing: {e}. Run: pip install networkx requests")
        return None

    os.makedirs(output_dir, exist_ok=True)
    edges_path = os.path.join(output_dir, "hetionet-v1.0-edges.sif.gz")
    nodes_path = os.path.join(output_dir, "hetionet-v1.0-nodes.json")

    # Download nodes
    if not os.path.exists(nodes_path):
        logger.info("Downloading Hetionet nodes...")
        try:
            r = requests.get(HETIONET_NODES_URL, timeout=60)
            r.raise_for_status()
            with open(nodes_path, "wb") as f:
                f.write(r.content)
            logger.info(f"Saved Hetionet nodes: {nodes_path}")
        except Exception as e:
            logger.error(f"Failed to download Hetionet nodes: {e}")
            return None

    # Download edges
    if not os.path.exists(edges_path):
        logger.info("Downloading Hetionet edges (~500MB)...")
        try:
            r = requests.get(HETIONET_EDGES_URL, stream=True, timeout=120)
            r.raise_for_status()
            with open(edges_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Saved Hetionet edges: {edges_path}")
        except Exception as e:
            logger.error(f"Failed to download Hetionet edges: {e}")
            return None

    # Build NetworkX graph
    logger.info("Building NetworkX graph from Hetionet...")
    G = nx.Graph()

    # Load nodes with type metadata
    with open(nodes_path) as f:
        nodes_data = json.load(f)
    for node in nodes_data.get("nodes", []):
        G.add_node(node["identifier"], name=node.get("name", ""), kind=node.get("kind", ""))

    # Load edges
    with gzip.open(edges_path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                src, rel, tgt = parts[0], parts[1], parts[2]
                G.add_edge(src, tgt, relation=rel)

    logger.info(
        f"Hetionet graph built: {G.number_of_nodes()} nodes, "
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
    """
    Train node2vec embeddings on the KG.

    node2vec learns vector representations of graph nodes via
    biased random walks. Nodes with similar graph neighborhoods
    get similar embeddings.

    Parameters:
        dimensions:  Embedding size. 128 is standard; increase to 256 for more
                     complex graphs but training takes 2× longer.
        walk_length: Steps per random walk. 80 is the node2vec paper default.
        num_walks:   Walks per node. More = better embeddings, slower training.
        workers:     CPU threads. Use all available cores.

    Runtime: ~30 min for Hetionet on 8 CPU cores.
    Memory: ~4GB RAM during training.

    Returns:
        dict: {node_id: embedding_vector}
    """
    try:
        from node2vec import Node2Vec
    except ImportError:
        logger.error("node2vec not installed. Run: pip install node2vec")
        return {}

    logger.info(
        f"Training node2vec: dim={dimensions}, walk_length={walk_length}, "
        f"num_walks={num_walks}, workers={workers}"
    )

    n2v = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        quiet=False,
    )

    model = n2v.fit(window=10, min_count=1, batch_words=4)

    # Extract embeddings to dict
    embeddings = {}
    for node in graph.nodes():
        try:
            embeddings[str(node)] = model.wv[str(node)].tolist()
        except KeyError:
            pass

    # Save to disk
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
    """
    Evaluate embedding quality on ground truth drug-disease pairs.
    Target: AUROC > 0.78.

    Args:
        ground_truth_positives: List of (drug_id, disease_id) known positive pairs
        ground_truth_negatives: List of (drug_id, disease_id) known negative pairs

    Returns:
        {auroc, avg_positive_similarity, avg_negative_similarity}
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.error("scikit-learn not installed")
        return {}

    layer = KGEmbeddingLayer()
    layer._embeddings = embeddings

    scores = []
    labels = []

    for drug_id, disease_id in ground_truth_positives:
        d_emb = layer._get_embedding(drug_id, "Drug")
        dis_emb = layer._get_embedding(disease_id, "Disease")
        if d_emb and dis_emb:
            scores.append(layer._cosine_similarity(d_emb, dis_emb))
            labels.append(1)

    for drug_id, disease_id in ground_truth_negatives:
        d_emb = layer._get_embedding(drug_id, "Drug")
        dis_emb = layer._get_embedding(disease_id, "Disease")
        if d_emb and dis_emb:
            scores.append(layer._cosine_similarity(d_emb, dis_emb))
            labels.append(0)

    if len(set(labels)) < 2:
        return {"error": "Need both positive and negative pairs for AUROC"}

    auroc = roc_auc_score(labels, scores)
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    result = {
        "auroc": round(auroc, 4),
        "n_positives_evaluated": len(pos_scores),
        "n_negatives_evaluated": len(neg_scores),
        "avg_positive_similarity": round(np.mean(pos_scores), 4) if pos_scores else None,
        "avg_negative_similarity": round(np.mean(neg_scores), 4) if neg_scores else None,
        "passes_threshold": auroc > 0.78,
    }

    logger.info(
        f"KG Embedding evaluation: AUROC={auroc:.4f} "
        f"({'PASS' if result['passes_threshold'] else 'FAIL — debug graph structure'})"
    )
    return result


if __name__ == "__main__":
    """
    Build and train the KG embeddings.

    Usage:
        python -m src.layers.layer3_kg_embedding build
        python -m src.layers.layer3_kg_embedding evaluate
    """
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "build":
        graph = build_hetionet_graph()
        if graph:
            embeddings = train_node2vec_embeddings(graph)
            print(f"Built embeddings for {len(embeddings)} nodes")

    elif command == "evaluate":
        if not os.path.exists(EMBEDDINGS_PATH):
            print(f"No embeddings found at {EMBEDDINGS_PATH}. Run 'build' first.")
            sys.exit(1)

        with open(EMBEDDINGS_PATH) as f:
            embeddings = json.load(f)

        # Use seed ground truth pairs
        positives = [
            ("CHEMBL1520", "ORPHA:422"),    # sildenafil × PAH
            ("CHEMBL53463", "ORPHA:77"),    # miglustat × Gaucher
        ]
        negatives = [
            ("CHEMBL192", "ORPHA:101435"),  # imatinib × microcephaly
        ]

        results = evaluate_embeddings(embeddings, positives, negatives)
        for k, v in results.items():
            print(f"  {k}: {v}")

    else:
        print("Usage: python -m src.layers.layer3_kg_embedding [build|evaluate]")