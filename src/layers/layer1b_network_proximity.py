"""
src/layers/layer1b_network_proximity.py

Layer 1B — Network Proximity Score (Guney et al. 2016, Nature Communications).
PMC4718842 — the method used by BenevolentAI and Recursion as a core signal.

Key insight: A drug target doesn't need to directly overlap with disease genes.
If drug targets sit within ~2 hops of disease genes in the human protein-protein
interactome (PPI), there is likely a biologically meaningful indirect relationship.

This is more powerful than Jaccard because it captures indirect relationships.

Setup (one-time, ~30 min):
    1. Download STRING DB: https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
    2. Run: python -m src.graph.ppi_network build --input data/raw/string_db/9606.protein.links.v12.0.txt.gz

Bio team:
    - Proximity score < 2.0 = strong signal (drug targets are biologically close to disease)
    - Proximity score > 3.5 = weak signal
    - Score is null if either drug targets or disease genes are not in the PPI network
      (small proteins, poorly characterized targets)
"""

from __future__ import annotations
import logging
import os
import pickle
from typing import Optional

import networkx as nx
import numpy as np

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.chembl_client import ChEMBLClient
from src.ingestion.disgenet_client import DisGeNETClient

logger = logging.getLogger(__name__)

# Singleton PPI graph — loaded once, shared across all scoring runs
_PPI_GRAPH: Optional[nx.Graph] = None
_PPI_GRAPH_PATH = "data/processed/ppi_network.pkl"

STRING_CONFIDENCE_CUTOFF = 400   # Medium confidence (max 1000)


def load_ppi_graph() -> Optional[nx.Graph]:
    """
    Load the human PPI network from disk (NetworkX graph).
    Nodes = UniProt IDs, edges weighted by STRING combined score.

    Returns None if the network file doesn't exist yet.
    Run `python -m src.graph.ppi_network build` to generate it.
    """
    global _PPI_GRAPH
    if _PPI_GRAPH is not None:
        return _PPI_GRAPH

    if os.path.exists(_PPI_GRAPH_PATH):
        logger.info(f"Loading PPI network from {_PPI_GRAPH_PATH}…")
        with open(_PPI_GRAPH_PATH, "rb") as f:
            _PPI_GRAPH = pickle.load(f)
        logger.info(
            f"PPI network loaded: {_PPI_GRAPH.number_of_nodes()} nodes, "
            f"{_PPI_GRAPH.number_of_edges()} edges"
        )
        return _PPI_GRAPH

    logger.warning(
        f"PPI network not found at {_PPI_GRAPH_PATH}. "
        "Network proximity scoring will be skipped. "
        "Build it with: python -m src.graph.ppi_network build"
    )
    return None


def compute_network_proximity(
    drug_targets: set[str],
    disease_genes: set[str],
    graph: nx.Graph,
) -> Optional[float]:
    """
    Compute average minimum shortest path distance between drug targets
    and disease gene modules in the PPI network.

    Method from Guney et al. 2016:
        d(S, T) = mean over all nodes t in T of min_s(d(s, t))
        where S = drug target set, T = disease gene set

    Args:
        drug_targets:   Set of UniProt IDs for drug protein targets.
        disease_genes:  Set of UniProt IDs for disease causal genes.
        graph:          PPI network graph.

    Returns:
        Average shortest path distance (float). Lower = more proximate.
        None if insufficient nodes in network.
    """
    # Filter to nodes actually in the network
    S = drug_targets & set(graph.nodes)
    T = disease_genes & set(graph.nodes)

    if not S:
        logger.debug("No drug targets found in PPI network")
        return None
    if not T:
        logger.debug("No disease genes found in PPI network")
        return None

    distances = []
    for t in T:
        min_dist = float("inf")
        for s in S:
            try:
                d = nx.shortest_path_length(graph, source=s, target=t)
                min_dist = min(min_dist, d)
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue
        if min_dist < float("inf"):
            distances.append(min_dist)

    if not distances:
        return None

    return float(np.mean(distances))


def compute_z_score_proximity(
    drug_targets: set[str],
    disease_genes: set[str],
    graph: nx.Graph,
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> Optional[float]:
    """
    Compute z-score normalized proximity (more statistically rigorous).
    Compares observed proximity to null distribution from random gene sets
    of the same size matched by degree.

    This is the preferred metric for publication-quality results.
    For rapid screening, use compute_network_proximity() instead.

    Args:
        n_permutations: Number of random permutations for null distribution.
        random_seed:    Random seed for reproducibility.
    """
    observed = compute_network_proximity(drug_targets, disease_genes, graph)
    if observed is None:
        return None

    rng = np.random.default_rng(random_seed)
    nodes = list(graph.nodes)
    null_distances = []

    n_targets = len(drug_targets & set(graph.nodes))
    n_disease = len(disease_genes & set(graph.nodes))

    if n_targets == 0 or n_disease == 0:
        return None

    for _ in range(n_permutations):
        rand_targets = set(rng.choice(nodes, size=n_targets, replace=False))
        rand_disease = set(rng.choice(nodes, size=n_disease, replace=False))
        d = compute_network_proximity(rand_targets, rand_disease, graph)
        if d is not None:
            null_distances.append(d)

    if not null_distances:
        return None

    null_mean = np.mean(null_distances)
    null_std = np.std(null_distances)
    if null_std == 0:
        return 0.0

    return float((observed - null_mean) / null_std)


class NetworkProximityLayer(BaseLayer):
    """
    Layer 1B — Network proximity score.

    Scores:
        pair.scores.network_proximity  (average shortest path hops)

    Interpretation:
        < 2.0 hops = strong repurposing signal
        2.0–3.0   = moderate signal
        > 3.0     = weak signal
        None      = targets not in network (log a warning)

    Note: This layer is expensive (O(|S|×|T|) shortest paths).
    Cached results are stored per drug-disease pair in Parquet files.
    """

    layer_name = "layer1b_network_proximity"
    version = "1.0"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.chembl = ChEMBLClient()
        self.disgenet = DisGeNETClient(
            api_key=config.get("disgenet_api_key") if config else None
        )
        self.use_z_score = (config or {}).get("use_z_score", False)
        self._graph = None

    @property
    def graph(self) -> Optional[nx.Graph]:
        if self._graph is None:
            self._graph = load_ppi_graph()
        return self._graph

    def score(self, pair: CandidatePair) -> CandidatePair:
        if self.graph is None:
            logger.warning(
                f"[{self.layer_name}] PPI network not available — skipping proximity scoring"
            )
            return pair

        drug_targets = self.chembl.get_target_uniprot_ids(pair.drug_id)
        disease_genes = self.disgenet.get_disease_uniprot_ids(pair.disease_id)

        if self.use_z_score:
            proximity = compute_z_score_proximity(drug_targets, disease_genes, self.graph)
        else:
            proximity = compute_network_proximity(drug_targets, disease_genes, self.graph)

        pair.scores.network_proximity = proximity

        if proximity is not None:
            signal_label = (
                "STRONG" if proximity < 2.0
                else "MODERATE" if proximity < 3.0
                else "WEAK"
            )
            logger.debug(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"proximity={proximity:.3f} hops [{signal_label}]"
            )
        else:
            logger.warning(
                f"[{self.layer_name}] Could not compute proximity for "
                f"{pair.drug_name}×{pair.disease_name} — targets or genes not in PPI network"
            )

        return pair
