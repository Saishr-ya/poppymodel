"""
src/scoring/engine.py

Main ScoringEngine — orchestrates all layers in order, applies disqualifiers,
and returns ranked CandidatePairs.

Usage:
    engine = ScoringEngine.build(config)
    pair = engine.score_pair("CHEMBL192", "Chronic myeloid leukemia", "ORPHA:355")
    pairs = engine.score_batch(drug_disease_list)

Layer execution order follows the spec:
  1. Layer 6 (Business) — fast, disqualifies unpatentable candidates early
  2. Layer 4 ADMET disqualifiers — fast, cheap API calls
  3. Layer 1A Target Overlap — moderate cost
  4. Layer 1B Network Proximity — expensive (graph traversal)
  5. Layer 5 Literature — moderate cost (PubMed + ClinicalTrials APIs)
  6. PGx layer — moderate cost (PharmGKB)
  7. Composite scoring
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm

from src.scoring.candidate import CandidatePair
from src.scoring.composite import WeightedCompositeScorer, CandidateRanker
from src.layers.layer1_target_overlap import TargetOverlapLayer
from src.layers.layer1b_network_proximity import NetworkProximityLayer
from src.layers.layer4_admet import ADMETLayer
from src.layers.layer5_literature import LiteratureLayer
from src.layers.layer6_business import BusinessLayer, BusinessScoreConfig
from src.layers.layer_pgx import SouthAsianPGxLayer

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """
    Master configuration for the ScoringEngine.
    Passed to each layer during initialization.
    """

    # API keys
    disgenet_api_key: str = ""
    openfda_api_key: str = ""

    # Layer enable/disable (for iterative development — enable layers as built)
    enable_layer1_target_overlap: bool = True
    enable_layer1b_network_proximity: bool = True
    enable_layer4_admet: bool = True
    enable_layer5_literature: bool = True
    enable_layer6_business: bool = True
    enable_layer_pgx: bool = True

    # Business scoring overrides (drug_id×disease_id → BusinessScoreConfig)
    business_overrides: dict[str, BusinessScoreConfig] = field(default_factory=dict)

    # Composite scorer weights
    composite_weights: Optional[dict[str, float]] = None

    # Ranking thresholds
    min_business_score: int = 24
    top_n_candidates: int = 20


class ScoringEngine:
    """
    Central orchestrator for the drug repurposing scoring pipeline.

    Layers execute in defined order. Each layer receives the CandidatePair,
    mutates it, and passes it to the next layer. Disqualified pairs skip
    expensive layers early.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self._layers = self._build_layers()
        self.scorer = WeightedCompositeScorer(weights=config.composite_weights)
        self.ranker = CandidateRanker(scorer=self.scorer)

    @classmethod
    def build(cls, config: Optional[EngineConfig] = None) -> "ScoringEngine":
        """Factory method with sensible defaults."""
        return cls(config or EngineConfig())

    def _build_layers(self) -> list:
        """Initialize all enabled layers in execution order."""
        cfg = self.config
        layer_cfg = {"disgenet_api_key": cfg.disgenet_api_key}
        layers = []

        # Business first — disqualifies patent conflicts cheaply
        if cfg.enable_layer6_business:
            layers.append(
                BusinessLayer(
                    config=layer_cfg,
                    overrides=cfg.business_overrides,
                )
            )
            logger.info("Layer 6 (Business) enabled")

        # ADMET disqualifiers — cheap API calls
        if cfg.enable_layer4_admet:
            layers.append(ADMETLayer(config=layer_cfg))
            logger.info("Layer 4 (ADMET) enabled")

        # Biological signal layers
        if cfg.enable_layer1_target_overlap:
            layers.append(TargetOverlapLayer(config=layer_cfg))
            logger.info("Layer 1A (Target Overlap) enabled")

        if cfg.enable_layer1b_network_proximity:
            layers.append(NetworkProximityLayer(config=layer_cfg))
            logger.info("Layer 1B (Network Proximity) enabled")

        if cfg.enable_layer5_literature:
            layers.append(LiteratureLayer(config=layer_cfg))
            logger.info("Layer 5 (Literature) enabled")

        if cfg.enable_layer_pgx:
            layers.append(SouthAsianPGxLayer(config=layer_cfg))
            logger.info("PGx (South Asian) layer enabled")

        logger.info(f"ScoringEngine initialized with {len(layers)} active layers")
        return layers

    def score_pair(
        self,
        drug_id: str,
        drug_name: str,
        disease_name: str,
        disease_id: str,
    ) -> CandidatePair:
        """
        Score a single drug-disease pair through all enabled layers.

        Args:
            drug_id:      ChEMBL ID (e.g., "CHEMBL192")
            drug_name:    Human-readable drug name (e.g., "Imatinib")
            disease_name: Human-readable disease name
            disease_id:   Orphanet or OMIM ID (e.g., "ORPHA:355")

        Returns:
            Scored CandidatePair with composite_score set.
        """
        pair = CandidatePair(
            drug_id=drug_id,
            drug_name=drug_name,
            disease_id=disease_id,
            disease_name=disease_name,
        )

        logger.info(f"Scoring: {drug_name} × {disease_name}")

        for layer in self._layers:
            pair = layer.run(pair)
            # Stop early if disqualified by a hard flag
            if pair.flags.is_disqualified:
                logger.info(
                    f"DISQUALIFIED after {layer.layer_name}: {pair.flags.disqualify_reason}"
                )
                break

        self.scorer.compute(pair)
        return pair

    def score_batch(
        self,
        drug_disease_pairs: list[dict],
        show_progress: bool = True,
    ) -> list[CandidatePair]:
        """
        Score a batch of drug-disease pairs.

        Args:
            drug_disease_pairs: List of dicts with keys:
                drug_id, drug_name, disease_id, disease_name

        Returns:
            Ranked list of CandidatePairs (top candidates first).
        """
        pairs = []
        iterator = tqdm(drug_disease_pairs, desc="Scoring pairs") if show_progress \
            else drug_disease_pairs

        for entry in iterator:
            try:
                pair = self.score_pair(
                    drug_id=entry["drug_id"],
                    drug_name=entry["drug_name"],
                    disease_id=entry["disease_id"],
                    disease_name=entry["disease_name"],
                )
                pairs.append(pair)
            except Exception as e:
                logger.error(
                    f"Unexpected error scoring {entry.get('drug_id')}×{entry.get('disease_id')}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"Batch scoring complete: {len(pairs)} pairs scored, "
            f"{sum(1 for p in pairs if p.flags.is_disqualified)} disqualified"
        )

        return self.ranker.rank(pairs)

    def top_candidates(self, pairs: list[CandidatePair]) -> list[CandidatePair]:
        """Filter ranked pairs to top N meeting the minimum business score."""
        return self.ranker.top_candidates(
            pairs,
            n=self.config.top_n_candidates,
            min_business_score=self.config.min_business_score,
        )

    def report(self, pairs: list[CandidatePair]) -> str:
        """
        Generate a human-readable summary report for the top candidates.
        For full PDF reports, run the output pipeline.
        """
        lines = ["=" * 70, "DRUG REPURPOSING ENGINE — CANDIDATE REPORT", "=" * 70, ""]

        eligible = [p for p in pairs if not p.flags.is_disqualified]
        disqualified = [p for p in pairs if p.flags.is_disqualified]

        lines.append(f"Total pairs scored:   {len(pairs)}")
        lines.append(f"Disqualified:         {len(disqualified)}")
        lines.append(f"Eligible candidates:  {len(eligible)}")
        lines.append(f"Business threshold:   ≥ {self.config.min_business_score}/30")
        lines.append("")
        lines.append("─" * 70)
        lines.append("TOP CANDIDATES")
        lines.append("─" * 70)

        top = self.top_candidates(pairs)
        for i, pair in enumerate(top, 1):
            s = pair.scores
            lines.append(f"\n#{i} [{pair.composite_score:.3f}] {pair.drug_name} × {pair.disease_name}")
            lines.append(f"     IDs: {pair.drug_id} × {pair.disease_id}")
            lines.append(f"     Business total: {s.business_total}/30")
            lines.append(f"     Target overlap (Jaccard): {s.target_overlap_jaccard or 'N/A':.4f}" if s.target_overlap_jaccard else f"     Target overlap (Jaccard): N/A")
            lines.append(f"     Network proximity: {s.network_proximity or 'N/A'}")
            lines.append(f"     ClinicalTrials evidence: {s.clinical_trial_evidence or 0}/5")
            lines.append(f"     SA PGx risk: {s.pgx_metabolizer_risk_score or 'N/A'}")
            if pair.flags.pgx_poor_metabolizer_risk_high:
                lines.append(f"     ⚠ PGx genotyping required in trial protocol")
            if pair.flags.pediatric_formulation_needed:
                lines.append(f"     ⚠ Pediatric formulation needed")
            if pair.flags.polymorph_risk:
                lines.append(f"     ⚠ Polymorphism risk — verify CMO crystal form")

        lines.append("\n" + "─" * 70)
        lines.append("DISQUALIFIED CANDIDATES (summary)")
        lines.append("─" * 70)
        for pair in disqualified[:10]:
            lines.append(f"  {pair.drug_name} × {pair.disease_name}: {pair.flags.disqualify_reason}")
        if len(disqualified) > 10:
            lines.append(f"  … and {len(disqualified) - 10} more")

        return "\n".join(lines)
