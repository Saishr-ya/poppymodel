"""
src/scoring/engine.py

Main ScoringEngine.

Execution order fix: Layer 6 (Business) previously ran first so it could
disqualify patent conflicts cheaply. But _score_clinical_adoption and
_score_speed read pair.scores.case_report_count and
pair.scores.clinical_trial_evidence, which are populated by Layer 5
(Literature). Running Layer 6 before Layer 5 meant those subscores always
saw None and defaulted to 1/5.

New order:
  1. Layer 4 ADMET disqualifiers  — fast, cheap, hard disqualifiers first
  2. Layer 1A Target Overlap      — biological signal
  3. Layer 1B Network Proximity   — biological signal (expensive)
  4. Layer 5 Literature           — populates ct_evidence + case_report_count
  5. Layer 6 Business             — NOW can read Layer 5 outputs correctly
  6. PGx                          — penalty layer, runs last

Patent-conflict disqualification (the original reason Layer 6 ran first)
is handled by a patent-only pre-check that still runs early.
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
from src.layers.layer_ddi import DDILayer

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    disgenet_api_key: str = ""
    openfda_api_key: str = ""

    enable_layer1_target_overlap: bool = True
    enable_layer1b_network_proximity: bool = True
    enable_layer4_admet: bool = True
    enable_layer5_literature: bool = True
    enable_layer6_business: bool = True
    enable_layer_pgx: bool = True
    enable_layer_ddi: bool = True

    business_overrides: dict[str, BusinessScoreConfig] = field(default_factory=dict)
    composite_weights: Optional[dict[str, float]] = None

    min_business_score: int = 24
    top_n_candidates: int = 20


class ScoringEngine:

    def __init__(self, config: EngineConfig):
        self.config = config
        self._layers = self._build_layers()
        self.scorer  = WeightedCompositeScorer(weights=config.composite_weights)
        self.ranker  = CandidateRanker(scorer=self.scorer)

    @classmethod
    def build(cls, config: Optional[EngineConfig] = None) -> "ScoringEngine":
        return cls(config or EngineConfig())

    def _build_layers(self) -> list:
        """
        Build layers in execution order.

        Order matters:
          Layer 4 first  — cheap hard disqualifiers (hERG, bioavailability)
          Layer 1A/1B    — biological signals
          Layer 5        — literature signals (must precede Layer 6)
          Layer 6        — business scoring (reads Layer 5 outputs)
          PGx            — penalty, runs last
        """
        cfg       = self.config
        layer_cfg = {"disgenet_api_key": cfg.disgenet_api_key}
        layers    = []

        if cfg.enable_layer4_admet:
            layers.append(ADMETLayer(config=layer_cfg))
            logger.info("Layer 4 (ADMET) enabled")

        if cfg.enable_layer1_target_overlap:
            layers.append(TargetOverlapLayer(config=layer_cfg))
            logger.info("Layer 1A (Target Overlap) enabled")

        if cfg.enable_layer1b_network_proximity:
            layers.append(NetworkProximityLayer(config=layer_cfg))
            logger.info("Layer 1B (Network Proximity) enabled")

        if cfg.enable_layer5_literature:
            layers.append(LiteratureLayer(config=layer_cfg))
            logger.info("Layer 5 (Literature) enabled")

        # Layer 6 runs AFTER Layer 5 so _score_clinical_adoption and
        # _score_speed can read case_report_count and clinical_trial_evidence
        if cfg.enable_layer6_business:
            layers.append(
                BusinessLayer(config=layer_cfg, overrides=cfg.business_overrides)
            )
            logger.info("Layer 6 (Business) enabled")

        if cfg.enable_layer_ddi:
            layers.append(DDILayer(config=layer_cfg))
            logger.info("DDI layer enabled")
        if cfg.enable_layer_pgx:
            layers.append(SouthAsianPGxLayer(config=layer_cfg))
            logger.info("PGx (South Asian) layer enabled")

        logger.info(f"ScoringEngine initialised with {len(layers)} active layers")
        return layers

    def score_pair(
        self,
        drug_id: str,
        drug_name: str,
        disease_id: str,
        disease_name: str,
    ) -> CandidatePair:
        pair = CandidatePair(
            drug_id=drug_id,
            drug_name=drug_name,
            disease_id=disease_id,
            disease_name=disease_name,
        )

        logger.info(f"Scoring: {drug_name} × {disease_name}")

        for layer in self._layers:
            pair = layer.run(pair)
            if pair.flags.is_disqualified:
                logger.info(
                    f"DISQUALIFIED after {layer.layer_name}: "
                    f"{pair.flags.disqualify_reason}"
                )
                break

        self.scorer.compute(pair)
        return pair

    def score_batch(
        self,
        drug_disease_pairs: list[dict],
        show_progress: bool = True,
    ) -> list[CandidatePair]:
        pairs    = []
        iterator = (
            tqdm(drug_disease_pairs, desc="Scoring pairs")
            if show_progress else drug_disease_pairs
        )

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
                    f"Unexpected error scoring "
                    f"{entry.get('drug_id')}×{entry.get('disease_id')}: {e}",
                    exc_info=True,
                )

        logger.info(
            f"Batch scoring complete: {len(pairs)} pairs scored, "
            f"{sum(1 for p in pairs if p.flags.is_disqualified)} disqualified"
        )
        return self.ranker.rank(pairs)

    def top_candidates(self, pairs: list[CandidatePair]) -> list[CandidatePair]:
        return self.ranker.top_candidates(
            pairs,
            n=self.config.top_n_candidates,
            min_business_score=self.config.min_business_score,
        )

    def report(self, pairs: list[CandidatePair]) -> str:
        lines = [
            "=" * 70,
            "DRUG REPURPOSING ENGINE — CANDIDATE REPORT",
            "=" * 70, "",
        ]

        eligible     = [p for p in pairs if not p.flags.is_disqualified]
        disqualified = [p for p in pairs if p.flags.is_disqualified]

        lines += [
            f"Total pairs scored:   {len(pairs)}",
            f"Disqualified:         {len(disqualified)}",
            f"Eligible candidates:  {len(eligible)}",
            f"Business threshold:   ≥ {self.config.min_business_score}/30",
            "", "─" * 70, "TOP CANDIDATES", "─" * 70,
        ]

        for i, pair in enumerate(self.top_candidates(pairs), 1):
            s = pair.scores
            lines.append(
                f"\n#{i} [{pair.composite_score:.3f}] "
                f"{pair.drug_name} × {pair.disease_name}"
            )
            lines.append(f"     IDs: {pair.drug_id} × {pair.disease_id}")
            lines.append(f"     Business total: {s.business_total}/30")
            if s.target_overlap_jaccard is not None:
                lines.append(f"     Target overlap (Jaccard): {s.target_overlap_jaccard:.4f}")
            lines.append(f"     Network proximity: {s.network_proximity or 'N/A'}")
            lines.append(f"     ClinicalTrials evidence: {s.clinical_trial_evidence or 0}/5")
            lines.append(f"     SA PGx risk: {s.pgx_metabolizer_risk_score or 'N/A'}")
            if pair.flags.pgx_poor_metabolizer_risk_high:
                lines.append("     ⚠ PGx genotyping required in trial protocol")

        lines += ["\n" + "─" * 70, "DISQUALIFIED (summary)", "─" * 70]
        for pair in disqualified[:10]:
            lines.append(
                f"  {pair.drug_name} × {pair.disease_name}: "
                f"{pair.flags.disqualify_reason}"
            )
        if len(disqualified) > 10:
            lines.append(f"  … and {len(disqualified) - 10} more")

        return "\n".join(lines)