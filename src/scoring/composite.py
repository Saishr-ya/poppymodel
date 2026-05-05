"""
src/scoring/composite.py

Composite Scoring Engine — combines all layer scores into a single 0–1 probability.

Two methods:
  1. WeightedCompositeScorer — simple weighted average (use for rapid screening)
  2. LogisticRegressionScorer — trained on ground truth pairs (use for final ranking)

Architecture note from the spec:
  "A candidate with a perfect network proximity score but an existing patent is worthless.
   A candidate with a moderate biological signal but strong off-label physician adoption
   in India, no prior patent, and a clean 25-year safety record is gold."

This means:
  - Business layer (Layer 6) weighted equally with biology
  - Hard disqualifiers (patent, hERG, FAERS) applied BEFORE composite scoring
  - The engine generates candidates; the business layer tells you which ones to pursue

Weights (tunable — bio team and ML engineer should validate against ground truth):
  target_overlap_jaccard:     0.12
  network_proximity:          0.15   (inverted: lower distance = higher score)
  transcriptomic_reversal_ks: 0.13   (inverted: negative KS = higher score)
  kg_embedding_cosine:        0.10
  admet_composite:            0.10
  literature_cooccurrence:    0.08
  clinical_trial_evidence:    0.12   (normalized from 0–5 scale)
  business_total:             0.20   (normalized from 0–30 scale)

Total: 1.00
"""

from __future__ import annotations
import logging
import math
from typing import Optional

import numpy as np

from src.scoring.candidate import CandidatePair, LayerScores

logger = logging.getLogger(__name__)


# ── Score normalization helpers ────────────────────────────────────────────────

def _normalize_proximity(proximity: Optional[float]) -> Optional[float]:
    """
    Convert network proximity (hops, lower = better) to 0–1 score (higher = better).
    Sigmoid mapping: 0 hops → 1.0, 2 hops → 0.73, 4 hops → 0.27
    """
    if proximity is None:
        return None
    return 1.0 / (1.0 + math.exp(proximity - 2.0))


def _normalize_ks(ks: Optional[float]) -> Optional[float]:
    """
    Convert KS statistic (negative = good reversal) to 0–1 score.
    KS range is approximately [-1, 1].
    -1.0 (perfect reversal) → 1.0
     0.0 (no signal)        → 0.5
    +1.0 (same direction)   → 0.0
    """
    if ks is None:
        return None
    return max(0.0, min(1.0, (-ks + 1.0) / 2.0))


def _normalize_business(total: Optional[int]) -> Optional[float]:
    """Convert business total (0–30) to 0–1 score."""
    if total is None:
        return None
    return min(1.0, max(0.0, total / 30.0))


def _normalize_clinical_trial(score: Optional[int]) -> Optional[float]:
    """Convert clinical trial evidence (0–5) to 0–1 score."""
    if score is None:
        return None
    return score / 5.0


def _normalize_cooccurrence(score: Optional[float], cap: float = 50.0) -> Optional[float]:
    """Normalize pubmed co-occurrence score (log-scale, capped)."""
    if score is None:
        return None
    return min(1.0, math.log1p(score) / math.log1p(cap))


class WeightedCompositeScorer:
    """
    Simple weighted average composite scorer.
    Use for rapid candidate screening before running the ML scorer.

    Handles None scores gracefully: missing layers are excluded from the weighted
    average and their weights are redistributed proportionally.
    """

    DEFAULT_WEIGHTS = {
        "target_overlap_jaccard": 0.12,
        "network_proximity": 0.15,
        "transcriptomic_reversal_ks": 0.13,
        "kg_embedding_cosine": 0.10,
        "admet_composite": 0.10,
        "pubmed_cooccurrence_score": 0.08,
        "clinical_trial_evidence": 0.12,
        "business_total": 0.20,
    }

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total_w:.3f}, not 1.0 — normalizing")
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def compute(self, pair: CandidatePair) -> Optional[float]:
        """
        Compute composite score.
        Returns None if the pair is disqualified (score not meaningful).
        Returns 0–1 float otherwise.
        """
        if pair.flags.is_disqualified:
            logger.debug(
                f"Composite: {pair.drug_name}×{pair.disease_name} is disqualified — skipping"
            )
            pair.composite_score = 0.0
            return 0.0

        s = pair.scores

        # Normalize each raw score to 0–1
        normalized = {
            "target_overlap_jaccard":    s.target_overlap_jaccard,
            "network_proximity":         _normalize_proximity(s.network_proximity),
            "transcriptomic_reversal_ks": _normalize_ks(s.transcriptomic_reversal_ks),
            "kg_embedding_cosine":       s.kg_embedding_cosine,
            "admet_composite":           s.admet_composite,
            "pubmed_cooccurrence_score": _normalize_cooccurrence(s.pubmed_cooccurrence_score),
            "clinical_trial_evidence":   _normalize_clinical_trial(s.clinical_trial_evidence),
            "business_total":            _normalize_business(s.business_total),
        }

        # Weighted average, excluding None values
        total_weight = 0.0
        weighted_sum = 0.0
        for key, value in normalized.items():
            if value is not None:
                w = self.weights.get(key, 0.0)
                weighted_sum += w * value
                total_weight += w

        if total_weight == 0:
            logger.warning(
                f"No scores available for {pair.drug_name}×{pair.disease_name}"
            )
            return None

        composite = weighted_sum / total_weight

        # PGx penalty: reduce score for high-risk candidates in SA population
        pgx_risk = s.pgx_metabolizer_risk_score
        if pgx_risk is not None and pgx_risk > 0.15:
            pgx_penalty = min(0.10, pgx_risk * 0.3)   # max 10% penalty
            composite = max(0.0, composite - pgx_penalty)
            logger.debug(
                f"PGx penalty applied for {pair.drug_name}: -{pgx_penalty:.3f}"
            )

        pair.composite_score = round(composite, 4)
        return pair.composite_score

    def score_explanation(self, pair: CandidatePair) -> dict:
        """Return a human-readable breakdown of the composite score components."""
        s = pair.scores
        normalized = {
            "target_overlap_jaccard": s.target_overlap_jaccard,
            "network_proximity": _normalize_proximity(s.network_proximity),
            "transcriptomic_reversal_ks": _normalize_ks(s.transcriptomic_reversal_ks),
            "kg_embedding_cosine": s.kg_embedding_cosine,
            "admet_composite": s.admet_composite,
            "pubmed_cooccurrence_score": _normalize_cooccurrence(s.pubmed_cooccurrence_score),
            "clinical_trial_evidence": _normalize_clinical_trial(s.clinical_trial_evidence),
            "business_total": _normalize_business(s.business_total),
        }
        return {
            "pair": repr(pair),
            "composite_score": pair.composite_score,
            "business_total_raw": s.business_total,
            "is_disqualified": pair.flags.is_disqualified,
            "disqualify_reason": pair.flags.disqualify_reason,
            "components": {
                k: {
                    "normalized": round(v, 4) if v is not None else None,
                    "weight": self.weights.get(k, 0),
                    "contribution": round(v * self.weights.get(k, 0), 4)
                    if v is not None else None,
                }
                for k, v in normalized.items()
            },
        }


class CandidateRanker:
    """
    Ranks a list of CandidatePairs by composite score.

    Usage:
        ranker = CandidateRanker(scorer=WeightedCompositeScorer())
        ranked = ranker.rank(pairs)
        top_20 = [p for p in ranked if not p.flags.is_disqualified][:20]
    """

    def __init__(self, scorer: Optional[WeightedCompositeScorer] = None):
        self.scorer = scorer or WeightedCompositeScorer()

    def rank(self, pairs: list[CandidatePair]) -> list[CandidatePair]:
        """
        Score all pairs and return sorted by composite score (descending).
        Disqualified pairs are sorted to the end with score = 0.
        """
        for pair in pairs:
            if pair.composite_score is None:
                self.scorer.compute(pair)

        ranked = sorted(
            pairs,
            key=lambda p: (
                0 if p.flags.is_disqualified else 1,    # disqualified to end
                p.composite_score or 0,
            ),
            reverse=True,
        )

        for i, pair in enumerate(ranked):
            pair.rank = i + 1

        logger.info(
            f"Ranked {len(ranked)} pairs. "
            f"Disqualified: {sum(1 for p in ranked if p.flags.is_disqualified)}. "
            f"Top score: {ranked[0].composite_score if ranked else 'N/A'}"
        )

        return ranked

    def top_candidates(
        self,
        pairs: list[CandidatePair],
        n: int = 20,
        min_business_score: int = 24,
    ) -> list[CandidatePair]:
        """
        Return top N candidates meeting the minimum business score threshold.

        Args:
            n:                   Number of top candidates to return.
            min_business_score:  Minimum business total (per spec: 24/30).
        """
        ranked = self.rank(pairs)
        eligible = [
            p for p in ranked
            if not p.flags.is_disqualified
            and (p.scores.business_total is None or p.scores.business_total >= min_business_score)
        ]
        return eligible[:n]
