"""
tests/integration/test_known_pairs.py

Integration tests against known drug-disease pairs.
These tests call real APIs — run them separately from unit tests.

Run with: pytest tests/integration/ -v --timeout=120

KNOWN POSITIVES must score above POSITIVE_THRESHOLD.
KNOWN NEGATIVES must score below NEGATIVE_THRESHOLD.

If a known positive scores too low, the bio team must investigate:
  - Is the correct drug ID being used? (Check ChEMBL)
  - Is the correct disease ID being used? (Check Orphanet)
  - Which layer is failing to capture the signal?
  - Is this a data source coverage gap?
"""

import pytest
from src.scoring.engine import ScoringEngine, EngineConfig

POSITIVE_THRESHOLD = 0.45   # Known positives should score above this
NEGATIVE_THRESHOLD = 0.35   # Known negatives should score below this

# These pairs are well-established and should score reliably
KNOWN_POSITIVES = [
    {
        "drug_id": "CHEMBL1520",
        "drug_name": "Sildenafil",
        "disease_id": "ORPHA:422",
        "disease_name": "Pulmonary arterial hypertension",
        "rationale": "PDE5 inhibitor approved for PAH (Revatio, FDA 2005). "
                     "Strong target overlap and clinical trial evidence expected.",
    },
    {
        "drug_id": "CHEMBL53463",
        "drug_name": "Miglustat",
        "disease_id": "ORPHA:77",
        "disease_name": "Gaucher disease type 1",
        "rationale": "Substrate reduction therapy approved by EMA 2002 for Gaucher Type 1.",
    },
]

KNOWN_NEGATIVES = [
    {
        "drug_id": "CHEMBL192",
        "drug_name": "Imatinib",
        "disease_id": "ORPHA:101435",
        "disease_name": "Autosomal recessive primary microcephaly",
        "rationale": "No mechanistic relationship. Imatinib targets BCR-ABL; "
                     "MCPH is a neuronal development disorder.",
    },
]


@pytest.fixture(scope="module")
def engine():
    config = EngineConfig(
        # Disable expensive layers for integration test speed
        enable_layer1b_network_proximity=False,  # requires local PPI graph
        enable_layer_pgx=False,                  # requires PharmGKB key
    )
    return ScoringEngine.build(config)


@pytest.mark.integration
class TestKnownPairs:

    @pytest.mark.parametrize("pair_info", KNOWN_POSITIVES)
    def test_known_positive_scores_above_threshold(self, engine, pair_info):
        pair = engine.score_pair(
            drug_id=pair_info["drug_id"],
            drug_name=pair_info["drug_name"],
            disease_id=pair_info["disease_id"],
            disease_name=pair_info["disease_name"],
        )
        assert not pair.flags.is_disqualified, (
            f"{pair_info['drug_name']}×{pair_info['disease_name']} was disqualified: "
            f"{pair.flags.disqualify_reason}"
        )
        assert pair.composite_score is not None
        assert pair.composite_score >= POSITIVE_THRESHOLD, (
            f"{pair_info['drug_name']}×{pair_info['disease_name']} scored "
            f"{pair.composite_score:.3f} (threshold: {POSITIVE_THRESHOLD})\n"
            f"Rationale: {pair_info['rationale']}\n"
            f"Scores: {pair.scores.__dict__}"
        )

    @pytest.mark.parametrize("pair_info", KNOWN_NEGATIVES)
    def test_known_negative_scores_below_threshold(self, engine, pair_info):
        pair = engine.score_pair(
            drug_id=pair_info["drug_id"],
            drug_name=pair_info["drug_name"],
            disease_id=pair_info["disease_id"],
            disease_name=pair_info["disease_name"],
        )
        # Negatives might be disqualified (that's fine) or score low
        if pair.flags.is_disqualified:
            return  # Disqualified = correctly rejected

        assert pair.composite_score is not None
        assert pair.composite_score < NEGATIVE_THRESHOLD, (
            f"{pair_info['drug_name']}×{pair_info['disease_name']} scored "
            f"{pair.composite_score:.3f} — higher than expected for a negative control "
            f"(threshold: {NEGATIVE_THRESHOLD})\n"
            f"Rationale: {pair_info['rationale']}\n"
            f"Scores: {pair.scores.__dict__}"
        )


@pytest.mark.integration
class TestRegressionScores:
    """
    Regression tests: prevent accidental score changes when refactoring.
    Scores are captured once and locked. If a score changes unexpectedly,
    the bio team must verify the change is intentional before merging.
    """

    BASELINE_SCORES = {
        # Format: "drug_id×disease_id": expected_composite_score (±0.02 tolerance)
        # Add entries here after each validated engine run.
        # "CHEMBL1520×ORPHA:422": 0.621,  # Example — populate after first run
    }

    @pytest.mark.parametrize("key,expected", BASELINE_SCORES.items())
    def test_score_unchanged(self, engine, key, expected):
        drug_id, disease_id = key.split("×")
        # Look up name from known pairs (simplified)
        pair = engine.score_pair(
            drug_id=drug_id,
            drug_name=drug_id,         # placeholder — use actual names in production
            disease_id=disease_id,
            disease_name=disease_id,
        )
        assert pair.composite_score is not None
        assert abs(pair.composite_score - expected) < 0.02, (
            f"Score regression for {key}: expected {expected:.3f}, "
            f"got {pair.composite_score:.3f}. "
            f"If this is intentional, update BASELINE_SCORES."
        )
