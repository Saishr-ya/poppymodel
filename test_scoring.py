"""
tests/unit/test_scoring.py

Unit tests for core scoring logic.
Run with: pytest tests/unit/ -v

These tests use no external APIs — all API calls should be mocked.
Bio team: add regression test cases here when you find scoring bugs.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.scoring.candidate import CandidatePair, LayerScores, Flags
from src.scoring.composite import (
    WeightedCompositeScorer,
    CandidateRanker,
    _normalize_proximity,
    _normalize_ks,
    _normalize_business,
    _normalize_clinical_trial,
)
from src.layers.layer1_target_overlap import jaccard_similarity, pathway_overlap_pvalue


# ── Fixtures ────────────────────────────────────────────────────────────────────

def make_pair(drug_id="CHEMBL192", drug_name="Imatinib",
              disease_id="ORPHA:355", disease_name="CML") -> CandidatePair:
    return CandidatePair(
        drug_id=drug_id,
        drug_name=drug_name,
        disease_id=disease_id,
        disease_name=disease_name,
    )


def make_scored_pair(**kwargs) -> CandidatePair:
    """Create a pair with all scores set to reasonable values."""
    pair = make_pair()
    pair.scores.target_overlap_jaccard = kwargs.get("jaccard", 0.3)
    pair.scores.network_proximity = kwargs.get("proximity", 2.0)
    pair.scores.transcriptomic_reversal_ks = kwargs.get("ks", -0.5)
    pair.scores.kg_embedding_cosine = kwargs.get("cosine", 0.6)
    pair.scores.admet_composite = kwargs.get("admet", 0.8)
    pair.scores.pubmed_cooccurrence_score = kwargs.get("cooc", 10.0)
    pair.scores.clinical_trial_evidence = kwargs.get("ct", 3)
    pair.scores.business_ip = kwargs.get("ip", 5)
    pair.scores.business_regulatory = kwargs.get("reg", 4)
    pair.scores.business_market = kwargs.get("mkt", 5)
    pair.scores.business_manufacturing = kwargs.get("mfg", 4)
    pair.scores.business_clinical_adoption = kwargs.get("clin", 3)
    pair.scores.business_speed_to_revenue = kwargs.get("speed", 4)
    return pair


# ── CandidatePair tests ─────────────────────────────────────────────────────────

class TestCandidatePair:
    def test_business_total_sums_subscores(self):
        pair = make_pair()
        pair.scores.business_ip = 5
        pair.scores.business_regulatory = 4
        pair.scores.business_market = 5
        pair.scores.business_manufacturing = 4
        pair.scores.business_clinical_adoption = 3
        pair.scores.business_speed_to_revenue = 4
        assert pair.scores.business_total == 25

    def test_business_total_none_when_any_subscore_missing(self):
        pair = make_pair()
        pair.scores.business_ip = 5
        # Other scores not set
        assert pair.scores.business_total is None

    def test_disqualified_when_patent_flag_set(self):
        pair = make_pair()
        pair.flags.existing_patent_on_indication = True
        assert pair.flags.is_disqualified is True
        assert "patent" in pair.flags.disqualify_reason.lower()

    def test_disqualified_when_herg_flag_set(self):
        pair = make_pair()
        pair.flags.herg_risk_high = True
        assert pair.flags.is_disqualified is True

    def test_not_disqualified_by_warnings_only(self):
        pair = make_pair()
        pair.flags.pgx_poor_metabolizer_risk_high = True
        pair.flags.polymorph_risk = True
        pair.flags.ddi_risk_narrow_index = True
        assert pair.flags.is_disqualified is False

    def test_to_dict_contains_required_keys(self):
        pair = make_pair()
        d = pair.to_dict()
        assert "drug_id" in d
        assert "disease_id" in d
        assert "composite_score" in d
        assert "is_disqualified" in d
        assert "scores" in d

    def test_repr_includes_drug_and_disease(self):
        pair = make_pair()
        pair.composite_score = 0.75
        r = repr(pair)
        assert "Imatinib" in r
        assert "CML" in r
        assert "0.750" in r


# ── Jaccard similarity tests ─────────────────────────────────────────────────────

class TestJaccard:
    def test_identical_sets_score_one(self):
        s = {"P12345", "P67890"}
        assert jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets_score_zero(self):
        assert jaccard_similarity({"P11111"}, {"P99999"}) == 0.0

    def test_partial_overlap(self):
        a = {"P1", "P2", "P3"}
        b = {"P2", "P3", "P4"}
        # intersection = {P2, P3}, union = {P1, P2, P3, P4}
        assert jaccard_similarity(a, b) == pytest.approx(2 / 4)

    def test_empty_sets_return_zero(self):
        assert jaccard_similarity(set(), set()) == 0.0

    def test_one_empty_set_returns_zero(self):
        assert jaccard_similarity({"P1"}, set()) == 0.0


# ── Normalization tests ──────────────────────────────────────────────────────────

class TestNormalization:
    def test_proximity_zero_hops_maps_to_near_one(self):
        score = _normalize_proximity(0.0)
        assert score > 0.85

    def test_proximity_two_hops_maps_to_half(self):
        # 2 hops is the inflection point
        score = _normalize_proximity(2.0)
        assert 0.45 < score < 0.55

    def test_proximity_five_hops_maps_to_low(self):
        score = _normalize_proximity(5.0)
        assert score < 0.15

    def test_proximity_none_returns_none(self):
        assert _normalize_proximity(None) is None

    def test_ks_negative_one_maps_to_one(self):
        # Perfect reversal signal
        assert _normalize_ks(-1.0) == pytest.approx(1.0)

    def test_ks_zero_maps_to_half(self):
        # No signal
        assert _normalize_ks(0.0) == pytest.approx(0.5)

    def test_ks_positive_one_maps_to_zero(self):
        # Same-direction expression — bad
        assert _normalize_ks(1.0) == pytest.approx(0.0)

    def test_business_thirty_maps_to_one(self):
        assert _normalize_business(30) == pytest.approx(1.0)

    def test_business_none_returns_none(self):
        assert _normalize_business(None) is None

    def test_clinical_five_maps_to_one(self):
        assert _normalize_clinical_trial(5) == pytest.approx(1.0)

    def test_clinical_zero_maps_to_zero(self):
        assert _normalize_clinical_trial(0) == pytest.approx(0.0)


# ── WeightedCompositeScorer tests ────────────────────────────────────────────────

class TestWeightedCompositeScorer:
    def test_disqualified_pair_scores_zero(self):
        scorer = WeightedCompositeScorer()
        pair = make_scored_pair()
        pair.flags.existing_patent_on_indication = True
        result = scorer.compute(pair)
        assert result == 0.0
        assert pair.composite_score == 0.0

    def test_fully_scored_pair_produces_value_in_range(self):
        scorer = WeightedCompositeScorer()
        pair = make_scored_pair()
        result = scorer.compute(pair)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_high_scores_produce_higher_composite(self):
        scorer = WeightedCompositeScorer()
        pair_high = make_scored_pair(
            jaccard=0.9, proximity=0.5, ks=-0.9, cosine=0.95,
            admet=1.0, cooc=100.0, ct=5, ip=5, reg=5, mkt=5, mfg=5, clin=5, speed=5
        )
        pair_low = make_scored_pair(
            jaccard=0.05, proximity=4.0, ks=0.5, cosine=0.1,
            admet=0.3, cooc=1.0, ct=1, ip=2, reg=2, mkt=2, mfg=2, clin=1, speed=1
        )
        scorer.compute(pair_high)
        scorer.compute(pair_low)
        assert pair_high.composite_score > pair_low.composite_score

    def test_missing_scores_handled_gracefully(self):
        scorer = WeightedCompositeScorer()
        pair = make_pair()
        # Only set business scores
        pair.scores.business_ip = 5
        pair.scores.business_regulatory = 4
        pair.scores.business_market = 5
        pair.scores.business_manufacturing = 4
        pair.scores.business_clinical_adoption = 4
        pair.scores.business_speed_to_revenue = 5
        # All biological scores are None
        result = scorer.compute(pair)
        assert result is not None
        assert result > 0

    def test_pgx_penalty_applied_for_high_risk(self):
        scorer = WeightedCompositeScorer()
        pair_no_pgx = make_scored_pair()
        pair_high_pgx = make_scored_pair()
        pair_high_pgx.scores.pgx_metabolizer_risk_score = 0.5
        pair_high_pgx.flags.pgx_poor_metabolizer_risk_high = True

        scorer.compute(pair_no_pgx)
        scorer.compute(pair_high_pgx)
        assert pair_no_pgx.composite_score > pair_high_pgx.composite_score

    def test_score_explanation_includes_all_components(self):
        scorer = WeightedCompositeScorer()
        pair = make_scored_pair()
        scorer.compute(pair)
        explanation = scorer.score_explanation(pair)
        assert "components" in explanation
        assert "business_total" in explanation["components"]
        assert "composite_score" in explanation


# ── CandidateRanker tests ────────────────────────────────────────────────────────

class TestCandidateRanker:
    def test_ranked_by_score_descending(self):
        ranker = CandidateRanker()
        p1 = make_pair(drug_id="D1", drug_name="Drug1",
                       disease_id="DIS1", disease_name="Dis1")
        p2 = make_pair(drug_id="D2", drug_name="Drug2",
                       disease_id="DIS2", disease_name="Dis2")
        p1.composite_score = 0.8
        p2.composite_score = 0.5
        ranked = ranker.rank([p1, p2])
        assert ranked[0].drug_id == "D1"
        assert ranked[1].drug_id == "D2"

    def test_disqualified_sorted_to_end(self):
        ranker = CandidateRanker()
        p1 = make_pair(drug_id="D1", drug_name="Drug1",
                       disease_id="DIS1", disease_name="Dis1")
        p2 = make_pair(drug_id="D2", drug_name="Drug2",
                       disease_id="DIS2", disease_name="Dis2")
        p1.composite_score = 0.9
        p1.flags.existing_patent_on_indication = True  # disqualified despite high score
        p2.composite_score = 0.3
        ranked = ranker.rank([p1, p2])
        assert ranked[0].drug_id == "D2"   # lower score but not disqualified → first
        assert ranked[1].drug_id == "D1"   # disqualified → last

    def test_top_candidates_filters_below_business_threshold(self):
        ranker = CandidateRanker()
        p_good = make_pair(drug_id="D_GOOD", drug_name="GoodDrug",
                           disease_id="DIS_GOOD", disease_name="GoodDis")
        p_good.scores.business_ip = 5
        p_good.scores.business_regulatory = 5
        p_good.scores.business_market = 5
        p_good.scores.business_manufacturing = 5
        p_good.scores.business_clinical_adoption = 4
        p_good.scores.business_speed_to_revenue = 5
        p_good.composite_score = 0.8

        p_low_biz = make_pair(drug_id="D_LOW", drug_name="LowBizDrug",
                              disease_id="DIS_LOW", disease_name="LowBizDis")
        p_low_biz.scores.business_ip = 1
        p_low_biz.scores.business_regulatory = 2
        p_low_biz.scores.business_market = 2
        p_low_biz.scores.business_manufacturing = 2
        p_low_biz.scores.business_clinical_adoption = 1
        p_low_biz.scores.business_speed_to_revenue = 1
        p_low_biz.composite_score = 0.9  # high biological score but low business (total=9)
        top = ranker.top_candidates([p_good, p_low_biz], min_business_score=24)
        ids = [p.drug_id for p in top]
        assert p_good.drug_id in ids
        assert p_low_biz.drug_id not in ids

    def test_rank_assigned_to_all_pairs(self):
        ranker = CandidateRanker()
        pairs = [make_pair(drug_id=f"D{i}", drug_name=f"Drug{i}",
                           disease_id=f"DIS{i}", disease_name=f"Dis{i}") for i in range(5)]
        for p in pairs:
            p.composite_score = 0.5
        ranked = ranker.rank(pairs)
        for i, p in enumerate(ranked):
            assert p.rank == i + 1
