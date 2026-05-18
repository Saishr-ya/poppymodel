"""
src/validation/ground_truth.py

Ground truth dataset for engine validation.

Known positive pairs: drugs with established efficacy in a disease
Known negative pairs: drugs that failed Phase III for a disease

Load these first, score them with the engine, and measure:
  - AUROC (Area Under ROC Curve): target > 0.75
  - Precision@20: what fraction of your top 20 are known positives?
  - False negative rate: what fraction of known positives score in the bottom half?

Source for positives: DrugBank "approved" drug-indication pairs
Source for negatives: ClinicalTrials.gov Phase III failures (status=Terminated + primary_outcome failed)

Bio team: add known pairs from your disease domain knowledge here.
Every pair must have an evidence_source — no undocumented assertions.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

GROUND_TRUTH_PATH = "data/processed/ground_truth.json"


@dataclass
class GroundTruthPair:
    drug_id: str
    drug_name: str
    disease_id: str
    disease_name: str
    label: int              # 1 = known positive, 0 = known negative
    evidence_source: str    # e.g., "DrugBank approved", "Phase III failure NCT01234"
    notes: str = ""


# ── Hardcoded seed pairs (expand via ground_truth.json in production) ──────────

SEED_POSITIVES: list[GroundTruthPair] = [
    GroundTruthPair(
        drug_id="CHEMBL192",
        drug_name="Imatinib",
        disease_id="ORPHA:355",
        disease_name="Chronic myeloid leukemia",
        label=1,
        evidence_source="DrugBank approved — FDA 2001",
        notes="First BCR-ABL inhibitor. Gold standard positive for KG and target overlap layers.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL1520",
        drug_name="Sildenafil",
        disease_id="ORPHA:422",
        disease_name="Pulmonary arterial hypertension",
        label=1,
        evidence_source="DrugBank approved — FDA 2005 (Revatio)",
        notes="Classic repurposing case: ED → PAH. Strong transcriptomic reversal signal expected.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL53463",
        drug_name="Miglustat",
        disease_id="ORPHA:77",
        disease_name="Gaucher disease type 1",
        label=1,
        evidence_source="DrugBank approved — EMA 2002",
        notes="Substrate reduction therapy. Validates metabolic rare disease scoring.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL1201585",
        drug_name="Fenfluramine",
        disease_id="ORPHA:33069",
        disease_name="Dravet syndrome",
        label=1,
        evidence_source="FDA approved 2020 (Fintepla)",
        notes="Chiral switch case study. l-fenfluramine separated from cardiac-toxic d-enantiomer.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL579",
        drug_name="Metformin",
        disease_id="ORPHA:586",
        disease_name="Polycystic ovary syndrome",
        label=1,
        evidence_source="DrugBank approved — off-label use well-documented",
        notes="Tests literature co-occurrence and clinical adoption layers.",
    ),
    GroundTruthPair(
    drug_id="CHEMBL192", drug_name="Imatinib",
    disease_id="ORPHA:566", disease_name="Pompe disease",
    label=1,
    evidence_source="Phase II trial NCT00093015 — imatinib in Pompe disease",
    notes="Targets PDGFR pathway, showed signal in Pompe fibroblasts.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL714", drug_name="Hydroxychloroquine",
        disease_id="ORPHA:77", disease_name="Gaucher disease type 1",
        label=1,
        evidence_source="Case reports + mechanistic studies — lysosomal pathway",
        notes="Lysosomal pH modulation — indirect mechanism, published case series.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL1421", drug_name="Bosentan",
        disease_id="ORPHA:422", disease_name="Pulmonary arterial hypertension",
        label=1,
        evidence_source="FDA approved 2001 (Tracleer) — endothelin receptor antagonist",
        notes="Core PAH therapy. Strong ground truth positive.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL1500", drug_name="Tadalafil",
        disease_id="ORPHA:422", disease_name="Pulmonary arterial hypertension",
        label=1,
        evidence_source="FDA approved 2009 (Adcirca) — PDE5 inhibitor same class as sildenafil",
        notes="Direct comparator to sildenafil. Should score very similarly.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL1201585", drug_name="Fenfluramine",
        disease_id="ORPHA:33069", disease_name="Dravet syndrome",
        label=1,
        evidence_source="FDA approved 2020 (Fintepla) — low-dose for Dravet",
        notes="Chiral switch case study. Should score high on chirality layer.",
    ),
]

SEED_NEGATIVES: list[GroundTruthPair] = [
    GroundTruthPair(
        drug_id="CHEMBL192",
        drug_name="Imatinib",
        disease_id="ORPHA:101435",
        disease_name="Autosomal recessive primary microcephaly",
        label=0,
        evidence_source="No mechanistic relationship — control negative",
        notes="Imatinib targets BCR-ABL; MCPH is a neuronal development disorder. No pathway overlap expected.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL579",
        drug_name="Metformin",
        disease_id="ORPHA:566",
        disease_name="Pompe disease",
        label=0,
        evidence_source="No mechanistic relationship — control negative",
        notes="Pompe is GAA enzyme deficiency; metformin targets AMPK. Negative control.",
    ),
    GroundTruthPair(
    drug_id="CHEMBL192", drug_name="Imatinib",
    disease_id="ORPHA:33069", disease_name="Dravet syndrome",
    label=0,
    evidence_source="No mechanistic relationship — BCR-ABL inhibitor vs SCN1A channelopathy",
    notes="Negative control. Completely different biology.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL579", drug_name="Metformin",
        disease_id="ORPHA:422", disease_name="Pulmonary arterial hypertension",
        label=0,
        evidence_source="No established mechanism for PAH — AMPK activator",
        notes="Negative control for PAH.",
    ),
    GroundTruthPair(
        drug_id="CHEMBL714", drug_name="Hydroxychloroquine",
        disease_id="ORPHA:33069", disease_name="Dravet syndrome",
        label=0,
        evidence_source="No mechanistic relationship — antimalarial vs SCN1A channelopathy",
        notes="Negative control for Dravet.",
    ),
]


def load_ground_truth(path: str = GROUND_TRUTH_PATH) -> list[GroundTruthPair]:
    """
    Load ground truth pairs from JSON file (supplementing seed pairs).
    Returns combined list of seed pairs + file pairs.
    """
    all_pairs = list(SEED_POSITIVES) + list(SEED_NEGATIVES)

    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        for item in data:
            all_pairs.append(GroundTruthPair(**item))

    return all_pairs


def save_ground_truth(pairs: list[GroundTruthPair], path: str = GROUND_TRUTH_PATH):
    """Persist additional ground truth pairs to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [
        {
            "drug_id": p.drug_id,
            "drug_name": p.drug_name,
            "disease_id": p.disease_id,
            "disease_name": p.disease_name,
            "label": p.label,
            "evidence_source": p.evidence_source,
            "notes": p.notes,
        }
        for p in pairs
        if p not in SEED_POSITIVES and p not in SEED_NEGATIVES
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class ValidationMetrics:
    """
    Compute validation metrics for the scoring engine against ground truth.
    """

    @staticmethod
    def auroc(
        scored_pairs,    # list of CandidatePair (with composite_score set)
        ground_truth: list[GroundTruthPair],
    ) -> float:
        """
        Compute AUROC on the intersection of scored pairs and ground truth.
        Target: > 0.75 by Month 4 of the build plan.
        """
        from sklearn.metrics import roc_auc_score

        gt_lookup = {
            f"{p.drug_id}×{p.disease_id}": p.label
            for p in ground_truth
        }

        labels = []
        scores = []
        for pair in scored_pairs:
            key = f"{pair.drug_id}×{pair.disease_id}"
            if key in gt_lookup:
                labels.append(gt_lookup[key])
                scores.append(pair.composite_score or 0.0)

        if len(set(labels)) < 2:
            raise ValueError(
                f"AUROC requires both positive and negative labels. "
                f"Found: {len(labels)} matched pairs with labels: {set(labels)}"
            )

        return roc_auc_score(labels, scores)

    @staticmethod
    def precision_at_k(
        ranked_pairs,    # list of CandidatePair, already sorted by rank
        ground_truth: list[GroundTruthPair],
        k: int = 20,
    ) -> float:
        """
        Precision@K: fraction of top-K candidates that are known positives.
        Target: > 0.40 by Month 5 (assuming rare disease enrichment is hard).
        """
        positive_keys = {
            f"{p.drug_id}×{p.disease_id}"
            for p in ground_truth
            if p.label == 1
        }

        top_k = ranked_pairs[:k]
        hits = sum(
            1 for pair in top_k
            if f"{pair.drug_id}×{pair.disease_id}" in positive_keys
        )
        return hits / k if k > 0 else 0.0

    @staticmethod
    def false_negative_analysis(
        ranked_pairs,
        ground_truth: list[GroundTruthPair],
        bottom_fraction: float = 0.5,
    ) -> list[dict]:
        """
        Find known positives that ranked in the bottom half.
        These are false negatives — the bio team should review why they scored low.
        """
        n = len(ranked_pairs)
        bottom_start = int(n * (1 - bottom_fraction))
        bottom_pairs = ranked_pairs[bottom_start:]

        positive_keys = {
            f"{p.drug_id}×{p.disease_id}": p
            for p in ground_truth
            if p.label == 1
        }

        false_negatives = []
        for pair in bottom_pairs:
            key = f"{pair.drug_id}×{pair.disease_id}"
            if key in positive_keys:
                gt = positive_keys[key]
                false_negatives.append({
                    "drug": pair.drug_name,
                    "disease": pair.disease_name,
                    "rank": pair.rank,
                    "composite_score": pair.composite_score,
                    "disqualified": pair.flags.is_disqualified,
                    "disqualify_reason": pair.flags.disqualify_reason,
                    "evidence_source": gt.evidence_source,
                    "notes": gt.notes,
                    "scores": {
                        k: v for k, v in pair.scores.__dict__.items()
                        if v is not None
                    },
                })

        return false_negatives
