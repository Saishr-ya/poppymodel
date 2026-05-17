"""
src/scoring/candidate.py

Core data model for the drug repurposing engine.
Every scoring layer reads and writes to CandidatePair.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import datetime


@dataclass
class LayerScores:
    """
    Scores produced by each biological and business layer.
    None = not yet computed.
    """

    # Layer 1 — Target & Gene Overlap
    target_overlap_jaccard: Optional[float] = None          # 0–1, higher = better
    network_proximity: Optional[float] = None               # avg shortest-path hops; lower = better
    pathway_enrichment_pvalue: Optional[float] = None       # hypergeometric p-value; lower = better

    # Layer 2 — Transcriptomic Signature Reversal
    transcriptomic_reversal_ks: Optional[float] = None      # KS statistic; strongly negative = good

    # Layer 3 — Knowledge Graph Embedding
    kg_embedding_cosine: Optional[float] = None             # cosine similarity; higher = better

    # Layer 4 — ADMET
    admet_composite: Optional[float] = None                 # 0–1 normalized; higher = safer
    oral_bioavailability_pct: Optional[float] = None
    bcs_class: Optional[str] = None                        # I, II, III, IV
    herg_ic50_um: Optional[float] = None

    # Layer 5 — Literature & Clinical Evidence
    pubmed_cooccurrence_score: Optional[float] = None       # weighted paper count
    clinical_trial_evidence: Optional[int] = None          # 0–5 scale
    case_report_count: Optional[int] = None

    # Layer 6 — Business Scoring (each 1–5, max 30 total)
    business_ip: Optional[int] = None
    business_regulatory: Optional[int] = None
    business_market: Optional[int] = None
    business_manufacturing: Optional[int] = None
    business_clinical_adoption: Optional[int] = None
    business_speed_to_revenue: Optional[int] = None

    # Chiral Switch Module
    chirality_divergence_score: Optional[float] = None      # receptor divergence between enantiomers
    chiral_switch_candidate: Optional[bool] = None

    # South Asian PGx Layer
    pgx_metabolizer_risk_score: Optional[float] = None      # 0–1; higher = more risk in SA population
    cyp_substrate_enzymes: Optional[list] = None

    # Additional Tier-1 Layers
    ddi_risk_score: Optional[float] = None                  # 0–1; higher = more DDI risk
    active_metabolite_flag: Optional[bool] = None
    pk_window_fit: Optional[bool] = None                    # True if required dose is within therapeutic window

    @property
    def business_total(self) -> Optional[int]:
        scores = [
            self.business_ip,
            self.business_regulatory,
            self.business_market,
            self.business_manufacturing,
            self.business_clinical_adoption,
            self.business_speed_to_revenue,
        ]
        if any(s is None for s in scores):
            return None
        return sum(scores)


@dataclass
class Flags:
    """
    Hard disqualifiers (any True = candidate dropped from ranking).
    Warnings (flagged but not disqualifying).
    """

    # ── Hard disqualifiers ────────────────────────────────────────────
    existing_patent_on_indication: bool = False             # IP score = 1
    herg_risk_high: bool = False                            # hERG IC50 < 1 µM
    faers_ror_critical: bool = False                        # ROR > 3 for serious adverse events
    bioavailability_insufficient: bool = False              # oral BA < 20%
    lipinski_violations: int = 0                            # >1 violation = penalized

    # ── Warnings ──────────────────────────────────────────────────────
    ddi_risk_narrow_index: bool = False                     # interacts with narrow-TI co-meds
    polymorph_risk: bool = False                            # multiple crystal forms documented
    pediatric_formulation_needed: bool = False              # onset <12 yrs, no peds formulation
    pgx_poor_metabolizer_risk_high: bool = False            # >10% SA population affected
    south_asian_founder_variant_specific: bool = False      # founder variant limits generalizability
    variant_specific_mechanism: bool = False                # drug mechanism is variant-specific

    @property
    def is_disqualified(self) -> bool:
        # Only two hard disqualifiers — both are commercially fatal:
        #   Patent conflict: a competitor owns the exact indication we'd claim.
        #   hERG risk: cardiac arrhythmia at therapeutic doses kills the IND.
        #
        # FAERS, Lipinski, and bioavailability are NOT hard disqualifiers for
        # repurposed approved drugs. These drugs are already approved — regulators
        # have already evaluated safety. FAERS signals are in the context of the
        # existing indication, not the new one. Lipinski was designed for novel
        # compound screening, not approved-drug evaluation. These flags penalize
        # the composite score via admet_composite but do not eliminate candidates.
        return (
            self.existing_patent_on_indication
            or self.herg_risk_high
        )

    @property
    def disqualify_reason(self) -> Optional[str]:
        if self.existing_patent_on_indication:
            return "Existing patent covers this indication"
        if self.herg_risk_high:
            return "High hERG cardiotoxicity risk (IC50 < 1 µM)"
        return None


@dataclass
class CandidatePair:
    """
    The central object passed through every scoring layer.
    Engineers: mutate scores and flags only via the layer's score() method.
    Bio team: review composite_score and flags for every top-ranked pair.
    """

    drug_id: str                                            # ChEMBL ID e.g. "CHEMBL192"
    drug_name: str
    disease_id: str                                         # "OMIM:123456" or "ORPHA:12345"
    disease_name: str

    scores: LayerScores = field(default_factory=LayerScores)
    flags: Flags = field(default_factory=Flags)

    composite_score: Optional[float] = None                 # Final 0–1 probability
    rank: Optional[int] = None

    # Traceability — every layer logs its data source and version here
    data_sources: dict = field(default_factory=dict)

    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    notes: str = ""

    def touch(self):
        self.updated_at = datetime.datetime.now(datetime.timezone.utc)

    def to_dict(self) -> dict:
        return {
            "drug_id": self.drug_id,
            "drug_name": self.drug_name,
            "disease_id": self.disease_id,
            "disease_name": self.disease_name,
            "composite_score": self.composite_score,
            "business_total": self.scores.business_total,
            "is_disqualified": self.flags.is_disqualified,
            "disqualify_reason": self.flags.disqualify_reason,
            "scores": {
                k: v for k, v in self.scores.__dict__.items() if v is not None
            },
            "flags": {k: v for k, v in self.flags.__dict__.items()},
            "data_sources": self.data_sources,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self):
        score_str = f"{self.composite_score:.3f}" if self.composite_score else "unscored"
        dq = " [DISQUALIFIED]" if self.flags.is_disqualified else ""
        return f"CandidatePair({self.drug_name} × {self.disease_name} | {score_str}{dq})"
