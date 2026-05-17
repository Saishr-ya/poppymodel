"""
src/layers/layer_pgx.py

South Asian Pharmacogenomics Layer (Tier 1 Critical).
Updated to use the fixed PharmGKBClient that works without the broken REST API.

No logic changes — only the import and data source are updated.
All scoring formulas, thresholds, and PMRS calculation are unchanged.
"""

from __future__ import annotations
import logging
from typing import Optional

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.pharmgkb_client import PharmGKBClient   # updated client

logger = logging.getLogger(__name__)


# ── South Asian allele frequency reference data ────────────────────────────────
# Source: gnomAD v3 South Asian (SAS) cohort + IndiGen project
SA_POOR_METABOLIZER_FREQ = {
    "CYP2C19":   0.18,    # 18% SA average (range: 13–23%)
    "CYP2D6_PM": 0.02,    # ~2% SA poor metabolizer
    "CYP2D6_UM": 0.07,    # ~7% SA ultra-rapid metabolizer (treatment failure risk)
    "CYP3A5":    0.12,    # 12% lack CYP3A5 expression (non-expresser *3/*3)
    "CYP3A4":    0.12,    # treated same as CYP3A5 for simplicity
    "TPMT_PM":   0.003,
    "UGT1A1_PM": 0.05,
    "SLCO1B1":   0.08,
}

ENZYME_SEVERITY_WEIGHT = {
    "CYP2C19":   0.9,
    "CYP2D6_PM": 0.7,
    "CYP2D6_UM": 0.6,
    "CYP3A5":    0.5,
    "CYP3A4":    0.5,
    "TPMT_PM":   0.95,
    "UGT1A1_PM": 0.6,
    "SLCO1B1":   0.5,
}

NTI_KEYWORDS = {
    "anticonvulsant", "antiepileptic", "immunosuppressant",
    "anticoagulant", "antiarrhythmic", "cardiac", "digoxin",
    "tacrolimus", "cyclosporine", "warfarin", "phenytoin",
    "carbamazepine", "valproate",
}


class SouthAsianPGxLayer(BaseLayer):
    """
    South Asian Pharmacogenomics Layer.

    Scores:
        pair.scores.pgx_metabolizer_risk_score   (0–1; higher = more risk in SA population)
        pair.scores.cyp_substrate_enzymes         (list of relevant CYP enzymes)

    Flags:
        pair.flags.pgx_poor_metabolizer_risk_high   (risk score > 0.15)
    """

    layer_name = "layer_pgx_south_asian"
    version = "1.1"   # bumped: updated to use fixed PharmGKBClient

    PGX_HIGH_RISK_THRESHOLD = 0.15

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.pharmgkb = PharmGKBClient()   # now uses hardcoded reference + file

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Get CYP substrate enzymes ──────────────────────────────────
        cyp_substrates = self.pharmgkb.get_cyp_substrates(pair.drug_name)

        if not cyp_substrates:
            logger.warning(
                f"[{self.layer_name}] No CYP substrate data for '{pair.drug_name}'. "
                f"Add it to CYP_REFERENCE in pharmgkb_client.py or "
                f"download PharmGKB relationships.tsv."
            )
            pair.scores.cyp_substrate_enzymes = []
            pair.scores.pgx_metabolizer_risk_score = None
            return pair

        pair.scores.cyp_substrate_enzymes = cyp_substrates

        # ── 2. Compute Population Metabolizer Risk Score ───────────────────
        is_nti = self._is_narrow_therapeutic_index(pair.drug_name, pair.disease_name)
        nti_weight = 1.5 if is_nti else 1.0

        pmrs = 0.0
        for enzyme in cyp_substrates:
            freq_key = self._enzyme_to_freq_key(enzyme)
            if freq_key not in SA_POOR_METABOLIZER_FREQ:
                continue
            pm_freq = SA_POOR_METABOLIZER_FREQ[freq_key]
            severity = ENZYME_SEVERITY_WEIGHT.get(freq_key, 0.5)
            pmrs += pm_freq * severity * nti_weight

        pmrs = min(1.0, pmrs)
        pair.scores.pgx_metabolizer_risk_score = pmrs

        # ── 3. Flag high-risk candidates ──────────────────────────────────
        if pmrs > self.PGX_HIGH_RISK_THRESHOLD:
            pair.flags.pgx_poor_metabolizer_risk_high = True

        logger.info(
            f"[{self.layer_name}] {pair.drug_name}: "
            f"CYP={cyp_substrates}, PMRS={pmrs:.3f}"
            + (" [HIGH RISK]" if pair.flags.pgx_poor_metabolizer_risk_high else "")
            + (" [NTI]" if is_nti else "")
        )
        return pair

    def _enzyme_to_freq_key(self, enzyme: str) -> str:
        e = enzyme.upper().strip()
        if e == "CYP2C19":    return "CYP2C19"
        if e == "CYP2D6":     return "CYP2D6_PM"
        if e in ("CYP3A5", "CYP3A4"): return e
        if e == "TPMT":       return "TPMT_PM"
        if e == "UGT1A1":     return "UGT1A1_PM"
        if e == "SLCO1B1":    return "SLCO1B1"
        return e

    def _is_narrow_therapeutic_index(self, drug_name: str, disease_name: str) -> bool:
        combined = (drug_name + " " + disease_name).lower()
        return any(kw in combined for kw in NTI_KEYWORDS)