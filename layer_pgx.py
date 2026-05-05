"""
src/layers/layer_pgx.py

South Asian Pharmacogenomics Layer (Tier 1 Critical).

Problem: Every ADMET score in your engine is implicitly calibrated to Western
trial populations. CYP enzyme allele frequencies differ meaningfully between
South Asian and European populations — this makes Western ADMET scores wrong
for a non-trivial fraction of your Indian trial population.

Key enzymes:
  CYP2C19: Poor metabolizer in 13–23% of South Asians vs 2–5% of Europeans
            Loss-of-function alleles: *2 and *3
            → Drug accumulates 3–5× in poor metabolizers → toxicity risk
  CYP2D6:  Ultra-rapid metabolizer frequency higher in some Indian subpopulations
            → Treatment failure (drug metabolized too quickly → insufficient exposure)
  CYP3A5:  *1 expressed allele more common in South Asians
            → Rapid metabolism → may need higher doses for efficacy

Output: Population Metabolizer Risk Score (PMRS)
  Formula: PMRS = Σ (PM_frequency × severity_weight × NTI_weight)
  Range:   0 (no risk) to 1 (high risk in SA population)

Data sources:
  - PharmGKB: https://www.pharmgkb.org/downloads — drug-gene pairs + dosing guidelines
  - gnomAD South Asian cohort (SAS): https://gnomad.broadinstitute.org — allele frequencies
  - IndiGen: https://clingen.igib.res.in — India-specific variant frequencies
  - CPIC guidelines: https://cpicpgx.org — clinical dosing adjustment recommendations

References:
  - Whirl-Carrillo et al. 2021, Clinical Pharmacology & Therapeutics (PharmGKB)
  - Karczewski et al. 2020, Nature (gnomAD v3)
  - Sivasubbu & Scaria 2020, npj Genomics Medicine (IndiGen)
  - Bhatt et al. 2024, Nature Genetics (GenomeIndia)
"""

from __future__ import annotations
import logging
from typing import Optional

import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

PHARMGKB_BASE = "https://api.pharmgkb.org/v1"


# ── South Asian allele frequency reference data ────────────────────────────────
# Source: gnomAD v3 South Asian (SAS) cohort + IndiGen project
# These are population-level poor metabolizer (PM) frequencies

SA_POOR_METABOLIZER_FREQ = {
    "CYP2C19": 0.18,    # 18% SA average (range: 13–23%)
    "CYP2D6_PM": 0.02,  # ~2% SA poor metabolizer
    "CYP2D6_UM": 0.07,  # ~7% SA ultra-rapid metabolizer (treatment failure risk)
    "CYP3A5": 0.12,     # 12% lack CYP3A5 expression (non-expresser *3/*3)
    "TPMT_PM": 0.003,   # 0.3% SA (thiopurine drugs)
    "UGT1A1_PM": 0.05,  # 5% SA (glucuronidation)
    "SLCO1B1": 0.08,    # 8% SA (hepatic uptake, relevant for statins + some antivirals)
}

# Severity weight: how much does PM status affect drug concentration?
# Based on whether drug is primarily metabolized by this enzyme
ENZYME_SEVERITY_WEIGHT = {
    "CYP2C19": 0.9,    # Primary substrate: drug exposure increases 3–5×
    "CYP2D6_PM": 0.7,  # Primary substrate: 2–4× accumulation
    "CYP2D6_UM": 0.6,  # Ultra-rapid: treatment failure
    "CYP3A5": 0.5,     # Partial contributor in most cases
    "TPMT_PM": 0.95,   # Narrow therapeutic index (thiopurines)
    "UGT1A1_PM": 0.6,
    "SLCO1B1": 0.5,
}

# Narrow therapeutic index drugs are weighted higher — small PK changes = serious events
NTI_KEYWORDS = {
    "anticonvulsant", "antiepileptic", "immunosuppressant",
    "anticoagulant", "antiarrhythmic", "cardiac", "digoxin",
    "tacrolimus", "cyclosporine", "warfarin", "phenytoin",
    "carbamazepine", "valproate",
}


class PharmGKBClient:
    """Queries PharmGKB for drug-gene interaction data."""

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_drug_gene_pairs(self, drug_name: str) -> list[dict]:
        """
        Fetch drug-gene interaction pairs (CYP substrate/inhibitor/inducer status).

        Returns list of dicts: {gene_symbol, annotation_types, level_of_evidence}
        """
        url = f"{PHARMGKB_BASE}/data/drugLabel/search"
        params = {"drug": drug_name, "view": "max"}
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                pairs = []
                for item in data.get("data", []):
                    for gene in item.get("relatedGenes", []):
                        pairs.append({
                            "gene_symbol": gene.get("symbol"),
                            "annotation_types": item.get("annotationTypes", []),
                            "level": item.get("level"),
                        })
                return pairs
        except Exception as e:
            logger.debug(f"PharmGKB query failed for {drug_name}: {e}")
        return []

    @cached_api_call(ttl_seconds=86400 * 90)
    def get_cyp_substrates(self, drug_name: str) -> list[str]:
        """
        Return list of CYP enzymes for which this drug is a substrate.
        e.g., ['CYP2C19', 'CYP3A4']
        """
        # In production: parse PharmGKB drug labels + DrugBank CYP profiles
        # Simplified: query PharmGKB gene-drug pairs filtered to CYP genes
        pairs = self.get_drug_gene_pairs(drug_name)
        cyp_genes = []
        for pair in pairs:
            gene = pair.get("gene_symbol", "")
            if gene and gene.startswith("CYP"):
                types = pair.get("annotation_types", [])
                if "Metabolism" in types or "Pharmacokinetics" in types:
                    cyp_genes.append(gene)
        return list(set(cyp_genes))


class SouthAsianPGxLayer(BaseLayer):
    """
    South Asian Pharmacogenomics Layer.

    Scores:
        pair.scores.pgx_metabolizer_risk_score   (0–1; higher = more risk in SA population)
        pair.scores.cyp_substrate_enzymes         (list of relevant CYP enzymes)

    Flags:
        pair.flags.pgx_poor_metabolizer_risk_high   (risk score > 0.15)

    This score should be used to:
        1. Penalize candidates where a large fraction of SA patients will
           accumulate dangerous drug levels
        2. Flag candidates requiring mandatory PGx genotyping in trial protocol
        3. Alert bio team to consider dose adjustment for SA population
    """

    layer_name = "layer_pgx_south_asian"
    version = "1.0"

    PGX_HIGH_RISK_THRESHOLD = 0.15    # PMRS above this → flag for bio review

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.pharmgkb = PharmGKBClient()

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Get CYP substrate enzymes ──────────────────────────────────
        cyp_substrates = self.pharmgkb.get_cyp_substrates(pair.drug_name)

        if not cyp_substrates:
            # No CYP data found — flag as unknown, not zero risk
            logger.warning(
                f"[{self.layer_name}] No CYP substrate data found for {pair.drug_name}. "
                f"Manual PharmGKB lookup recommended."
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
            # Map enzyme to our frequency table
            freq_key = self._enzyme_to_freq_key(enzyme)
            if freq_key not in SA_POOR_METABOLIZER_FREQ:
                continue

            pm_freq = SA_POOR_METABOLIZER_FREQ[freq_key]
            severity = ENZYME_SEVERITY_WEIGHT.get(freq_key, 0.5)
            pmrs += pm_freq * severity * nti_weight

        # Normalize to 0–1 range (cap at 1)
        pmrs = min(1.0, pmrs)
        pair.scores.pgx_metabolizer_risk_score = pmrs

        # ── 3. Flag high-risk candidates ──────────────────────────────────
        if pmrs > self.PGX_HIGH_RISK_THRESHOLD:
            pair.flags.pgx_poor_metabolizer_risk_high = True

        logger.info(
            f"[{self.layer_name}] {pair.drug_name}: "
            f"CYP substrates={cyp_substrates}, PMRS={pmrs:.3f}"
            + (" [HIGH RISK — genotyping required]" if pair.flags.pgx_poor_metabolizer_risk_high else "")
            + (" [NTI drug — elevated weight applied]" if is_nti else "")
        )

        return pair

    def _enzyme_to_freq_key(self, enzyme: str) -> str:
        """Map ChEMBL/PharmGKB enzyme name to our frequency table key."""
        enzyme = enzyme.upper().strip()
        if enzyme in ("CYP2C19",):
            return "CYP2C19"
        if enzyme in ("CYP2D6",):
            return "CYP2D6_PM"
        if enzyme in ("CYP3A5", "CYP3A4"):
            return "CYP3A5"
        if enzyme in ("TPMT",):
            return "TPMT_PM"
        if enzyme in ("UGT1A1",):
            return "UGT1A1_PM"
        if enzyme in ("SLCO1B1",):
            return "SLCO1B1"
        return enzyme

    def _is_narrow_therapeutic_index(self, drug_name: str, disease_name: str) -> bool:
        """
        Heuristic: check if drug or disease name contains NTI keywords.
        Bio team should validate for every candidate.
        """
        combined = (drug_name + " " + disease_name).lower()
        return any(kw in combined for kw in NTI_KEYWORDS)
