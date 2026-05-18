"""
src/layers/layer_pgx.py

South Asian Pharmacogenomics Layer.

Fix #8: Replaced hardcoded SA_POOR_METABOLIZER_FREQ dict with dynamic gnomAD API
        lookups. The hardcoded values were a reasonable starting point but cannot
        account for variant discovery in newer gnomAD releases. The new approach
        queries gnomAD v3 for the SAS (South Asian) population allele frequency of
        known loss-of-function variants for each enzyme, then caches the result for
        90 days. Falls back to the hardcoded values if gnomAD is unreachable.
"""

from __future__ import annotations
import logging
from typing import Optional

import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.cache import cached_api_call
from src.ingestion.pharmgkb_client import PharmGKBClient

logger = logging.getLogger(__name__)

GNOMAD_GRAPHQL = "https://gnomad.broadinstitute.org/api"

# Known loss-of-function variants per enzyme in gnomAD coordinates.
# Format: variant_id is the gnomAD 'chrom-pos-ref-alt' string.
ENZYME_VARIANTS: dict[str, list[dict]] = {
    "CYP2C19": [
        {"variant_id": "10-94781858-G-A",  "allele": "*2", "effect": "poor"},   # CYP2C19*2 (most common PM allele)
        {"variant_id": "10-94761900-C-T",  "allele": "*3", "effect": "poor"},   # CYP2C19*3
    ],
    "CYP2D6": [
        {"variant_id": "22-42524947-G-A",  "allele": "*4", "effect": "poor"},   # CYP2D6*4
    ],
    "CYP3A5": [
        {"variant_id": "7-99672916-T-C",   "allele": "*3", "effect": "non_expresser"},  # CYP3A5*3
    ],
}

# Severity weights — how much a poor-metabolizer event at this enzyme matters
# clinically (used in the PMRS formula).
ENZYME_SEVERITY_WEIGHT: dict[str, float] = {
    "CYP2C19":   0.9,
    "CYP2D6_PM": 0.7,
    "CYP2D6_UM": 0.6,
    "CYP3A5":    0.5,
    "CYP3A4":    0.5,
    "TPMT_PM":   0.95,
    "UGT1A1_PM": 0.6,
    "SLCO1B1":   0.5,
}

# Fallback frequencies (gnomAD SAS cohort estimates from literature)
# Used if the gnomAD API is unreachable.
_FALLBACK_FREQ: dict[str, float] = {
    "CYP2C19":   0.18,
    "CYP2D6_PM": 0.02,
    "CYP2D6_UM": 0.07,
    "CYP3A5":    0.12,
    "CYP3A4":    0.12,
    "TPMT_PM":   0.003,
    "UGT1A1_PM": 0.05,
    "SLCO1B1":   0.08,
}

NTI_KEYWORDS = {
    "anticonvulsant", "antiepileptic", "immunosuppressant",
    "anticoagulant", "antiarrhythmic", "cardiac", "digoxin",
    "tacrolimus", "cyclosporine", "warfarin", "phenytoin",
    "carbamazepine", "valproate",
}

_GNOMAD_VARIANT_QUERY = """
query VariantFreq($variantId: String!, $dataset: DatasetId!) {
  variant(variantId: $variantId, dataset: $dataset) {
    genome {
      populations {
        id
        ac
        an
      }
    }
  }
}
"""


@cached_api_call(ttl_seconds=86400 * 90)
def _get_sa_allele_frequency(enzyme: str) -> float:
    """
    Fix #8: Query gnomAD v3 for South Asian allele frequency of loss-of-function
    variants for the given CYP enzyme.

    Returns combined allele frequency across all known LoF alleles.
    Falls back to hardcoded estimate if gnomAD is unreachable.
    """
    variants = ENZYME_VARIANTS.get(enzyme, [])
    if not variants:
        return _FALLBACK_FREQ.get(enzyme, 0.0)

    total_freq = 0.0
    for variant_info in variants:
        query_vars = {
            "variantId": variant_info["variant_id"],
            "dataset": "gnomad_r3",
        }
        try:
            r = requests.post(
                GNOMAD_GRAPHQL,
                json={"query": _GNOMAD_VARIANT_QUERY, "variables": query_vars},
                timeout=20,
            )
            r.raise_for_status()
            populations = (
                r.json()
                .get("data", {})
                .get("variant", {})
                .get("genome", {})
                .get("populations", [])
            )
            for pop in populations:
                if pop.get("id") == "sas":   # South Asian cohort
                    an = pop.get("an", 0)
                    ac = pop.get("ac", 0)
                    if an > 0:
                        total_freq += ac / an
        except Exception as e:
            logger.debug(
                f"gnomAD query failed for {enzyme} variant "
                f"{variant_info['variant_id']}: {e}"
            )

    if total_freq == 0.0:
        # gnomAD unreachable or variant not found — use fallback
        fallback = _FALLBACK_FREQ.get(enzyme, 0.0)
        logger.debug(
            f"gnomAD returned 0 for {enzyme}; using fallback frequency {fallback}"
        )
        return fallback

    return min(1.0, total_freq)


class SouthAsianPGxLayer(BaseLayer):
    """
    South Asian Pharmacogenomics Layer.

    Fix #8: SA allele frequencies are now fetched from gnomAD v3 rather than
    hardcoded. This makes the PMRS accurately track published allele frequencies
    as the gnomAD database grows.

    Scores:
        pair.scores.pgx_metabolizer_risk_score   (0–1; higher = more risk in SA population)
        pair.scores.cyp_substrate_enzymes         (list of relevant CYP enzymes)

    Flags:
        pair.flags.pgx_poor_metabolizer_risk_high   (risk score > 0.15)
    """

    layer_name = "layer_pgx_south_asian"
    version = "1.2"

    PGX_HIGH_RISK_THRESHOLD = 0.15

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.pharmgkb = PharmGKBClient()

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Get CYP substrate enzymes ──────────────────────────────────
        cyp_substrates = self.pharmgkb.get_cyp_substrates(pair.drug_name)

        if not cyp_substrates:
            logger.warning(
                f"[{self.layer_name}] No CYP substrate data for '{pair.drug_name}'. "
                f"Add it to CYP_REFERENCE in pharmgkb_client.py or download "
                f"PharmGKB relationships.tsv."
            )
            pair.scores.cyp_substrate_enzymes = []
            pair.scores.pgx_metabolizer_risk_score = None
            return pair

        pair.scores.cyp_substrate_enzymes = cyp_substrates

        # ── 2. Compute Population Metabolizer Risk Score ───────────────────
        is_nti      = self._is_narrow_therapeutic_index(pair.drug_name, pair.disease_name)
        nti_weight  = 1.5 if is_nti else 1.0

        pmrs = 0.0
        for enzyme in cyp_substrates:
            freq_key = self._enzyme_to_freq_key(enzyme)
            # Fix #8: fetch from gnomAD (cached 90 days) instead of hardcoded dict
            pm_freq  = _get_sa_allele_frequency(freq_key)
            severity = ENZYME_SEVERITY_WEIGHT.get(freq_key, 0.5)
            pmrs    += pm_freq * severity * nti_weight

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
        if e == "CYP2C19":              return "CYP2C19"
        if e == "CYP2D6":               return "CYP2D6_PM"
        if e in ("CYP3A5", "CYP3A4"):   return e
        if e == "TPMT":                  return "TPMT_PM"
        if e == "UGT1A1":               return "UGT1A1_PM"
        if e == "SLCO1B1":              return "SLCO1B1"
        return e

    def _is_narrow_therapeutic_index(self, drug_name: str, disease_name: str) -> bool:
        combined = (drug_name + " " + disease_name).lower()
        return any(kw in combined for kw in NTI_KEYWORDS)