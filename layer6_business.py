"""
src/layers/layer6_business.py

Layer 6 — Business and Regulatory Scoring.

This is what separates your engine from academic repurposing pipelines.
A candidate with perfect biology but an existing patent is worthless.
A candidate with moderate biology, no patent, strong Indian adoption, and
a clean 25-year safety record is gold.

Scoring dimensions (each 1–5, max 30):
  1. IP score          — is this indication patentable?
  2. Regulatory score  — approved where? Clean safety record?
  3. Market score      — patient population size in India (Goldilocks window)
  4. Manufacturing     — oral tablet, Indian CMO available?
  5. Clinical adoption — Indian physicians using off-label?
  6. Speed to revenue  — how quickly can you get to trial/approval?

Threshold: only pursue candidates scoring ≥ 24/30.

Data sources:
  - Patent search: Google Patents API / USPTO API / Espacenet
  - Drug approval: FDA Orange Book, EMA EPAR, CDSCO database
  - Patient population: Orphanet prevalence × India population adjustment
  - Manufacturing: DrugBank route + Indian CMO database (manual lookup)
  - Clinical adoption: PubMed Indian author case reports (Layer 5)

Bio team: This layer requires manual input for several dimensions.
The _manual_overrides config dict allows the bio team to set known values
directly when they have better data than the automated sources.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

INDIA_POPULATION = 1_400_000_000
ORPHANET_API = "https://api.orphacode.org/EN/ClinicalEntity"

# Goldilocks patient window for India (per the spec)
PATIENT_MIN = 1_000
PATIENT_MAX = 100_000
PATIENT_SWEET_SPOT_MIN = 10_000
PATIENT_SWEET_SPOT_MAX = 50_000


@dataclass
class BusinessScoreConfig:
    """
    Manual overrides for business scoring dimensions.
    Bio team sets these when they have definitive data (e.g., patent search result).

    Values of None = use automated scoring.
    Values of 1–5  = override automated scoring.
    """
    ip_score: Optional[int] = None
    regulatory_score: Optional[int] = None
    market_score: Optional[int] = None
    manufacturing_score: Optional[int] = None
    clinical_adoption_score: Optional[int] = None
    speed_score: Optional[int] = None
    notes: str = ""


class BusinessLayer(BaseLayer):
    """
    Layer 6 — Commercial viability scoring.

    Scores (1–5 each):
        pair.scores.business_ip
        pair.scores.business_regulatory
        pair.scores.business_market
        pair.scores.business_manufacturing
        pair.scores.business_clinical_adoption
        pair.scores.business_speed_to_revenue

    Composite:
        pair.scores.business_total (sum, /30)

    Flags:
        pair.flags.existing_patent_on_indication   (IP score = 1)
    """

    layer_name = "layer6_business"
    version = "1.0"

    # Threshold from the spec: only pursue ≥ 24/30
    MINIMUM_BUSINESS_SCORE = 24

    def __init__(
        self,
        config: Optional[dict] = None,
        overrides: Optional[dict[str, BusinessScoreConfig]] = None,
    ):
        """
        Args:
            overrides: Dict mapping "drug_id×disease_id" to BusinessScoreConfig.
                       Allows the bio team to manually set scores for reviewed pairs.
                       e.g., {"CHEMBL192×ORPHA:355": BusinessScoreConfig(ip_score=5)}
        """
        super().__init__(config)
        self.overrides: dict[str, BusinessScoreConfig] = overrides or {}

    def score(self, pair: CandidatePair) -> CandidatePair:
        override_key = f"{pair.drug_id}×{pair.disease_id}"
        override = self.overrides.get(override_key, BusinessScoreConfig())

        # ── 1. IP Score ───────────────────────────────────────────────────
        ip = override.ip_score if override.ip_score is not None \
            else self._score_ip(pair)
        pair.scores.business_ip = ip
        if ip == 1:
            pair.flags.existing_patent_on_indication = True
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"PATENT CONFLICT — disqualifying"
            )

        # ── 2. Regulatory Score ───────────────────────────────────────────
        reg = override.regulatory_score if override.regulatory_score is not None \
            else self._score_regulatory(pair)
        pair.scores.business_regulatory = reg

        # ── 3. Market Score (India patient count) ─────────────────────────
        mkt = override.market_score if override.market_score is not None \
            else self._score_market(pair)
        pair.scores.business_market = mkt

        # ── 4. Manufacturing Score ────────────────────────────────────────
        mfg = override.manufacturing_score if override.manufacturing_score is not None \
            else self._score_manufacturing(pair)
        pair.scores.business_manufacturing = mfg

        # ── 5. Clinical Adoption Score ────────────────────────────────────
        clin = override.clinical_adoption_score if override.clinical_adoption_score is not None \
            else self._score_clinical_adoption(pair)
        pair.scores.business_clinical_adoption = clin

        # ── 6. Speed to Revenue ───────────────────────────────────────────
        speed = override.speed_score if override.speed_score is not None \
            else self._score_speed(pair)
        pair.scores.business_speed_to_revenue = speed

        total = pair.scores.business_total
        logger.info(
            f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
            f"Business total = {total}/30 "
            f"(IP={ip}, Reg={reg}, Mkt={mkt}, Mfg={mfg}, Clin={clin}, Speed={speed})"
            + (" [BELOW THRESHOLD]" if total and total < self.MINIMUM_BUSINESS_SCORE else "")
        )

        return pair

    # ── Automated scoring methods ──────────────────────────────────────────────

    def _score_ip(self, pair: CandidatePair) -> int:
        """
        IP Score 1–5.
        5 = No prior method-of-use patent found, novel indication
        3 = Tangentially related patents exist but don't cover exact use
        1 = Existing patent directly covers this indication

        Note: Automated patent search is approximate.
        Bio team MUST manually verify before filing.
        """
        # Query Google Patents / USPTO API
        patent_hits = self._search_patents(pair.drug_name, pair.disease_name)

        if patent_hits == 0:
            return 5
        elif patent_hits <= 3:
            return 3
        else:
            return 1

    @cached_api_call(ttl_seconds=86400 * 30)
    def _search_patents(self, drug_name: str, disease_name: str) -> int:
        """
        Search USPTO Patents Full-Text for method-of-use patents.
        Returns approximate count of potentially conflicting patents.

        Note: This is a screening search. Legal review is required before filing.
        """
        # USPTO full-text search API
        url = "https://efts.uspto.gov/LATEST/search-index"
        query = f'"{drug_name}" AND "{disease_name}" AND "method of treatment"'
        params = {"q": query, "dateRangeField": "datePublished", "rows": 10}
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                return data.get("response", {}).get("numFound", 0)
        except Exception as e:
            logger.debug(f"Patent search failed: {e}")
        return 0   # Return 0 on failure (don't penalize on API error)

    def _score_regulatory(self, pair: CandidatePair) -> int:
        """
        Regulatory Score 1–5.
        5 = Approved in US + EU + 20-year clean safety record + Phase III data
        3 = Approved in one reference country, limited post-market data
        1 = Only approved in non-reference countries
        """
        # In production: query DrugBank approval data + FDA Orange Book
        # Simplified: check if drug appears in ChEMBL as approved
        # TODO: integrate DrugBank approved_in_countries field
        return 3   # Default: assume single-country approval; bio team overrides

    def _score_market(self, pair: CandidatePair) -> int:
        """
        Market Score 1–5 based on Indian patient population size.

        5 = 10,000–50,000 Indian patients (Goldilocks sweet spot)
        3 = 1,000–10,000 patients
        1 = Under 500 or over 100,000
        """
        india_patients = self._estimate_india_patients(pair.disease_id)
        if india_patients is None:
            return 2   # Unknown — conservative

        if PATIENT_SWEET_SPOT_MIN <= india_patients <= PATIENT_SWEET_SPOT_MAX:
            return 5
        elif PATIENT_MIN <= india_patients < PATIENT_SWEET_SPOT_MIN:
            return 3
        elif india_patients > PATIENT_SWEET_SPOT_MAX and india_patients <= PATIENT_MAX:
            return 3
        elif india_patients < PATIENT_MIN:
            return 1
        else:   # > 100,000
            return 1

    @cached_api_call(ttl_seconds=86400 * 90)
    def _estimate_india_patients(self, disease_id: str) -> Optional[int]:
        """
        Estimate Indian patient count from Orphanet global prevalence.
        India ≈ 17.5% of global population.
        """
        orpha_id = disease_id.replace("ORPHA:", "")
        url = f"{ORPHANET_API}/{orpha_id}/Prevalence"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                return None
            data = r.json()
            prevalences = data.get("items", [])
            for p in prevalences:
                val_per_million = p.get("ValMoy")   # average per million
                if val_per_million:
                    global_patients = (float(val_per_million) / 1_000_000) * 8_000_000_000
                    india_patients = int(global_patients * 0.175)
                    return india_patients
        except Exception as e:
            logger.debug(f"Prevalence fetch failed for {disease_id}: {e}")
        return None

    def _score_manufacturing(self, pair: CandidatePair) -> int:
        """
        Manufacturing Score 1–5.
        5 = Oral tablet, API manufactured in India, multiple Indian CMOs
        3 = Oral capsule/simple injectable, limited Indian manufacturers
        1 = Complex injectable, biologic, cold-chain required

        In production: query DrugBank route of administration + Indian CMO database.
        """
        # Placeholder: bio team should override with manual assessment
        # Default assumes oral formulation (filtered upstream in drug universe)
        return 4   # Oral assumed; bio team downgrades if cold-chain etc.

    def _score_clinical_adoption(self, pair: CandidatePair) -> int:
        """
        Clinical Adoption Score 1–5.
        5 = Indian physicians already using off-label, Indian case reports published
        3 = Off-label documented internationally but not India-specific
        1 = Theoretical only, no real-world adoption

        Inferred from Layer 5 literature signal.
        """
        # Use case report count from Layer 5 as a proxy
        case_count = pair.scores.case_report_count
        pubmed_score = pair.scores.pubmed_cooccurrence_score

        if case_count is None:
            return 1

        if case_count >= 5:
            return 5
        elif case_count >= 2:
            return 4
        elif case_count == 1:
            return 3
        elif pubmed_score and pubmed_score > 5:
            return 3
        else:
            return 1

    def _score_speed(self, pair: CandidatePair) -> int:
        """
        Speed to Revenue Score 1–5.
        5 = Drug known to CDSCO (approved in India for another indication),
            patient advocacy group ready, trial sites pre-identified
        3 = Strong international data, some physician interest
        1 = Needs significant groundwork

        In production: check CDSCO database for existing Indian approval.
        """
        # Proxy: if Phase II+ trial exists, regulatory pathway is clearer
        ct_evidence = pair.scores.clinical_trial_evidence
        if ct_evidence is None:
            return 1

        if ct_evidence >= 4:
            return 5
        elif ct_evidence >= 3:
            return 3
        elif ct_evidence >= 2:
            return 2
        else:
            return 1
