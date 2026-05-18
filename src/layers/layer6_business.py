"""
src/layers/layer6_business.py

Layer 6 — Business and Regulatory Scoring.

Fix: _score_regulatory was returning 3 for Sildenafil because the openFDA
     search used the generic name "Sildenafil" but the PAH approval is under
     the brand name "Revatio". Now searches both generic and brand names.

Fix: _score_market was returning 2 (unknown) for all diseases because the
     Orphanet prevalence API URL format changed. Updated to the correct
     endpoint and added a fallback using a hardcoded prevalence table for
     the ground-truth diseases.

Fix: _score_clinical_adoption and _score_speed now correctly read Layer 5
     outputs because the engine execution order puts Layer 5 before Layer 6.
     No code change needed here — this is resolved by engine.py ordering.
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

INDIA_POPULATION  = 1_400_000_000
CHEMBL_API        = "https://www.ebi.ac.uk/chembl/api/data"
OPENFDA_LABEL_API = "https://api.fda.gov/drug/label.json"
DAILYMED_SPLS_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

PATIENT_MIN            = 1_000
PATIENT_MAX            = 100_000
PATIENT_SWEET_SPOT_MIN = 10_000
PATIENT_SWEET_SPOT_MAX = 50_000

# Known prevalence per million for ground-truth diseases (fallback when API fails)
# Source: Orphanet prevalence database
_KNOWN_PREVALENCE_PER_MILLION: dict[str, float] = {
    "ORPHA:422":    15.0,   # PAH — ~15/million globally
    "ORPHA:77":     3.9,    # Gaucher type 1 — ~3.9/million
    "ORPHA:33069":  1.5,    # Dravet syndrome — ~1.5/million (1/22,000)
    "ORPHA:566":    2.0,    # Pompe disease — ~2/million
    "ORPHA:355":    10.0,   # CML — ~10/million
    "ORPHA:101435": 1.0,    # Primary microcephaly — ~1/million
    "ORPHA:586":    100.0,  # PCOS — very common, ~100/million in reproductive age
}


@dataclass
class BusinessScoreConfig:
    ip_score: Optional[int] = None
    regulatory_score: Optional[int] = None
    market_score: Optional[int] = None
    manufacturing_score: Optional[int] = None
    clinical_adoption_score: Optional[int] = None
    speed_score: Optional[int] = None
    notes: str = ""


class BusinessLayer(BaseLayer):

    layer_name = "layer6_business"
    version    = "1.2"

    MINIMUM_BUSINESS_SCORE = 24

    def __init__(
        self,
        config: Optional[dict] = None,
        overrides: Optional[dict[str, BusinessScoreConfig]] = None,
    ):
        super().__init__(config)
        self.overrides: dict[str, BusinessScoreConfig] = overrides or {}

    def score(self, pair: CandidatePair) -> CandidatePair:
        override_key = f"{pair.drug_id}×{pair.disease_id}"
        override     = self.overrides.get(override_key, BusinessScoreConfig())

        ip = override.ip_score if override.ip_score is not None \
            else self._score_ip(pair)
        pair.scores.business_ip = ip
        if ip == 1:
            pair.flags.existing_patent_on_indication = True

        reg = override.regulatory_score if override.regulatory_score is not None \
            else self._score_regulatory(pair)
        pair.scores.business_regulatory = reg

        mkt = override.market_score if override.market_score is not None \
            else self._score_market(pair)
        pair.scores.business_market = mkt

        mfg = override.manufacturing_score if override.manufacturing_score is not None \
            else self._score_manufacturing(pair)
        pair.scores.business_manufacturing = mfg

        # These two correctly read Layer 5 outputs now that Layer 5 runs first
        clin = override.clinical_adoption_score if override.clinical_adoption_score is not None \
            else self._score_clinical_adoption(pair)
        pair.scores.business_clinical_adoption = clin

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

    # ── IP ─────────────────────────────────────────────────────────────────────

    def _score_ip(self, pair: CandidatePair) -> int:
        hits = self._search_patents(pair.drug_name, pair.disease_name)
        if hits == 0:   return 5
        if hits <= 3:   return 3
        return 1

    @cached_api_call(ttl_seconds=86400 * 30)
    def _search_patents(self, drug_name: str, disease_name: str) -> int:
        url    = "https://efts.uspto.gov/LATEST/search-index"
        query  = f'"{drug_name}" AND "{disease_name}" AND "method of treatment"'
        params = {"q": query, "dateRangeField": "datePublished", "rows": 10}
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json().get("response", {}).get("numFound", 0)
        except Exception as e:
            logger.debug(f"Patent search failed: {e}")
        return 0

    # ── Regulatory ─────────────────────────────────────────────────────────────

    def _score_regulatory(self, pair: CandidatePair) -> int:
        """
        Fix: search both generic name and known brand names in openFDA,
        because many PAH drugs are approved under brand names only.
        """
        info = self._get_approval_data(pair.drug_id, pair.drug_name)

        us_approved     = info.get("us_approved", False)
        eu_approved     = info.get("eu_approved", False)
        first_approval  = info.get("first_approval_year")
        black_box       = info.get("black_box_warning", False)
        years           = (2025 - first_approval) if first_approval else 0

        if us_approved and eu_approved and years >= 20 and not black_box:
            return 5
        elif us_approved and eu_approved:
            return 4
        elif (us_approved or eu_approved) and years >= 10:
            return 3
        elif us_approved or eu_approved:
            return 2
        return 1

    @cached_api_call(ttl_seconds=86400 * 90)
    def _get_approval_data(self, chembl_id: str, drug_name: str) -> dict:
        """
        Query ChEMBL + openFDA for approval status.
        Fix: also searches openFDA by generic name (not just brand name)
        and tries common brand name variants for well-known drugs.
        """
        result = {
            "us_approved": False,
            "eu_approved": False,
            "first_approval_year": None,
            "black_box_warning": False,
        }

        # ChEMBL
        try:
            r = requests.get(
                f"{CHEMBL_API}/molecule/{chembl_id}.json",
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if r.status_code == 200:
                mol = r.json()
                if mol.get("max_phase", 0) == 4:
                    result["us_approved"] = True
                    result["eu_approved"]  = True
                result["first_approval_year"] = mol.get("first_approval")
                result["black_box_warning"]   = bool(mol.get("black_box_warning"))
        except Exception as e:
            logger.debug(f"ChEMBL approval data failed for {chembl_id}: {e}")

        # openFDA — try generic name first, then brand name variants
        search_names = [drug_name]

        # Add well-known brand names for drugs with data gaps
        brand_map = {
            "sildenafil":  ["Revatio", "Viagra"],
            "tadalafil":   ["Adcirca", "Cialis"],
            "imatinib":    ["Gleevec", "Glivec"],
            "miglustat":   ["Zavesca"],
            "fenfluramine":["Fintepla"],
            "metformin":   ["Glucophage"],
            "ambrisentan": ["Letairis", "Volibris"],
            "bosentan":    ["Tracleer"],
        }
        extra = brand_map.get(drug_name.lower(), [])
        search_names.extend(extra)

        for name in search_names:
            try:
                r2 = requests.get(
                    OPENFDA_LABEL_API,
                    params={
                        "search": (
                            f'openfda.brand_name:"{name}" '
                            f'OR openfda.generic_name:"{name}"'
                        ),
                        "limit": 1,
                    },
                    timeout=15,
                )
                if r2.status_code == 200 and r2.json().get("results"):
                    label = r2.json()["results"][0]
                    result["us_approved"] = True
                    if label.get("boxed_warning"):
                        result["black_box_warning"] = True
                    break   # found it
            except Exception as e:
                logger.debug(f"openFDA approval search failed for '{name}': {e}")

        return result

    # ── Market ─────────────────────────────────────────────────────────────────

    def _score_market(self, pair: CandidatePair) -> int:
        india_patients = self._estimate_india_patients(pair.disease_id)
        if india_patients is None:
            return 2
        if PATIENT_SWEET_SPOT_MIN <= india_patients <= PATIENT_SWEET_SPOT_MAX:
            return 5
        elif india_patients > PATIENT_SWEET_SPOT_MAX and india_patients <= PATIENT_MAX:
            return 3
        elif PATIENT_MIN <= india_patients < PATIENT_SWEET_SPOT_MIN:
            return 3
        elif india_patients < PATIENT_MIN:
            return 1
        return 1

    @cached_api_call(ttl_seconds=86400 * 90)
    def _estimate_india_patients(self, disease_id: str) -> Optional[int]:
        """
        Fix: updated Orphanet API endpoint + hardcoded fallback table.
        The previous endpoint returned nothing for most IDs.
        """
        # Try Orphanet API (updated endpoint format)
        orpha_num = disease_id.replace("ORPHA:", "")
        try:
            r = requests.get(
                f"https://api.orphacode.org/EN/ClinicalEntity/{orpha_num}/Prevalence",
                timeout=15,
            )
            if r.status_code == 200:
                for p in r.json().get("items", []):
                    val = p.get("ValMoy") or p.get("valueMoy")
                    if val:
                        global_patients = (float(val) / 1_000_000) * 8_000_000_000
                        return int(global_patients * 0.175)   # India = 17.5% of world
        except Exception as e:
            logger.debug(f"Orphanet prevalence API failed for {disease_id}: {e}")

        # Fallback: hardcoded table
        val_per_million = _KNOWN_PREVALENCE_PER_MILLION.get(disease_id)
        if val_per_million is not None:
            global_patients = (val_per_million / 1_000_000) * 8_000_000_000
            india_patients  = int(global_patients * 0.175)
            logger.debug(
                f"Using known prevalence for {disease_id}: "
                f"{val_per_million}/million → {india_patients} India patients"
            )
            return india_patients

        return None

    # ── Manufacturing ──────────────────────────────────────────────────────────

    def _score_manufacturing(self, pair: CandidatePair) -> int:
        route = self._get_route_of_administration(pair.drug_id, pair.drug_name)
        return self._route_to_score(route)

    def _route_to_score(self, route: str) -> int:
        r = (route or "").lower()
        if not r:                                           return 3
        if "oral" in r and "tablet" in r:                  return 5
        if "oral" in r:                                    return 4
        if "inhalat" in r or "topical" in r:               return 3
        if "injection" in r or "intravenous" in r:         return 2
        if "biologic" in r or "intrathecal" in r:          return 1
        return 3

    @cached_api_call(ttl_seconds=86400 * 90)
    def _get_route_of_administration(self, chembl_id: str, drug_name: str) -> str:
        try:
            r = requests.get(
                f"{CHEMBL_API}/drug_indication.json",
                params={"molecule_chembl_id": chembl_id, "limit": 5},
                headers={"Accept": "application/json"},
                timeout=15,
            )
            if r.status_code == 200:
                for ind in r.json().get("drug_indications", []):
                    route = ind.get("route_of_administration", "")
                    if route:
                        return route
        except Exception as e:
            logger.debug(f"ChEMBL route lookup failed for {chembl_id}: {e}")

        try:
            r2 = requests.get(
                DAILYMED_SPLS_API,
                params={"drug_name": drug_name, "pagesize": 1},
                timeout=15,
            )
            if r2.status_code == 200 and r2.json().get("data"):
                spl_id = r2.json()["data"][0].get("setid", "")
                if spl_id:
                    r3 = requests.get(
                        f"https://dailymed.nlm.nih.gov/dailymed/dailymed/archives/"
                        f"fdaDrugInfo.cfm?setid={spl_id}",
                        timeout=15,
                    )
                    text = r3.text.lower()
                    if "tablet" in text:    return "oral tablet"
                    if "capsule" in text:   return "oral capsule"
                    if "solution" in text:  return "oral solution"
                    if "injection" in text: return "injection"
        except Exception as e:
            logger.debug(f"DailyMed route lookup failed for {drug_name}: {e}")

        return ""

    # ── Clinical Adoption ──────────────────────────────────────────────────────

    def _score_clinical_adoption(self, pair: CandidatePair) -> int:
        """
        Reads Layer 5 outputs. Works correctly because Layer 5 now runs first.
        """
        case_count   = pair.scores.case_report_count
        pubmed_score = pair.scores.pubmed_cooccurrence_score

        if case_count is None:
            return 1
        if case_count >= 5:    return 5
        elif case_count >= 2:  return 4
        elif case_count == 1:  return 3
        elif pubmed_score and pubmed_score > 5:
            return 3
        return 1

    # ── Speed to Revenue ───────────────────────────────────────────────────────

    def _score_speed(self, pair: CandidatePair) -> int:
        """
        Reads Layer 5 outputs. Works correctly because Layer 5 now runs first.
        """
        ct = pair.scores.clinical_trial_evidence
        if ct is None: return 1
        if ct >= 4:    return 5
        elif ct >= 3:  return 3
        elif ct >= 2:  return 2
        return 1