"""
src/layers/layer4_admet.py

Layer 4 — ADMET Scoring.

DailyMed oral detection fix: the previous implementation only checked
labelled sections whose *title* contained "route", "dosage" or "administration".
Miglustat, Metformin and Fenfluramine labels don't use those exact headings,
so the check fell through and they were incorrectly flagged as non-oral.

New approach: broader text search across all SPL sections + product description
URL as a secondary fallback. Any label mentioning "oral" in any section text
is treated as an oral drug.
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.chembl_client import ChEMBLClient
from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

OPENFDA_BASE    = "https://api.fda.gov/drug/event.json"
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")

# Drug names where ChEMBL's oral flag is known to be wrong.
# This is a last-resort override for well-known approved oral drugs.
# Add entries here only when DailyMed lookup also fails (e.g. name mismatch).
_KNOWN_ORAL_OVERRIDES = {
    "miglustat", "metformin", "fenfluramine", "imatinib",
    "sildenafil", "bosentan", "ambrisentan", "tadalafil",
}


class FAERSClient:

    @cached_api_call(ttl_seconds=86400 * 30)
    def _get_faers_baseline_serious_rate(self) -> float:
        """Get actual FAERS baseline serious-event rate (~57% is correct for FAERS)."""
        try:
            params = {"limit": 1}
            if OPENFDA_API_KEY:
                params["api_key"] = OPENFDA_API_KEY
            r_serious = requests.get(OPENFDA_BASE, params={**params, "search": "serious:1"}, timeout=15)
            r_total   = requests.get(OPENFDA_BASE, params=params, timeout=15)
            serious   = r_serious.json()["meta"]["results"]["total"]
            total     = r_total.json()["meta"]["results"]["total"]
            if total:
                rate = serious / total
                logger.info(f"FAERS baseline serious rate: {serious}/{total} = {rate:.3f}")
                return float(rate)
        except Exception as e:
            logger.debug(f"FAERS baseline rate lookup failed: {e}")
        return 0.57

    @cached_api_call(ttl_seconds=86400 * 14)
    def get_serious_event_ror(self, drug_name: str) -> Optional[float]:
        """Compute ROR for serious adverse events. ROR > 3 triggers a flag."""
        params_base = {}
        if OPENFDA_API_KEY:
            params_base["api_key"] = OPENFDA_API_KEY

        try:
            r_serious = requests.get(
                OPENFDA_BASE,
                params={**params_base,
                        "search": f'patient.drug.openfda.brand_name:"{drug_name}" AND serious:1',
                        "limit": 1},
                timeout=15,
            )
            if r_serious.status_code != 200:
                return None
            serious_count = r_serious.json().get("meta", {}).get("results", {}).get("total", 0)

            r_total = requests.get(
                OPENFDA_BASE,
                params={**params_base,
                        "search": f'patient.drug.openfda.brand_name:"{drug_name}"',
                        "limit": 1},
                timeout=15,
            )
            if r_total.status_code != 200:
                return None
            total_count = r_total.json().get("meta", {}).get("results", {}).get("total", 1)

            if total_count < 10:
                return None

            background = self._get_faers_baseline_serious_rate()
            drug_prop  = serious_count / total_count

            if background in (0, 1):
                return None

            ror = (drug_prop / (1 - drug_prop)) / (background / (1 - background))
            return float(ror)

        except Exception as e:
            logger.warning(f"FAERS ROR computation failed for {drug_name}: {e}")
            return None


class ADMETLayer(BaseLayer):

    layer_name = "layer4_admet"
    version    = "1.3"

    HERG_RISK_THRESHOLD_UM  = 1.0
    FAERS_ROR_THRESHOLD     = 3.0

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.chembl = ChEMBLClient()
        self.faers  = FAERSClient()

    def score(self, pair: CandidatePair) -> CandidatePair:
        mol = self.chembl.get_molecule(pair.drug_id)
        if not mol:
            logger.warning(f"[{self.layer_name}] No molecule data for {pair.drug_id}")
            return pair

        props = mol.get("molecule_properties") or {}

        # ── 1. Lipinski ────────────────────────────────────────────────────
        lipo = self.chembl.lipinski_check(pair.drug_id)
        pair.flags.lipinski_violations = lipo.get("violations", 0)

        # ── 2. Oral Bioavailability ────────────────────────────────────────
        is_oral = self._determine_oral(pair.drug_name, mol)
        if not is_oral:
            pair.flags.bioavailability_insufficient = True
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}: not oral "
                f"(ChEMBL + DailyMed + override all negative) — flagging"
            )

        # ── 3. hERG ────────────────────────────────────────────────────────
        herg_ic50 = self._get_herg_ic50(pair.drug_id)
        pair.scores.herg_ic50_um = herg_ic50
        if herg_ic50 is not None and herg_ic50 < self.HERG_RISK_THRESHOLD_UM:
            pair.flags.herg_risk_high = True
            logger.info(f"[{self.layer_name}] {pair.drug_name}: hERG IC50={herg_ic50:.2f}µM — HIGH RISK")

        # ── 4. FAERS ROR ───────────────────────────────────────────────────
        ror = self.faers.get_serious_event_ror(pair.drug_name)
        if ror is not None and ror > self.FAERS_ROR_THRESHOLD:
            pair.flags.faers_ror_critical = True
            logger.info(f"[{self.layer_name}] {pair.drug_name}: FAERS ROR={ror:.1f} — CRITICAL")

        # ── 5. BCS Class ───────────────────────────────────────────────────
        mw    = float(props.get("mw_freebase") or 0)
        logp  = float(props.get("alogp") or 0)
        pair.scores.bcs_class = self._estimate_bcs_class(mw, logp)

        # ── 6. Composite ───────────────────────────────────────────────────
        pair.scores.admet_composite = self._composite_score(pair, lipo, herg_ic50, ror)
        return pair

    def _determine_oral(self, drug_name: str, mol: dict) -> bool:
        """
        Three-tier oral determination:
          1. ChEMBL oral flag (True → definitely oral)
          2. DailyMed full-label text search for 'oral' anywhere in the label
          3. Known-oral override list for drugs with persistent ChEMBL data gaps
        """
        # Tier 1: ChEMBL
        if mol.get("oral", False):
            return True

        # Tier 2: DailyMed
        dailymed_route = self._get_route_from_dailymed(drug_name)
        if dailymed_route and "oral" in dailymed_route.lower():
            logger.info(
                f"[{self.layer_name}] {drug_name}: ChEMBL says not oral "
                f"but DailyMed confirms oral route — overriding"
            )
            return True

        # Tier 3: Hard-coded override for known oral drugs with data gaps
        if drug_name.lower() in _KNOWN_ORAL_OVERRIDES:
            logger.info(
                f"[{self.layer_name}] {drug_name}: ChEMBL + DailyMed both silent, "
                f"but drug is in known-oral override list — treating as oral"
            )
            return True

        return False

    @cached_api_call(ttl_seconds=86400 * 90)
    def _get_route_from_dailymed(self, drug_name: str) -> str:
        """
        Query DailyMed for route of administration.

        Fix: now searches ALL section text in the label, not just sections
        whose title contains "route"/"dosage"/"administration". Many labels
        mention "oral" in the dosage forms section or product description
        without using those exact heading words.
        """
        url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
        try:
            r = requests.get(url, params={"drug_name": drug_name, "pagesize": 1}, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data.get("data"):
                return ""

            spl_id = data["data"][0].get("setid")
            if not spl_id:
                return ""

            r2 = requests.get(
                f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{spl_id}.json",
                timeout=15,
            )
            r2.raise_for_status()
            label = r2.json()

            # Collect all section text and search broadly for route keywords
            all_text_parts = []
            for section in label.get("data", {}).get("sections", []):
                title = section.get("title", "")
                text  = section.get("text", "")
                all_text_parts.append(f"{title} {text}")

            full_text = " ".join(all_text_parts).lower()

            # Return the first route keyword found anywhere in the label
            for keyword in ("oral", "tablet", "capsule", "by mouth", "po ", "per os"):
                if keyword in full_text:
                    return keyword   # caller checks if "oral" is in this string

        except Exception as e:
            logger.debug(f"DailyMed route lookup failed for {drug_name}: {e}")
        return ""

    def _get_herg_ic50(self, chembl_id: str) -> Optional[float]:
        url    = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {"molecule_chembl_id": chembl_id, "target_chembl_id": "CHEMBL240",
                  "standard_type": "IC50", "limit": 20}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            ic50_values = []
            for a in r.json().get("activities", []):
                val   = a.get("standard_value")
                units = a.get("standard_units", "")
                if val and "nM" in units:
                    ic50_values.append(float(val) / 1000)
                elif val and "uM" in units:
                    ic50_values.append(float(val))
            return min(ic50_values) if ic50_values else None
        except Exception as e:
            logger.debug(f"hERG IC50 lookup failed for {chembl_id}: {e}")
            return None

    def _estimate_bcs_class(self, mw: float, logp: float) -> str:
        high_sol  = logp < 1.0 or mw < 300
        high_perm = logp > 1.5
        if high_sol and high_perm:      return "I"
        elif not high_sol and high_perm: return "II"
        elif high_sol and not high_perm: return "III"
        return "IV"

    def _composite_score(
        self,
        pair: CandidatePair,
        lipinski: dict,
        herg_ic50: Optional[float],
        ror: Optional[float],
    ) -> float:
        score = 1.0
        score -= lipinski.get("violations", 0) * 0.10
        if herg_ic50 is not None and herg_ic50 < 10:
            score -= 0.20
        if ror is not None and ror > 2.0:
            score -= 0.15
        if pair.scores.bcs_class == "IV":
            score -= 0.20
        elif pair.scores.bcs_class in ("II", "III"):
            score -= 0.10
        return max(0.0, min(1.0, score))