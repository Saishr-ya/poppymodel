"""
src/layers/layer4_admet.py

Layer 4 — ADMET Scoring (Absorption, Distribution, Metabolism, Excretion, Toxicity).

For repurposed drugs, ADMET is partially solved (human safety data exists).
This layer focuses on:
  1. Hard disqualifiers that eliminate candidates before expensive computation
  2. A composite safety score for ranking

Hard disqualifiers (set pair.flags):
  - Oral bioavailability < 20%
  - hERG IC50 < 1 µM (high cardiotoxicity risk)
  - FAERS ROR > 3 for serious adverse events
  - Lipinski violations > 1

Data sources:
  - DrugBank (bioavailability, CYP, hERG — requires API key or XML dump)
  - pkCSM web tool (https://biosig.unimelb.edu.au/pkcsm) — free, no key needed
  - ADMETlab 3.0 (https://admet.scbdd.com)
  - FDA FAERS via openFDA API (https://open.fda.gov/apis/drug/event/)

Bio team notes:
  - For repurposed drugs, existing drug labels are the gold standard for ADMET.
    DrugBank "pharmacokinetics" tab has this data for most approved drugs.
  - hERG flag is critical for cardiac rare disease patients who may already
    have compromised cardiac function.
  - FAERS ROR > 3 is a hard cutoff only for SERIOUS adverse events (hospitalization,
    death, life-threatening). Minor adverse events don't disqualify.
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

OPENFDA_BASE = "https://api.fda.gov/drug/event.json"
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")


class FAERSClient:
    """
    Queries FDA FAERS via openFDA API for adverse event signals.
    Computes Reporting Odds Ratio (ROR) for serious adverse events.

    ROR = (a/b) / (c/d) where:
        a = drug + event reports
        b = drug + no event reports
        c = all drugs + event reports
        d = all drugs + no event reports
    """

    @cached_api_call(ttl_seconds=86400 * 14)   # 14-day cache
    def get_serious_event_ror(self, drug_name: str) -> Optional[float]:
        """
        Compute ROR for serious adverse events (outcomes: death, hospitalization,
        life-threatening) for a given drug.

        Returns:
            ROR value (float) or None if insufficient data.
            ROR > 3 = high signal (disqualifying flag).
        """
        params = {
            "search": f'patient.drug.openfda.brand_name:"{drug_name}"'
                      ' AND serious:1',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 10,
        }
        if OPENFDA_API_KEY:
            params["api_key"] = OPENFDA_API_KEY

        try:
            # Count serious reports for this drug
            r_drug = requests.get(
                OPENFDA_BASE,
                params={
                    "search": f'patient.drug.openfda.brand_name:"{drug_name}" AND serious:1',
                    "limit": 1,
                },
                timeout=15,
            )
            if r_drug.status_code == 200:
                serious_count = r_drug.json().get("meta", {}).get("results", {}).get("total", 0)
            else:
                return None

            # Count total reports for this drug
            r_total = requests.get(
                OPENFDA_BASE,
                params={
                    "search": f'patient.drug.openfda.brand_name:"{drug_name}"',
                    "limit": 1,
                },
                timeout=15,
            )
            if r_total.status_code != 200:
                return None
            total_count = r_total.json().get("meta", {}).get("results", {}).get("total", 1)

            # Simplified ROR using serious event proportion vs background 10% rate
            if total_count == 0:
                return None
            serious_proportion = serious_count / total_count
            background_proportion = 0.10   # ~10% baseline serious event rate in FAERS

            if background_proportion == 0:
                return None
            ror = (serious_proportion / (1 - serious_proportion)) / \
                  (background_proportion / (1 - background_proportion))
            return float(ror)

        except Exception as e:
            logger.warning(f"FAERS ROR computation failed for {drug_name}: {e}")
            return None


class ADMETLayer(BaseLayer):
    """
    Layer 4 — ADMET safety scoring and hard disqualifier flagging.

    Scores:
        pair.scores.admet_composite         (0–1, higher = safer)
        pair.scores.oral_bioavailability_pct
        pair.scores.bcs_class
        pair.scores.herg_ic50_um

    Flags (hard disqualifiers):
        pair.flags.bioavailability_insufficient   (BA < 20%)
        pair.flags.herg_risk_high                 (hERG IC50 < 1 µM)
        pair.flags.faers_ror_critical             (ROR > 3)
        pair.flags.lipinski_violations            (count)
    """

    layer_name = "layer4_admet"
    version = "1.0"

    HERG_RISK_THRESHOLD_UM = 1.0        # µM — below this = high risk
    BIOAVAILABILITY_MIN_PCT = 20.0      # % — below this = disqualify
    FAERS_ROR_THRESHOLD = 3.0           # ROR above this = disqualify

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.chembl = ChEMBLClient()
        self.faers = FAERSClient()

    def score(self, pair: CandidatePair) -> CandidatePair:
        mol = self.chembl.get_molecule(pair.drug_id)
        if not mol:
            logger.warning(f"[{self.layer_name}] No molecule data for {pair.drug_id}")
            return pair

        props = mol.get("molecule_properties") or {}

        # ── 1. Lipinski Rule of 5 ─────────────────────────────────────────
        lipo = self.chembl.lipinski_check(pair.drug_id)
        pair.flags.lipinski_violations = lipo.get("violations", 0)

        # ── 2. Oral Bioavailability ───────────────────────────────────────
        # DrugBank has this; ChEMBL has a binary "oral" flag but not %.
        # In production, pull from DrugBank XML or label.
        # Here we use the ChEMBL oral flag as a proxy.
        is_oral = mol.get("oral", False)
        if not is_oral:
            pair.flags.bioavailability_insufficient = True
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}: not oral — disqualifying"
            )

        # ── 3. hERG Cardiotoxicity ────────────────────────────────────────
        # In production, pull from CardioToxDB or PDSP Ki database.
        # Placeholder: query ChEMBL bioactivity for hERG (CHEMBL240)
        herg_ic50 = self._get_herg_ic50(pair.drug_id)
        pair.scores.herg_ic50_um = herg_ic50
        if herg_ic50 is not None and herg_ic50 < self.HERG_RISK_THRESHOLD_UM:
            pair.flags.herg_risk_high = True
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}: hERG IC50={herg_ic50:.2f}µM — HIGH RISK"
            )

        # ── 4. FAERS Adverse Event Signal ────────────────────────────────
        ror = self.faers.get_serious_event_ror(pair.drug_name)
        if ror is not None and ror > self.FAERS_ROR_THRESHOLD:
            pair.flags.faers_ror_critical = True
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}: FAERS ROR={ror:.1f} — CRITICAL"
            )

        # ── 5. BCS Classification (from DrugBank / DailyMed in production) ──
        # Placeholder: derive rough BCS class from available properties
        mw   = float(props.get("mw_freebase") or 0)
        logp = float(props.get("alogp") or 0)
        bcs_class = self._estimate_bcs_class(mw, logp)
        pair.scores.bcs_class = bcs_class

        # ── 6. Composite ADMET Score ──────────────────────────────────────
        pair.scores.admet_composite = self._composite_score(pair, lipo, herg_ic50, ror)

        return pair

    def _get_herg_ic50(self, chembl_id: str) -> Optional[float]:
        """
        Query ChEMBL bioactivity for hERG (CHEMBL240) IC50 values.
        Returns minimum IC50 in µM (most pessimistic/conservative).
        """
        url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {
            "molecule_chembl_id": chembl_id,
            "target_chembl_id": "CHEMBL240",   # hERG potassium channel
            "standard_type": "IC50",
            "limit": 20,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            activities = r.json().get("activities", [])
            ic50_values = []
            for a in activities:
                val = a.get("standard_value")
                units = a.get("standard_units", "")
                if val and "nM" in units:
                    ic50_values.append(float(val) / 1000)  # nM → µM
                elif val and "uM" in units:
                    ic50_values.append(float(val))
            return min(ic50_values) if ic50_values else None
        except Exception as e:
            logger.debug(f"hERG IC50 lookup failed for {chembl_id}: {e}")
            return None

    def _estimate_bcs_class(self, mw: float, logp: float) -> str:
        """
        Rough BCS classification from physicochemical properties.
        High solubility: logP < 1 or MW < 300
        High permeability: logP > 1.5
        In production, use DrugBank solubility data + Caco-2 permeability.
        """
        high_sol = logp < 1.0 or mw < 300
        high_perm = logp > 1.5

        if high_sol and high_perm:
            return "I"
        elif not high_sol and high_perm:
            return "II"
        elif high_sol and not high_perm:
            return "III"
        else:
            return "IV"

    def _composite_score(
        self,
        pair: CandidatePair,
        lipinski: dict,
        herg_ic50: Optional[float],
        ror: Optional[float],
    ) -> float:
        """
        Compute a normalized 0–1 ADMET composite score.
        1.0 = safest possible profile.

        Penalize:
            - Each Lipinski violation: -0.1
            - hERG IC50 < 10µM: -0.2
            - hERG IC50 < 1µM:  disqualified (handled by flag)
            - FAERS ROR > 2:    -0.15
            - BCS Class IV:     -0.2
            - BCS Class II/III: -0.1
        """
        score = 1.0
        score -= lipinski.get("violations", 0) * 0.10
        if herg_ic50 is not None:
            if herg_ic50 < 10:
                score -= 0.20
        if ror is not None and ror > 2.0:
            score -= 0.15
        if pair.scores.bcs_class == "IV":
            score -= 0.20
        elif pair.scores.bcs_class in ("II", "III"):
            score -= 0.10

        return max(0.0, min(1.0, score))
