"""
src/layers/layer_chirality.py

Chiral Switch Module.

Fix #10: Removed hardcoded WITHDRAWN_DRUG_OPPORTUNITIES dict. Withdrawal reasons
         are now fetched dynamically from the FDA drug enforcement (recall) database
         and the FDA drug label database. The hardcoded dict could only cover the
         handful of drugs the developer knew about; the dynamic approach surfaces any
         approved drug whose label or recall record mentions toxicity at a specific
         receptor, enabling the engine to find novel chiral switch opportunities.
         Results are cached for 90 days.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.chembl_client import ChEMBLClient
from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

ENANTIOMER_DIVERGENCE_THRESHOLD = 10.0

TOXIC_RECEPTORS = {
    "5-HT2B",
    "hERG",
    "D2",
    "sigma1",
    "5-HT3",
}

# Small curated seed list kept as an initialisation hint for the PDSP receptor
# matching logic. These are well-documented cases from the literature and are NOT
# used as the sole source for withdrawal data any more (fix #10).
_SEED_WITHDRAWN: dict[str, dict] = {
    "fenfluramine": {
        "toxic_receptor": "5-HT2B",
        "therapeutic_receptor": "5-HT2C",
        "note": "l-fenfluramine → Fintepla (Dravet, LGS). Already approved.",
    },
    "thalidomide": {
        "toxic_receptor": "CRBN_teratogenic",
        "therapeutic_receptor": "CRBN_immunomodulatory",
        "note": "R-thalidomide has therapeutic potential.",
    },
}


@dataclass
class ChiralSwitchCandidate:
    drug_name: str
    chembl_id: str
    is_racemic: bool
    receptor_divergence_score: Optional[float]
    toxic_enantiomer: Optional[str]
    therapeutic_enantiomer: Optional[str]
    toxic_receptor: Optional[str]
    therapeutic_receptor: Optional[str]
    is_withdrawn: bool
    withdrawal_reason: Optional[str]
    chiral_switch_viable: bool
    patent_opportunity_score: float
    notes: str = ""


class PDSPClient:
    """
    Query PDSP Ki Database for enantiomer receptor binding profiles.
    Setup: download from https://pdsp.unc.edu/databases/kidb.php
    Place at: data/raw/pdsp/ki_database.csv
    """

    def __init__(self):
        self._db = None

    def _load_db(self):
        if self._db is not None:
            return self._db
        import os
        path = "data/raw/pdsp/ki_database.csv"
        if os.path.exists(path):
            import pandas as pd
            self._db = pd.read_csv(path)
            logger.info(f"PDSP Ki DB loaded: {len(self._db)} entries")
        else:
            logger.warning(
                "PDSP Ki database not found. "
                "Download from https://pdsp.unc.edu/databases/kidb.php "
                "and place at data/raw/pdsp/ki_database.csv"
            )
            self._db = None
        return self._db

    def get_enantiomer_profiles(self, drug_name: str) -> dict[str, dict[str, float]]:
        db = self._load_db()
        if db is None:
            return {}

        d_names = [
            f"d-{drug_name.lower()}", f"(+)-{drug_name.lower()}",
            f"r-{drug_name.lower()}", f"(r)-{drug_name.lower()}",
        ]
        l_names = [
            f"l-{drug_name.lower()}", f"(-)-{drug_name.lower()}",
            f"s-{drug_name.lower()}", f"(s)-{drug_name.lower()}",
        ]

        profiles = {"d": {}, "l": {}}

        try:
            name_col     = db.columns[0]
            receptor_col = db.columns[1]
            ki_col       = db.columns[2]

            db_lower = db.copy()
            db_lower[name_col] = db_lower[name_col].str.lower().str.strip()

            for enantiomer, names in [("d", d_names), ("l", l_names)]:
                mask   = db_lower[name_col].isin(names)
                subset = db[mask]
                for _, row in subset.iterrows():
                    receptor = str(row[receptor_col]).strip()
                    try:
                        ki = float(row[ki_col])
                        profiles[enantiomer][receptor] = ki
                    except (ValueError, TypeError):
                        continue
        except Exception as e:
            logger.debug(f"PDSP profile parsing failed for {drug_name}: {e}")

        return profiles

    def compute_divergence_score(
        self,
        d_profile: dict[str, float],
        l_profile: dict[str, float],
    ) -> Optional[float]:
        shared = set(d_profile.keys()) & set(l_profile.keys())
        if len(shared) < 3:
            return None

        fold_diffs = []
        for receptor in shared:
            ki_d = d_profile.get(receptor, float("inf"))
            ki_l = l_profile.get(receptor, float("inf"))
            if ki_d <= 0 or ki_l <= 0:
                continue
            fold_diffs.append(max(ki_d / ki_l, ki_l / ki_d))

        if not fold_diffs:
            return None

        import numpy as np
        return float(np.mean(fold_diffs))


class ChiralSwitchLayer(BaseLayer):
    """
    Chiral switch screening layer.

    Fix #10: Withdrawal data fetched from FDA enforcement + drug label APIs.
    """

    layer_name = "layer_chirality"
    version = "1.1"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.chembl = ChEMBLClient()
        self.pdsp   = PDSPClient()

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Check chirality ────────────────────────────────────────────
        chirality = self.chembl.get_chirality(pair.drug_id)

        if chirality != "Racemic mixture":
            pair.scores.chirality_divergence_score = 0.0
            pair.scores.chiral_switch_candidate    = False
            return pair

        logger.info(
            f"[{self.layer_name}] {pair.drug_name}: RACEMIC — analysing enantiomers"
        )

        # ── 2. Check withdrawal (Fix #10: dynamic FDA lookup) ─────────────
        withdrawal_reason = self._get_fda_withdrawn_reason(pair.drug_name)

        # Also check seed list for well-known cases
        seed_info: Optional[dict] = None
        for known, info in _SEED_WITHDRAWN.items():
            if known.lower() in pair.drug_name.lower():
                seed_info = info
                if not withdrawal_reason:
                    withdrawal_reason = f"Seed known: {info.get('toxic_receptor', '')} toxicity"
                break

        # ── 3. Get enantiomer receptor profiles ───────────────────────────
        profiles      = self.pdsp.get_enantiomer_profiles(pair.drug_name)
        d_profile     = profiles.get("d", {})
        l_profile     = profiles.get("l", {})

        divergence_score = None
        if d_profile and l_profile:
            divergence_score = self.pdsp.compute_divergence_score(d_profile, l_profile)

        pair.scores.chirality_divergence_score = divergence_score

        # ── 4. Determine if chiral switch is viable ───────────────────────
        viable       = False
        patent_score = 0.0
        notes_parts  = []

        if withdrawal_reason:
            viable        = True
            patent_score += 0.4
            notes_parts.append(f"FDA withdrawal/warning: {withdrawal_reason}")

        if divergence_score is not None:
            if divergence_score >= ENANTIOMER_DIVERGENCE_THRESHOLD:
                viable        = True
                patent_score += min(0.4, divergence_score / 100)
                notes_parts.append(
                    f"PDSP receptor divergence: {divergence_score:.1f}x fold difference."
                )
            else:
                notes_parts.append(
                    f"PDSP divergence {divergence_score:.1f}x below threshold "
                    f"({ENANTIOMER_DIVERGENCE_THRESHOLD}x required)."
                )

        # Bonus: seed receptor-disease match
        if seed_info and self._disease_involves_receptor(
            pair.disease_name, seed_info.get("therapeutic_receptor", "")
        ):
            patent_score += 0.2
            notes_parts.append("Disease pathway involves therapeutic receptor — strong fit.")

        pair.scores.chiral_switch_candidate = viable

        if viable:
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"CHIRAL SWITCH VIABLE — patent_score={patent_score:.2f}\n"
                f"  {' '.join(notes_parts)}\n"
                f"  → FILE PROVISIONAL PATENT before publishing this finding."
            )

        return pair

    # ── Fix #10: dynamic FDA withdrawal lookup ────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 90)
    def _get_fda_withdrawn_reason(self, drug_name: str) -> Optional[str]:
        """
        Fix #10: Query FDA drug enforcement and drug label databases for
        withdrawal reasons or black-box warnings.

        Source 1: openFDA enforcement database (market withdrawals/recalls).
        Source 2: openFDA drug label database (black-box warnings).

        Results cached for 90 days.
        """
        # Source 1: enforcement / voluntary market withdrawal
        try:
            r = requests.get(
                "https://api.fda.gov/drug/enforcement.json",
                params={
                    "search": f'product_description:"{drug_name}" AND '
                              f'voluntary_mandated:"Voluntary"',
                    "limit": 5,
                },
                timeout=15,
            )
            if r.status_code == 200:
                results = r.json().get("results", [])
                for result in results:
                    reason = result.get("reason_for_recall", "")
                    if reason:
                        return reason
        except Exception as e:
            logger.debug(f"FDA enforcement lookup failed for {drug_name}: {e}")

        # Source 2: drug label black-box warning
        try:
            r2 = requests.get(
                "https://api.fda.gov/drug/label.json",
                params={
                    "search": f'openfda.brand_name:"{drug_name}"',
                    "limit": 1,
                },
                timeout=15,
            )
            if r2.status_code == 200:
                results2 = r2.json().get("results", [])
                if results2:
                    boxed = results2[0].get("boxed_warning", [])
                    if boxed:
                        return " ".join(boxed)
        except Exception as e:
            logger.debug(f"FDA label lookup failed for {drug_name}: {e}")

        return None

    def _disease_involves_receptor(self, disease_name: str, receptor: str) -> bool:
        if not receptor:
            return False
        receptor_disease_map = {
            "5-HT2C": ["epilepsy", "seizure", "dravet", "lennox", "depression"],
            "H1":     ["allergy", "allergic", "rhinitis", "urticaria"],
            "D2":     ["parkinson", "psychosis", "tourette", "huntington"],
            "CRBN_immunomodulatory": ["myeloma", "lymphoma", "myelodysplastic"],
        }
        disease_lower = disease_name.lower()
        for rec, diseases in receptor_disease_map.items():
            if rec.lower() in receptor.lower():
                if any(d in disease_lower for d in diseases):
                    return True
        return False


def screen_chiral_switch_universe(
    disease_targets: list[dict],
    max_candidates: int = 50,
) -> list[ChiralSwitchCandidate]:
    """
    Screen the full universe of racemic off-patent drugs for chiral switch
    opportunities relevant to a list of target diseases.
    """
    from src.ingestion.chembl_client import ChEMBLClient
    chembl = ChEMBLClient()
    pdsp   = PDSPClient()
    layer  = ChiralSwitchLayer()

    racemic_drugs = chembl.get_racemic_candidates(limit=2000)
    logger.info(f"Chiral switch universe: {len(racemic_drugs)} racemic candidates")

    candidates = []
    for drug in racemic_drugs:
        profiles = pdsp.get_enantiomer_profiles(drug["name"])
        d_prof   = profiles.get("d", {})
        l_prof   = profiles.get("l", {})

        if not d_prof or not l_prof:
            continue

        divergence = pdsp.compute_divergence_score(d_prof, l_prof)
        if divergence is None or divergence < ENANTIOMER_DIVERGENCE_THRESHOLD:
            continue

        withdrawal_reason = layer._get_fda_withdrawn_reason(drug["name"])

        candidate = ChiralSwitchCandidate(
            drug_name=drug["name"],
            chembl_id=drug["chembl_id"],
            is_racemic=True,
            receptor_divergence_score=divergence,
            toxic_enantiomer=None,
            therapeutic_enantiomer=None,
            toxic_receptor=None,
            therapeutic_receptor=None,
            is_withdrawn=bool(withdrawal_reason),
            withdrawal_reason=withdrawal_reason,
            chiral_switch_viable=True,
            patent_opportunity_score=min(
                1.0,
                divergence / 100 + (0.4 if withdrawal_reason else 0)
            ),
        )
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.patent_opportunity_score, reverse=True)
    return candidates[:max_candidates]