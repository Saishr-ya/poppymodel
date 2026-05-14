"""
src/layers/layer_chirality.py

Chiral Switch Module — the single highest-IP-value layer in the engine.

Core strategy (from the spec):
  Racemic drugs where one enantiomer carries efficacy and the other carries
  toxicity can be "chiral switched" to produce a novel, patentable single-
  enantiomer drug for a new indication. This gives you TWO layers of novelty:
    1. New molecular form (the pure enantiomer)
    2. New indication (the repurposing claim)

This combination is dramatically harder to challenge in patent proceedings
than a pure method-of-use patent on a racemate.

Reference case: Fenfluramine
  - Racemic form: used as appetite suppressant, withdrawn 1997 (cardiac valve damage)
  - Cardiac toxicity = 5-HT2B receptor agonism (d-enantiomer dominant)
  - Antiseizure activity = different receptor, enantiomers separable
  - l-fenfluramine → repurposed as Fintepla (FDA 2020) for Dravet syndrome
  - Patent: method-of-use (Dravet) + composition (pure l-enantiomer) = iron-clad

Screening pipeline:
  1. Filter ChEMBL for racemic oral small molecules, off-patent before 2015
  2. Check PDSP Ki database for divergent enantiomer receptor binding
  3. Cross-reference FDA withdrawn drug list (withdrawn = toxicity documented = opportunity)
  4. Map therapeutic receptor (from divergent binding) to Orphanet rare disease targets

Data sources:
  - ChEMBL: chirality annotation, approval status
  - PDSP Ki Database (pdsp.unc.edu): enantiomer receptor binding affinities
  - FDA withdrawn drugs: fda.gov/drugs/drug-safety-and-availability
  - DrugBank: metabolite profiles, mechanism of action

Key papers:
  - "The Quest for Secondary Pharmaceuticals: Drug Repurposing/Chiral-Switches
    Combination Strategy" — ACS Pharm. & Transl. Sci. 2022. PMC9926527.
  - "Putting chirality to work: the strategy of chiral switches" — Nat Rev Drug Discov.
  - "Chirality of New Drug Approvals (2013-2022)" — J. Med. Chem. 2023. PMC10895675.
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

PDSP_BASE = "https://pdsp.unc.edu/databases/pdsp.php"

# Minimum fold difference in binding affinity between enantiomers
# to consider a chiral switch viable.
# Ki(d) / Ki(l) > 10 = 10-fold selectivity = meaningful separation
ENANTIOMER_DIVERGENCE_THRESHOLD = 10.0

# Receptors whose activation is known to cause serious toxicity.
# If one enantiomer preferentially binds these, the other is the therapeutic candidate.
TOXIC_RECEPTORS = {
    "5-HT2B",     # cardiac valvulopathy (fenfluramine, pergolide)
    "hERG",       # QT prolongation, arrhythmia
    "D2",         # when high occupancy = tardive dyskinesia risk
    "sigma1",     # some CNS toxicity signals
    "5-HT3",     # emesis (toxicity for some indications)
}

# FDA-withdrawn drugs known to have enantiomer-separable toxicity/efficacy.
# Source: FDA Orange Book + literature.
# Format: {drug_name: {withdrawn_reason, toxic_receptor, therapeutic_receptor}}
WITHDRAWN_DRUG_OPPORTUNITIES: dict[str, dict] = {
    "fenfluramine": {
        "withdrawn_reason": "Cardiac valvulopathy (5-HT2B)",
        "toxic_receptor": "5-HT2B",
        "therapeutic_receptor": "5-HT2C",
        "note": "l-fenfluramine → Fintepla (Dravet, LGS). Already approved.",
    },
    "dexfenfluramine": {
        "withdrawn_reason": "Cardiac valvulopathy (5-HT2B, d-enantiomer)",
        "toxic_receptor": "5-HT2B",
        "therapeutic_receptor": "5-HT2C",
        "note": "Parent of fenfluramine chiral switch story.",
    },
    "thalidomide": {
        "withdrawn_reason": "Teratogenicity (S-enantiomer)",
        "toxic_receptor": "CRBN_teratogenic",
        "therapeutic_receptor": "CRBN_immunomodulatory",
        "note": "R-thalidomide has therapeutic potential. Pomalidomide is a derivative.",
    },
    "terfenadine": {
        "withdrawn_reason": "hERG cardiotoxicity",
        "toxic_receptor": "hERG",
        "therapeutic_receptor": "H1",
        "note": "→ Fexofenadine (active metabolite, not enantiomer). Related strategy.",
    },
}


@dataclass
class ChiralSwitchCandidate:
    """Result of chiral switch analysis for a single drug."""
    drug_name: str
    chembl_id: str
    is_racemic: bool
    receptor_divergence_score: Optional[float]  # fold difference between enantiomers
    toxic_enantiomer: Optional[str]             # 'd' | 'l' | None
    therapeutic_enantiomer: Optional[str]       # 'd' | 'l' | None
    toxic_receptor: Optional[str]
    therapeutic_receptor: Optional[str]
    is_withdrawn: bool
    withdrawal_reason: Optional[str]
    chiral_switch_viable: bool
    patent_opportunity_score: float             # 0–1; higher = stronger IP position
    notes: str = ""


class PDSPClient:
    """
    Query PDSP Ki Database for enantiomer receptor binding profiles.
    PDSP (Psychoactive Drug Screening Program) at UNC.

    Note: PDSP does not have a formal REST API. This scrapes their web interface
    or uses their downloadable data files.

    Setup: Download the PDSP Ki database from https://pdsp.unc.edu/databases/kidb.php
    Place at: data/raw/pdsp/ki_database.csv
    """

    def __init__(self):
        self._db = None

    def _load_db(self):
        """Load PDSP Ki database from disk (CSV download)."""
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

    def get_enantiomer_profiles(
        self, drug_name: str
    ) -> dict[str, dict[str, float]]:
        """
        Return receptor binding profiles for both enantiomers of a drug.

        Returns:
            {
              'd': {'5-HT2B': 0.5, 'hERG': 1.2, ...},   # Ki values in nM
              'l': {'5-HT2B': 45.0, 'hERG': 0.8, ...},
            }
            Empty dict if drug not in database.
        """
        db = self._load_db()
        if db is None:
            return {}

        import pandas as pd

        # Try to find entries for this drug and its enantiomers
        # Common naming conventions: "d-drug", "(+)-drug", "R-drug"
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
            name_col = db.columns[0]   # First column = drug name
            receptor_col = db.columns[1]   # Second = receptor
            ki_col = db.columns[2]   # Third = Ki value

            db_lower = db.copy()
            db_lower[name_col] = db_lower[name_col].str.lower().str.strip()

            for enantiomer, names in [("d", d_names), ("l", l_names)]:
                mask = db_lower[name_col].isin(names)
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
        """
        Compute receptor divergence score between two enantiomers.

        Score = mean fold difference across shared receptors.
        Higher score = more divergent binding = better chiral switch candidate.

        Returns None if insufficient shared receptors (<3).
        """
        shared_receptors = set(d_profile.keys()) & set(l_profile.keys())
        if len(shared_receptors) < 3:
            return None

        fold_diffs = []
        for receptor in shared_receptors:
            ki_d = d_profile.get(receptor, float("inf"))
            ki_l = l_profile.get(receptor, float("inf"))

            if ki_d <= 0 or ki_l <= 0:
                continue

            # Fold difference (always ≥ 1)
            fold = max(ki_d / ki_l, ki_l / ki_d)
            fold_diffs.append(fold)

        if not fold_diffs:
            return None

        import numpy as np
        return float(np.mean(fold_diffs))


class ChiralSwitchLayer(BaseLayer):
    """
    Chiral switch screening layer.

    For each candidate drug:
      1. Checks if it is a racemic mixture
      2. If racemic, checks PDSP for divergent enantiomer binding
      3. If divergent, checks if toxic receptor is separable from therapeutic receptor
      4. Scores the chiral switch opportunity

    Scores:
        pair.scores.chirality_divergence_score   (fold difference between enantiomers)
        pair.scores.chiral_switch_candidate      (True if viable opportunity found)

    Flags:
        No hard disqualifiers — this is an opportunity flag, not a risk flag.

    IP note:
        A confirmed chiral switch candidate should immediately trigger a provisional
        patent application covering: "the [l/d]-enantiomer of [drug] for the treatment
        of [disease]". File before publishing or presenting the computational finding.
    """

    layer_name = "layer_chirality"
    version = "1.0"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.chembl = ChEMBLClient()
        self.pdsp = PDSPClient()

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Check chirality ────────────────────────────────────────────
        chirality = self.chembl.get_chirality(pair.drug_id)

        if chirality != "Racemic mixture":
            # Not racemic — set score to 0 (not a chiral switch opportunity)
            pair.scores.chirality_divergence_score = 0.0
            pair.scores.chiral_switch_candidate = False
            logger.debug(
                f"[{self.layer_name}] {pair.drug_name}: chirality={chirality}, "
                f"not a racemic mixture — skipping chiral switch analysis"
            )
            return pair

        logger.info(
            f"[{self.layer_name}] {pair.drug_name}: RACEMIC — analyzing enantiomers"
        )

        # ── 2. Check if withdrawn (strongest opportunity signal) ──────────
        drug_key = pair.drug_name.lower().replace("-", "").replace(" ", "")
        withdrawn_info = None
        for known_drug, info in WITHDRAWN_DRUG_OPPORTUNITIES.items():
            if known_drug.lower().replace("-", "").replace(" ", "") in drug_key:
                withdrawn_info = info
                break

        # ── 3. Get enantiomer receptor profiles from PDSP ─────────────────
        profiles = self.pdsp.get_enantiomer_profiles(pair.drug_name)
        d_profile = profiles.get("d", {})
        l_profile = profiles.get("l", {})

        divergence_score = None
        if d_profile and l_profile:
            divergence_score = self.pdsp.compute_divergence_score(d_profile, l_profile)

        pair.scores.chirality_divergence_score = divergence_score

        # ── 4. Determine if chiral switch is viable ───────────────────────
        viable = False
        patent_score = 0.0
        notes_parts = []

        if withdrawn_info:
            viable = True
            patent_score += 0.4
            notes_parts.append(
                f"Withdrawn drug: {withdrawn_info['withdrawn_reason']}. "
                f"Toxic receptor: {withdrawn_info['toxic_receptor']}. "
                f"Therapeutic receptor: {withdrawn_info['therapeutic_receptor']}."
            )

        if divergence_score is not None:
            if divergence_score >= ENANTIOMER_DIVERGENCE_THRESHOLD:
                viable = True
                patent_score += min(0.4, divergence_score / 100)
                notes_parts.append(
                    f"PDSP receptor divergence: {divergence_score:.1f}x fold difference."
                )
            else:
                notes_parts.append(
                    f"PDSP divergence score {divergence_score:.1f}x below threshold "
                    f"({ENANTIOMER_DIVERGENCE_THRESHOLD}x required)."
                )

        # Bonus: if disease target is the therapeutic receptor
        if withdrawn_info and self._disease_involves_receptor(
            pair.disease_name, withdrawn_info.get("therapeutic_receptor", "")
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
        else:
            logger.debug(
                f"[{self.layer_name}] {pair.drug_name}: racemic but no clear "
                f"chiral switch opportunity found."
            )

        return pair

    def _disease_involves_receptor(self, disease_name: str, receptor: str) -> bool:
        """
        Heuristic: check if disease name/type suggests involvement of the receptor.
        Bio team should validate for every flagged candidate.
        """
        if not receptor:
            return False

        receptor_disease_map = {
            "5-HT2C": ["epilepsy", "seizure", "dravet", "lennox", "depression"],
            "H1": ["allergy", "allergic", "rhinitis", "urticaria"],
            "D2": ["parkinson", "psychosis", "tourette", "huntington"],
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

    Args:
        disease_targets: List of {disease_id, disease_name, target_receptors}
        max_candidates:  Maximum candidates to return per disease.

    Returns:
        List of ChiralSwitchCandidates sorted by patent_opportunity_score desc.

    This is the entry point for the chiral switch module's independent screening
    (separate from per-pair scoring in ChiralSwitchLayer).
    """
    from src.ingestion.chembl_client import ChEMBLClient
    chembl = ChEMBLClient()
    pdsp = PDSPClient()

    racemic_drugs = chembl.get_racemic_candidates(limit=2000)
    logger.info(f"Chiral switch universe: {len(racemic_drugs)} racemic candidates")

    candidates = []
    for drug in racemic_drugs:
        profiles = pdsp.get_enantiomer_profiles(drug["name"])
        d_prof = profiles.get("d", {})
        l_prof = profiles.get("l", {})

        if not d_prof or not l_prof:
            continue

        divergence = pdsp.compute_divergence_score(d_prof, l_prof)
        if divergence is None or divergence < ENANTIOMER_DIVERGENCE_THRESHOLD:
            continue

        withdrawn = drug["name"].lower() in {k.lower() for k in WITHDRAWN_DRUG_OPPORTUNITIES}

        candidate = ChiralSwitchCandidate(
            drug_name=drug["name"],
            chembl_id=drug["chembl_id"],
            is_racemic=True,
            receptor_divergence_score=divergence,
            toxic_enantiomer=None,        # requires manual PDSP interpretation
            therapeutic_enantiomer=None,
            toxic_receptor=None,
            therapeutic_receptor=None,
            is_withdrawn=withdrawn,
            withdrawal_reason=WITHDRAWN_DRUG_OPPORTUNITIES.get(
                drug["name"].lower(), {}
            ).get("withdrawn_reason"),
            chiral_switch_viable=True,
            patent_opportunity_score=min(1.0, divergence / 100 + (0.4 if withdrawn else 0)),
        )
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.patent_opportunity_score, reverse=True)
    return candidates[:max_candidates]