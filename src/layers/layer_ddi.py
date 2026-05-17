"""
src/layers/layer_ddi.py

Drug-Drug Interaction Modeling Layer (Tier 1 Critical).

Problem: Rare disease patients are almost universally on multiple co-medications.
A serious DDI in your trial is both a patient safety event and a regulatory catastrophe.
This layer flags DDI risk BEFORE you commit to a trial.

Three DDI mechanisms:
  1. Pharmacokinetic DDI: one drug changes plasma concentration of another
     (via CYP enzyme inhibition/induction or transporter blockade)
  2. Pharmacodynamic DDI: two drugs act on the same physiological system
     (additive or antagonistic at the target)
  3. Transporter DDI: one drug blocks hepatic/renal transporters
     (OATP1B1, P-gp, BCRP, OCT2, MATE1)

Output: DDI risk matrix for each candidate vs standard-of-care co-medications.
Severity ratings use FDA DDI classification: contraindicated / major / moderate / minor.

Data sources:
  - DrugBank: CYP substrate/inhibitor/inducer profiles + known DDI list
  - FDA FAERS: real-world DDI adverse event signals (most trusted by regulators)
  - DailyMed (NLM): standard-of-care co-medications from drug labels
  - CREDIBLE: curated clinical DDI evidence database

Key papers:
  - Tatonetti et al. 2012, Sci Transl Med (FAERS DDI detection — gold standard)
  - Ryu et al. 2018, PLOS Comp Biol (DeepDDI neural network)
  - Boyce et al. 2017, Nucleic Acids Research (DrugBank DDI data)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

DRUGBANK_API = "https://api.drugbankplus.com/v1"
DAILYMED_API = "https://lhncbc.nlm.nih.gov/RxNav/APIs"
OPENFDA_API = "https://api.fda.gov/drug"

# CYP enzymes relevant to DDI risk
CYP_ENZYMES = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "CYP3A5"]

# Drug transporters with clinical DDI significance
TRANSPORTERS = ["P-gp", "BCRP", "OATP1B1", "OATP1B3", "OCT2", "MATE1"]

# DDI severity → numeric score (for composite scoring)
DDI_SEVERITY_SCORES = {
    "contraindicated": 4,
    "major": 3,
    "moderate": 2,
    "minor": 1,
    "none": 0,
}

# Standard-of-care co-medications by disease category.
# Bio team: add disease-specific co-medication lists here.
# These are the drugs your trial patients will most likely be on.
# Format: disease_id → list of drug names (common co-medications)
DISEASE_COMEDS: dict[str, list[str]] = {
    "ORPHA:33069": [    # Dravet syndrome
        "valproate", "clobazam", "topiramate", "stiripentol", "levetiracetam",
    ],
    "ORPHA:77": [       # Gaucher disease type 1
        "imiglucerase", "velaglucerase", "taliglucerase",   # ERT
        "eliglustat", "miglustat",   # SRT
        "alendronate",   # bone complications
    ],
    "ORPHA:422": [      # Pulmonary arterial hypertension
        "ambrisentan", "bosentan", "macitentan",   # ERA
        "sildenafil", "tadalafil",   # PDE5i
        "iloprost", "treprostinil",  # prostacyclin
        "warfarin",   # anticoagulation
    ],
    "ORPHA:566": [      # Pompe disease
        "alglucosidase",   # ERT
        "miglustat",       # chaperone
        "atorvastatin",    # cardiovascular
    ],
}

# Drugs with narrow therapeutic index — DDI interactions here are high-risk
NARROW_TI_DRUGS = {
    "warfarin", "digoxin", "phenytoin", "carbamazepine", "valproate",
    "tacrolimus", "cyclosporine", "methotrexate", "lithium",
    "theophylline", "aminoglycosides",
}


@dataclass
class DDIInteraction:
    """A single drug-drug interaction finding."""
    comed_name: str
    mechanism: str           # 'CYP2D6_inhibition', 'OATP1B1_transport', etc.
    severity: str            # 'contraindicated' | 'major' | 'moderate' | 'minor'
    direction: str           # 'candidate_inhibits' | 'candidate_is_substrate' | 'bidirectional'
    effect: str              # Human-readable description of the interaction
    evidence_source: str     # 'DrugBank' | 'FAERS' | 'CREDIBLE'
    is_narrow_ti: bool = False


@dataclass
class DDIProfile:
    """Complete DDI risk profile for a candidate drug."""
    drug_name: str
    cyp_substrates: list[str] = field(default_factory=list)
    cyp_inhibitors: list[str] = field(default_factory=list)
    cyp_inducers: list[str] = field(default_factory=list)
    transporter_substrates: list[str] = field(default_factory=list)
    transporter_inhibitors: list[str] = field(default_factory=list)
    interactions: list[DDIInteraction] = field(default_factory=list)
    max_severity: str = "none"
    narrow_ti_risk: bool = False


class DrugBankDDIClient:
    """
    Query DrugBank for CYP/transporter pharmacology and known DDIs.

    DrugBank Plus API requires a key (academic/commercial tiers available).
    Free alternative: download DrugBank XML dump and parse locally.
    Register at: https://www.drugbank.com/releases/latest

    Local XML approach (recommended for startup):
        wget https://go.drugbank.com/releases/latest/downloads/all-full-database
        python data/scripts/parse_drugbank_xml.py  → produces data/processed/drugbank_ddis.json
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._local_db = None

    def _load_local_db(self) -> dict:
        """Load pre-parsed DrugBank DDI data from local JSON file."""
        if self._local_db is not None:
            return self._local_db
        import os
        import json
        path = "data/processed/drugbank_ddis.json"
        if os.path.exists(path):
            with open(path) as f:
                self._local_db = json.load(f)
            logger.info(f"Loaded DrugBank DDI DB: {len(self._local_db)} drugs")
        else:
            logger.warning(
                "DrugBank DDI local DB not found at data/processed/drugbank_ddis.json. "
                "Parse from DrugBank XML dump using data/scripts/parse_drugbank_xml.py"
            )
            self._local_db = {}
        return self._local_db

    def get_cyp_profile(self, drug_name: str) -> dict[str, list[str]]:
        """
        Return CYP enzyme profile for a drug.

        Returns:
            {
              'substrates': ['CYP2C19', 'CYP3A4'],
              'inhibitors': ['CYP2D6'],
              'inducers':   []
            }
        """
        db = self._load_local_db()
        drug_data = db.get(drug_name.lower(), {})
        return {
            "substrates": drug_data.get("cyp_substrates", []),
            "inhibitors": drug_data.get("cyp_inhibitors", []),
            "inducers": drug_data.get("cyp_inducers", []),
        }

    def get_transporter_profile(self, drug_name: str) -> dict[str, list[str]]:
        """Return transporter substrate/inhibitor profile."""
        db = self._load_local_db()
        drug_data = db.get(drug_name.lower(), {})
        return {
            "substrates": drug_data.get("transporter_substrates", []),
            "inhibitors": drug_data.get("transporter_inhibitors", []),
        }

    def get_known_interactions(self, drug_name: str) -> list[dict]:
        """Return known DDIs from DrugBank."""
        db = self._load_local_db()
        drug_data = db.get(drug_name.lower(), {})
        return drug_data.get("interactions", [])


class DDILayer(BaseLayer):
    """
    Drug-drug interaction modeling layer.

    Scores:
        pair.scores.ddi_risk_score   (0–1; higher = more DDI risk vs typical co-medications)

    Flags:
        pair.flags.ddi_risk_narrow_index   (True if major/contraindicated DDI with NTI drug)

    Output also stored in pair.data_sources['ddi_interactions'] for report generation.

    Validation:
        Carbamazepine × Dravet syndrome should flag moderate-major DDI with CYP3A4 inducers
        (carbamazepine induces CYP3A4 → reduces plasma levels of CYP3A4 substrates)
    """

    layer_name = "layer_ddi"
    version = "1.0"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        drugbank_api_key = (config or {}).get("drugbank_api_key")
        self.drugbank = DrugBankDDIClient(api_key=drugbank_api_key)

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Get candidate drug CYP/transporter profile ─────────────────
        cyp_profile = self.drugbank.get_cyp_profile(pair.drug_name)
        transporter_profile = self.drugbank.get_transporter_profile(pair.drug_name)

        drug_profile = DDIProfile(
            drug_name=pair.drug_name,
            cyp_substrates=cyp_profile.get("substrates", []),
            cyp_inhibitors=cyp_profile.get("inhibitors", []),
            cyp_inducers=cyp_profile.get("inducers", []),
            transporter_substrates=transporter_profile.get("substrates", []),
            transporter_inhibitors=transporter_profile.get("inhibitors", []),
        )

        # ── 2. Get disease standard-of-care co-medications ────────────────
        comeds = DISEASE_COMEDS.get(pair.disease_id, [])
        if not comeds:
            logger.warning(
                f"[{self.layer_name}] No co-medication list for {pair.disease_id}. "
                f"Add to DISEASE_COMEDS dict. Skipping DDI analysis."
            )
            return pair

        # ── 3. Check interactions ─────────────────────────────────────────
        interactions = []
        known_ddis = self.drugbank.get_known_interactions(pair.drug_name)
        known_ddi_lookup = {d["drug_name"].lower(): d for d in known_ddis}

        for comed_name in comeds:
            is_nti = comed_name.lower() in NARROW_TI_DRUGS

            # Check direct DrugBank interaction
            if comed_name.lower() in known_ddi_lookup:
                ddi_data = known_ddi_lookup[comed_name.lower()]
                severity = ddi_data.get("severity", "moderate").lower()
                interactions.append(DDIInteraction(
                    comed_name=comed_name,
                    mechanism=ddi_data.get("mechanism", "unknown"),
                    severity=severity,
                    direction="bidirectional",
                    effect=ddi_data.get("description", "Known interaction"),
                    evidence_source="DrugBank",
                    is_narrow_ti=is_nti,
                ))
                continue

            # CYP-mediated interactions (predicted)
            comed_profile = self.drugbank.get_cyp_profile(comed_name)
            cyp_interactions = self._check_cyp_interactions(
                drug_profile, comed_profile, comed_name, is_nti
            )
            interactions.extend(cyp_interactions)

        drug_profile.interactions = interactions

        # ── 4. Compute DDI risk score ─────────────────────────────────────
        if not interactions:
            ddi_score = 0.0
        else:
            severity_scores = [
                DDI_SEVERITY_SCORES.get(i.severity, 0) for i in interactions
            ]
            # Weight NTI interactions more heavily
            weighted_scores = [
                s * (2.0 if i.is_narrow_ti else 1.0)
                for s, i in zip(severity_scores, interactions)
            ]
            ddi_score = min(1.0, sum(weighted_scores) / (len(weighted_scores) * 4))

        pair.scores.ddi_risk_score = ddi_score

        # ── 5. Flag serious interactions ──────────────────────────────────
        serious_interactions = [
            i for i in interactions
            if i.severity in ("contraindicated", "major") and i.is_narrow_ti
        ]
        if serious_interactions:
            pair.flags.ddi_risk_narrow_index = True
            logger.warning(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"DDI WARNING — {len(serious_interactions)} serious interactions with NTI drugs:\n"
                + "\n".join(
                    f"  {i.comed_name}: {i.severity} — {i.effect}"
                    for i in serious_interactions
                )
            )
        else:
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"DDI score={ddi_score:.3f}, {len(interactions)} interactions flagged"
            )

        # Store interaction details for report generation
        pair.data_sources["ddi_interactions"] = [
            {
                "comed": i.comed_name,
                "severity": i.severity,
                "mechanism": i.mechanism,
                "effect": i.effect,
                "narrow_ti": i.is_narrow_ti,
                "source": i.evidence_source,
            }
            for i in interactions
        ]

        return pair

    def _check_cyp_interactions(
        self,
        drug_profile: DDIProfile,
        comed_profile: dict[str, list[str]],
        comed_name: str,
        is_nti: bool,
    ) -> list[DDIInteraction]:
        """
        Check for CYP-mediated pharmacokinetic interactions.

        Scenarios:
          A) Candidate INHIBITS enzyme → co-med substrate accumulates → toxicity risk
          B) Candidate IS SUBSTRATE → co-med inhibitor raises candidate levels
          C) Co-med INDUCES enzyme → candidate substrate plasma levels drop → efficacy loss
        """
        # BUG FIX: drug_name is now taken directly from drug_profile.drug_name
        # (previously routed through a needless module-level placeholder function)
        candidate_name = drug_profile.drug_name

        interactions = []
        comed_substrates = comed_profile.get("substrates", [])
        comed_inhibitors = comed_profile.get("inhibitors", [])
        comed_inducers = comed_profile.get("inducers", [])

        # A: candidate inhibits CYP that metabolizes co-med
        for cyp in drug_profile.cyp_inhibitors:
            if cyp in comed_substrates:
                severity = "major" if is_nti else "moderate"
                interactions.append(DDIInteraction(
                    comed_name=comed_name,
                    mechanism=f"{cyp}_inhibition",
                    severity=severity,
                    direction="candidate_inhibits",
                    effect=(
                        f"{candidate_name} inhibits {cyp}, "
                        f"raising {comed_name} plasma levels"
                    ),
                    evidence_source="DrugBank_predicted",
                    is_narrow_ti=is_nti,
                ))

        # B: candidate is CYP substrate, co-med is inhibitor of same enzyme
        for cyp in drug_profile.cyp_substrates:
            if cyp in comed_inhibitors:
                interactions.append(DDIInteraction(
                    comed_name=comed_name,
                    mechanism=f"{cyp}_substrate_inhibited",
                    severity="moderate",
                    direction="candidate_is_substrate",
                    effect=(
                        f"{comed_name} inhibits {cyp}, "
                        f"raising {candidate_name} plasma levels"
                    ),
                    evidence_source="DrugBank_predicted",
                    is_narrow_ti=False,
                ))

        # C: co-med induces CYP that metabolizes candidate → efficacy loss
        for cyp in drug_profile.cyp_substrates:
            if cyp in comed_inducers:
                interactions.append(DDIInteraction(
                    comed_name=comed_name,
                    mechanism=f"{cyp}_induction",
                    severity="moderate",
                    direction="comed_induces",
                    effect=(
                        f"{comed_name} induces {cyp}, "
                        f"reducing {candidate_name} plasma levels → efficacy risk"
                    ),
                    evidence_source="DrugBank_predicted",
                    is_narrow_ti=False,
                ))

        return interactions