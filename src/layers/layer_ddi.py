"""
src/layers/layer_ddi.py

Drug-Drug Interaction Modeling Layer.

Fix #7: Removed hardcoded DISEASE_COMEDS dict. Co-medications are now fetched
        dynamically from OpenTargets knownDrugs API (drugs in trials/approved for
        the disease) and cached for 30 days. The hardcoded dict was wrong for any
        disease not in the small list, silently returning no DDI analysis.

Fix #9: Removed hardcoded NARROW_TI_DRUGS set. Narrow-therapeutic-index status is
        now determined dynamically by checking the FDA drug label for the phrase
        "narrow therapeutic" and cross-checking DailyMed. Both sources are cached.
        The hardcoded set missed many NTI drugs and cannot be kept current.

Fix #12: DrugBankDDIClient.get_cyp_profile() now falls back to PharmGKBClient
         when drugbank_ddis.json is absent (which it always is without a DrugBank
         academic license). Previously the DDI layer silently returned zero
         interactions for every pair because get_cyp_profile() returned empty
         lists. PharmGKB covers all CYP substrate/inhibitor/inducer data needed
         for the interaction checks in _check_cyp_interactions().

Fix #13: Updated OpenTargets co-medication query from deprecated knownDrugs field
         to drugAndClinicalCandidates (OpenTargets schema change). Also fixed
         response extraction path accordingly.
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

OPENFDA_LABEL_API = "https://api.fda.gov/drug/label.json"
DAILYMED_API      = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
OT_GRAPHQL        = "https://api.platform.opentargets.org/api/v4/graphql"

CYP_ENZYMES  = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "CYP3A5"]
TRANSPORTERS = ["P-gp", "BCRP", "OATP1B1", "OATP1B3", "OCT2", "MATE1"]

DDI_SEVERITY_SCORES = {
    "contraindicated": 4,
    "major": 3,
    "moderate": 2,
    "minor": 1,
    "none": 0,
}

_NTI_TERMS = [
    "narrow therapeutic index",
    "narrow therapeutic range",
    "narrow therapeutic window",
    "narrow margin of safety",
]


@dataclass
class DDIInteraction:
    comed_name: str
    mechanism: str
    severity: str
    direction: str
    effect: str
    evidence_source: str
    is_narrow_ti: bool = False


@dataclass
class DDIProfile:
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
    CYP/transporter pharmacology and known DDI lookup.

    Fix #12: When drugbank_ddis.json is absent, get_cyp_profile() now
    delegates to PharmGKBClient instead of silently returning empty lists.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._local_db = None
        self._pharmgkb = None

    def _load_local_db(self) -> dict:
        if self._local_db is not None:
            return self._local_db
        import os, json
        path = "data/processed/drugbank_ddis.json"
        if os.path.exists(path):
            with open(path) as f:
                self._local_db = json.load(f)
            logger.info(f"Loaded DrugBank DDI DB: {len(self._local_db)} drugs")
        else:
            logger.info(
                "drugbank_ddis.json not found — DDI CYP profiles will use "
                "PharmGKB as fallback (Fix #12)"
            )
            self._local_db = {}
        return self._local_db

    def _get_pharmgkb(self):
        if self._pharmgkb is None:
            from src.ingestion.pharmgkb_client import PharmGKBClient
            self._pharmgkb = PharmGKBClient()
        return self._pharmgkb

    def get_cyp_profile(self, drug_name: str) -> dict[str, list[str]]:
        db = self._load_local_db()
        drug_data = db.get(drug_name.lower(), {})
        if drug_data:
            return {
                "substrates": drug_data.get("cyp_substrates", []),
                "inhibitors": drug_data.get("cyp_inhibitors", []),
                "inducers":   drug_data.get("cyp_inducers", []),
            }
        try:
            profile = self._get_pharmgkb().get_full_cyp_profile(drug_name)
            if any(profile.values()):
                logger.debug(f"[DDI] CYP profile for '{drug_name}' sourced from PharmGKB")
            return profile
        except Exception as e:
            logger.debug(f"[DDI] PharmGKB fallback failed for {drug_name}: {e}")
            return {"substrates": [], "inhibitors": [], "inducers": []}

    def get_transporter_profile(self, drug_name: str) -> dict[str, list[str]]:
        db = self._load_local_db()
        drug_data = db.get(drug_name.lower(), {})
        return {
            "substrates": drug_data.get("transporter_substrates", []),
            "inhibitors": drug_data.get("transporter_inhibitors", []),
        }

    def get_known_interactions(self, drug_name: str) -> list[dict]:
        db = self._load_local_db()
        return db.get(drug_name.lower(), {}).get("interactions", [])


class DDILayer(BaseLayer):
    """
    Drug-drug interaction modeling layer.

    Fix #7:  Co-medications fetched dynamically from OpenTargets.
    Fix #9:  NTI status checked dynamically via FDA label + DailyMed.
    Fix #12: CYP profiles sourced from PharmGKB when DrugBank file absent.
    Fix #13: OpenTargets query updated to drugAndClinicalCandidates.
    """

    layer_name = "layer_ddi"
    version = "1.3"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        drugbank_api_key = (config or {}).get("drugbank_api_key")
        self.drugbank = DrugBankDDIClient(api_key=drugbank_api_key)

    @cached_api_call(ttl_seconds=86400 * 30)
    def _get_standard_comeds(self, disease_name: str, disease_id: str) -> list[str]:
        """
        Fetch standard-of-care co-medications from OpenTargets.
        Fix #13: Updated from deprecated knownDrugs to drugAndClinicalCandidates.
        """
        comeds: set[str] = set()
        efo_id = self._get_efo_id_for_disease(disease_id)
        if not efo_id:
            logger.warning(f"[{self.layer_name}] Cannot fetch co-meds for {disease_id}: no EFO ID")
            return []

        query = """
        query DiseaseComeds($efoId: String!) {
          disease(efoId: $efoId) {
            drugAndClinicalCandidates {
              rows {
                drug { name }
              }
            }
          }
        }
        """
        try:
            r = requests.post(
                OT_GRAPHQL,
                headers={"Content-Type": "application/json"},
                json={"query": query, "variables": {"efoId": efo_id}},
                timeout=20,
            )
            r.raise_for_status()
            rows = (
                r.json()
                .get("data", {})
                .get("disease", {})
                .get("drugAndClinicalCandidates", {})
                .get("rows", [])
            )
            for row in rows:
                name = row.get("drug", {}).get("name", "")
                if name:
                    comeds.add(name.lower())
            logger.info(
                f"[{self.layer_name}] {disease_id}: {len(comeds)} co-medications "
                f"fetched from OpenTargets"
            )
        except Exception as e:
            logger.debug(f"OpenTargets co-med lookup failed for {disease_id}: {e}")

        return list(comeds)

    def _get_efo_id_for_disease(self, disease_id: str) -> Optional[str]:
        try:
            from src.ingestion.opentargets_client import OpenTargetsClient
            return OpenTargetsClient().get_efo_id(disease_id)
        except Exception as e:
            logger.debug(f"EFO ID lookup failed for {disease_id}: {e}")
            return None

    @cached_api_call(ttl_seconds=86400 * 90)
    def _is_narrow_ti_drug(self, drug_name: str) -> bool:
        """Fix #9: Check NTI status via FDA label + DailyMed."""
        try:
            r = requests.get(
                OPENFDA_LABEL_API,
                params={
                    "search": f'openfda.brand_name:"{drug_name}" AND warnings:"narrow therapeutic"',
                    "limit": 1,
                },
                timeout=15,
            )
            if r.status_code == 200:
                if r.json().get("meta", {}).get("results", {}).get("total", 0) > 0:
                    return True
        except Exception:
            pass

        try:
            r2 = requests.get(
                f"{DAILYMED_API}/spls.json",
                params={"drug_name": drug_name, "pagesize": 1},
                timeout=15,
            )
            if r2.status_code == 200 and r2.json().get("data"):
                spl_id = r2.json()["data"][0].get("setid", "")
                if spl_id:
                    r3 = requests.get(f"{DAILYMED_API}/spls/{spl_id}.json", timeout=15)
                    if any(term in str(r3.json()).lower() for term in _NTI_TERMS):
                        return True
        except Exception:
            pass

        return False

    def score(self, pair: CandidatePair) -> CandidatePair:
        cyp_profile         = self.drugbank.get_cyp_profile(pair.drug_name)
        transporter_profile = self.drugbank.get_transporter_profile(pair.drug_name)

        drug_profile = DDIProfile(
            drug_name=pair.drug_name,
            cyp_substrates=cyp_profile.get("substrates", []),
            cyp_inhibitors=cyp_profile.get("inhibitors", []),
            cyp_inducers=cyp_profile.get("inducers", []),
            transporter_substrates=transporter_profile.get("substrates", []),
            transporter_inhibitors=transporter_profile.get("inhibitors", []),
        )

        logger.info(
            f"[{self.layer_name}] {pair.drug_name}: CYP substrates="
            f"{drug_profile.cyp_substrates}, inhibitors={drug_profile.cyp_inhibitors}, "
            f"inducers={drug_profile.cyp_inducers} (source: PharmGKB fallback)"
        )

        comeds = self._get_standard_comeds(pair.disease_name, pair.disease_id)
        if not comeds:
            logger.warning(
                f"[{self.layer_name}] No co-medication data for "
                f"{pair.disease_id} ({pair.disease_name}). Skipping DDI analysis."
            )
            return pair

        interactions = []
        known_ddis       = self.drugbank.get_known_interactions(pair.drug_name)
        known_ddi_lookup = {d["drug_name"].lower(): d for d in known_ddis}

        for comed_name in comeds:
            is_nti = self._is_narrow_ti_drug(comed_name)
            if comed_name.lower() in known_ddi_lookup:
                ddi_data = known_ddi_lookup[comed_name.lower()]
                interactions.append(DDIInteraction(
                    comed_name=comed_name,
                    mechanism=ddi_data.get("mechanism", "unknown"),
                    severity=ddi_data.get("severity", "moderate").lower(),
                    direction="bidirectional",
                    effect=ddi_data.get("description", "Known interaction"),
                    evidence_source="DrugBank",
                    is_narrow_ti=is_nti,
                ))
                continue
            comed_profile = self.drugbank.get_cyp_profile(comed_name)
            interactions.extend(
                self._check_cyp_interactions(drug_profile, comed_profile, comed_name, is_nti)
            )

        drug_profile.interactions = interactions

        if not interactions:
            ddi_score = 0.0
        else:
            severity_scores = [DDI_SEVERITY_SCORES.get(i.severity, 0) for i in interactions]
            weighted_scores = [s * (2.0 if i.is_narrow_ti else 1.0)
                               for s, i in zip(severity_scores, interactions)]
            ddi_score = min(1.0, sum(weighted_scores) / (len(weighted_scores) * 4))

        pair.scores.ddi_risk_score = ddi_score

        serious = [i for i in interactions
                   if i.severity in ("contraindicated", "major") and i.is_narrow_ti]
        if serious:
            pair.flags.ddi_risk_narrow_index = True
            logger.warning(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"DDI WARNING — {len(serious)} serious NTI interactions:\n"
                + "\n".join(f"  {i.comed_name}: {i.severity} — {i.effect}" for i in serious)
            )
        else:
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"DDI score={ddi_score:.3f}, {len(interactions)} interactions flagged"
            )

        pair.data_sources["ddi_interactions"] = [
            {"comed": i.comed_name, "severity": i.severity, "mechanism": i.mechanism,
             "effect": i.effect, "narrow_ti": i.is_narrow_ti, "source": i.evidence_source}
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
        candidate_name   = drug_profile.drug_name
        interactions     = []
        comed_substrates = comed_profile.get("substrates", [])
        comed_inhibitors = comed_profile.get("inhibitors", [])
        comed_inducers   = comed_profile.get("inducers", [])

        for cyp in drug_profile.cyp_inhibitors:
            if cyp in comed_substrates:
                interactions.append(DDIInteraction(
                    comed_name=comed_name, mechanism=f"{cyp}_inhibition",
                    severity="major" if is_nti else "moderate",
                    direction="candidate_inhibits",
                    effect=f"{candidate_name} inhibits {cyp}, raising {comed_name} plasma levels",
                    evidence_source="PharmGKB_predicted", is_narrow_ti=is_nti,
                ))

        for cyp in drug_profile.cyp_substrates:
            if cyp in comed_inhibitors:
                interactions.append(DDIInteraction(
                    comed_name=comed_name, mechanism=f"{cyp}_substrate_inhibited",
                    severity="moderate", direction="candidate_is_substrate",
                    effect=f"{comed_name} inhibits {cyp}, raising {candidate_name} plasma levels",
                    evidence_source="PharmGKB_predicted", is_narrow_ti=False,
                ))

        for cyp in drug_profile.cyp_substrates:
            if cyp in comed_inducers:
                interactions.append(DDIInteraction(
                    comed_name=comed_name, mechanism=f"{cyp}_induction",
                    severity="moderate", direction="comed_induces",
                    effect=f"{comed_name} induces {cyp}, reducing {candidate_name} plasma levels → efficacy risk",
                    evidence_source="PharmGKB_predicted", is_narrow_ti=False,
                ))

        return interactions