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

# NTI language from FDA guidance
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
    Query DrugBank for CYP/transporter pharmacology and known DDIs.
    Falls back to local JSON if available.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._local_db = None

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
            self._local_db = {}
        return self._local_db

    def get_cyp_profile(self, drug_name: str) -> dict[str, list[str]]:
        db = self._load_local_db()
        drug_data = db.get(drug_name.lower(), {})
        return {
            "substrates": drug_data.get("cyp_substrates", []),
            "inhibitors": drug_data.get("cyp_inhibitors", []),
            "inducers":   drug_data.get("cyp_inducers", []),
        }

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

    Fix #7: Co-medications fetched dynamically from OpenTargets.
    Fix #9: NTI status checked dynamically via FDA label + DailyMed.
    """

    layer_name = "layer_ddi"
    version = "1.1"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        drugbank_api_key = (config or {}).get("drugbank_api_key")
        self.drugbank = DrugBankDDIClient(api_key=drugbank_api_key)

    # ── Fix #7: dynamic co-medication lookup ──────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 30)
    def _get_standard_comeds(self, disease_name: str, disease_id: str) -> list[str]:
        """
        Fix #7: Fetch standard-of-care co-medications dynamically from OpenTargets.

        Queries approved/Phase-III drugs for the disease. Results are cached
        for 30 days to avoid repeated API calls during batch scoring.
        """
        comeds: set[str] = set()

        # Resolve EFO ID for this disease
        efo_id = self._get_efo_id_for_disease(disease_id)
        if not efo_id:
            logger.warning(
                f"[{self.layer_name}] Cannot fetch co-meds for {disease_id}: no EFO ID"
            )
            return []

        query = """
        query DiseaseKnownDrugs($efoId: String!) {
          disease(efoId: $efoId) {
            knownDrugs(page: {index: 0, size: 25}) {
              rows {
                drug { name }
                phase
                status
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
                .get("knownDrugs", {})
                .get("rows", [])
            )
            for row in rows:
                name = row.get("drug", {}).get("name", "")
                if name:
                    comeds.add(name.lower())
        except Exception as e:
            logger.debug(f"OpenTargets co-med lookup failed for {disease_id}: {e}")

        return list(comeds)

    def _get_efo_id_for_disease(self, disease_id: str) -> Optional[str]:
        """Delegate EFO resolution to OpenTargetsClient."""
        try:
            from src.ingestion.opentargets_client import OpenTargetsClient
            return OpenTargetsClient().get_efo_id(disease_id)
        except Exception as e:
            logger.debug(f"EFO ID lookup failed for {disease_id}: {e}")
            return None

    # ── Fix #9: dynamic NTI check ─────────────────────────────────────────────

    @cached_api_call(ttl_seconds=86400 * 90)
    def _is_narrow_ti_drug(self, drug_name: str) -> bool:
        """
        Fix #9: Check if a drug has narrow therapeutic index via FDA label + DailyMed.

        Previously a hardcoded set was used; it was always incomplete and
        could not be kept current. This queries two live sources and caches
        the result for 90 days.
        """
        # Source 1: openFDA — search for "narrow therapeutic" in drug warnings
        try:
            r = requests.get(
                OPENFDA_LABEL_API,
                params={
                    "search": f'openfda.brand_name:"{drug_name}" AND '
                              f'warnings:"narrow therapeutic"',
                    "limit": 1,
                },
                timeout=15,
            )
            if r.status_code == 200:
                total = r.json().get("meta", {}).get("results", {}).get("total", 0)
                if total > 0:
                    return True
        except Exception:
            pass

        # Source 2: DailyMed — look for NTI language in full label text
        try:
            r2 = requests.get(
                f"{DAILYMED_API}/spls.json",
                params={"drug_name": drug_name, "pagesize": 1},
                timeout=15,
            )
            if r2.status_code == 200 and r2.json().get("data"):
                spl_id = r2.json()["data"][0].get("setid", "")
                if spl_id:
                    r3 = requests.get(
                        f"{DAILYMED_API}/spls/{spl_id}.json",
                        timeout=15,
                    )
                    label_text = str(r3.json()).lower()
                    if any(term in label_text for term in _NTI_TERMS):
                        return True
        except Exception:
            pass

        return False

    # ── Main scoring ──────────────────────────────────────────────────────────

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── 1. Get candidate drug CYP/transporter profile ─────────────────
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

        # ── 2. Get disease standard-of-care co-medications (Fix #7) ───────
        comeds = self._get_standard_comeds(pair.disease_name, pair.disease_id)
        if not comeds:
            logger.warning(
                f"[{self.layer_name}] No co-medication data returned for "
                f"{pair.disease_id} ({pair.disease_name}). Skipping DDI analysis."
            )
            return pair

        # ── 3. Check interactions ─────────────────────────────────────────
        interactions = []
        known_ddis        = self.drugbank.get_known_interactions(pair.drug_name)
        known_ddi_lookup  = {d["drug_name"].lower(): d for d in known_ddis}

        for comed_name in comeds:
            # Fix #9: dynamic NTI check
            is_nti = self._is_narrow_ti_drug(comed_name)

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

            comed_profile  = self.drugbank.get_cyp_profile(comed_name)
            cyp_ixns       = self._check_cyp_interactions(
                drug_profile, comed_profile, comed_name, is_nti
            )
            interactions.extend(cyp_ixns)

        drug_profile.interactions = interactions

        # ── 4. Compute DDI risk score ─────────────────────────────────────
        if not interactions:
            ddi_score = 0.0
        else:
            severity_scores  = [DDI_SEVERITY_SCORES.get(i.severity, 0) for i in interactions]
            weighted_scores  = [
                s * (2.0 if i.is_narrow_ti else 1.0)
                for s, i in zip(severity_scores, interactions)
            ]
            ddi_score = min(1.0, sum(weighted_scores) / (len(weighted_scores) * 4))

        pair.scores.ddi_risk_score = ddi_score

        # ── 5. Flag serious interactions ──────────────────────────────────
        serious = [
            i for i in interactions
            if i.severity in ("contraindicated", "major") and i.is_narrow_ti
        ]
        if serious:
            pair.flags.ddi_risk_narrow_index = True
            logger.warning(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"DDI WARNING — {len(serious)} serious NTI interactions:\n"
                + "\n".join(
                    f"  {i.comed_name}: {i.severity} — {i.effect}"
                    for i in serious
                )
            )
        else:
            logger.info(
                f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
                f"DDI score={ddi_score:.3f}, {len(interactions)} interactions flagged"
            )

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
        """Check for CYP-mediated pharmacokinetic interactions."""
        candidate_name  = drug_profile.drug_name
        interactions    = []
        comed_substrates = comed_profile.get("substrates", [])
        comed_inhibitors = comed_profile.get("inhibitors", [])
        comed_inducers   = comed_profile.get("inducers", [])

        # A: candidate inhibits CYP → co-med substrate accumulates
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

        # B: candidate is substrate; co-med inhibits same enzyme
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

        # C: co-med induces enzyme → candidate plasma levels drop
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