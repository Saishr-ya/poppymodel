"""
src/layers/layer5_literature.py

Layer 5 — Literature and Clinical Evidence Scoring.

Signals computed:
  1. PubMed co-occurrence score (weighted paper count)
  2. ClinicalTrials.gov evidence score (0–5 scale)
  3. Case report count

Data sources:
  - PubMed E-utilities: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
  - ClinicalTrials.gov v2 API: https://clinicaltrials.gov/api/v2/

Clinical trial evidence scale (from the engine spec):
  Phase III positive result = 5
  Phase II positive result  = 4
  Phase I                   = 3
  Observational study       = 2
  No trial                  = 0

Bio team:
  - Weight Indian-author papers higher (signals off-label use in target market)
  - Search PubMed for both drug name AND disease name, plus synonyms
  - Case reports in Indian journals (IJDVL, Indian Journal of Pediatrics, JIMD Reports)
    are particularly valuable — they signal existing off-label physician behavior
"""

from __future__ import annotations
import logging
import math
import os
import time
from datetime import datetime
from typing import Optional

import requests

from src.layers.base import BaseLayer
from src.scoring.candidate import CandidatePair
from src.ingestion.cache import cached_api_call

logger = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2"

# Add to .env as NCBI_API_KEY — raises rate limit from 3 req/sec to 10 req/sec.
# Register free at: https://www.ncbi.nlm.nih.gov/account/
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Polite User-Agent for ClinicalTrials — prevents silent rate-limiting during batch runs.
# Replace your@email.com with a real contact email.
CLINICALTRIALS_HEADERS = {
    "User-Agent": "PoppyEngine/1.0 (drug repurposing research; contact shruthisathya@g.ucla.edu)"
}

# Indian journals that signal off-label use in target market
INDIAN_JOURNALS = {
    "Indian Journal of Pediatrics",
    "Indian Journal of Medical Research",
    "JIMD Reports",
    "Annals of Indian Academy of Neurology",
    "Journal of Rare Diseases",
    "Indian Pediatrics",
    "Indian Journal of Human Genetics",
}

CURRENT_YEAR = datetime.now().year


class PubMedClient:
    """Queries PubMed E-utilities for co-occurrence and case report detection."""

    def _ncbi_params(self, extra: dict) -> dict:
        """
        Build params dict for NCBI requests, injecting the API key when available.
        Without the key: 3 requests/second limit.
        With the key: 10 requests/second limit.
        """
        params = {**extra}
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY
        return params

    @cached_api_call(ttl_seconds=86400 * 7)   # 7-day cache
    def search(self, query: str, max_results: int = 500) -> list[dict]:
        """
        Run a PubMed search and return article metadata (PMID, year, journal, title).

        Uses PMIDs directly for esummary rather than WebEnv history server,
        which is more reliable when results are served from Redis cache.
        """
        # Step 1: esearch to get PMIDs
        esearch_url = f"{PUBMED_BASE}/esearch.fcgi"
        params = self._ncbi_params({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        })
        try:
            r = requests.get(esearch_url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return []

            # Step 2: esummary using PMIDs directly (not WebEnv — more cache-stable)
            time.sleep(0.4)
            esummary_url = f"{PUBMED_BASE}/esummary.fcgi"
            summary_params = self._ncbi_params({
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmax": max_results,
                "retmode": "json",
            })
            r2 = requests.get(esummary_url, params=summary_params, timeout=20)
            r2.raise_for_status()
            summary = r2.json()

            articles = []
            for pmid, article in summary.get("result", {}).items():
                if pmid == "uids":
                    continue
                pub_date = article.get("pubdate", "")
                year = int(pub_date[:4]) if len(pub_date) >= 4 and pub_date[:4].isdigit() else 0
                articles.append({
                    "pmid": pmid,
                    "year": year,
                    "journal": article.get("source", ""),
                    "title": article.get("title", ""),
                    "authors": article.get("authors", []),
                })
            return articles

        except Exception as e:
            logger.error(f"PubMed search failed for '{query}': {e}")
            return []

    @cached_api_call(ttl_seconds=86400 * 7)
    def cooccurrence_score(
        self,
        drug_name: str,
        disease_name: str,
        recency_half_life_years: int = 5,
        indian_weight: float = 2.0,
    ) -> float:
        """
        Compute weighted co-occurrence score.
        Recent papers weighted more heavily (exponential decay by age).
        Indian-journal papers weighted by indian_weight multiplier.

        Cached directly so batch runs don't recompute for the same pair.
        """
        query = f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract]'
        articles = self.search(query, max_results=200)

        if not articles:
            return 0.0

        score = 0.0
        for article in articles:
            age = max(0, CURRENT_YEAR - article.get("year", CURRENT_YEAR))
            recency_weight = math.exp(-0.693 * age / recency_half_life_years)
            journal = article.get("journal", "")
            journal_weight = indian_weight if journal in INDIAN_JOURNALS else 1.0
            score += recency_weight * journal_weight

        return score

    @cached_api_call(ttl_seconds=86400 * 7)
    def count_case_reports(self, drug_name: str, disease_name: str) -> int:
        """Count case reports/series for this drug-disease combination. Cached."""
        query = (
            f'"{drug_name}"[Title/Abstract] AND "{disease_name}"[Title/Abstract] '
            f'AND ("case report"[Publication Type] OR "case series"[Publication Type])'
        )
        articles = self.search(query, max_results=100)
        return len(articles)


class ClinicalTrialsClient:
    """Queries ClinicalTrials.gov v2 API for trial evidence."""

    @cached_api_call(ttl_seconds=86400 * 3)   # 3-day cache (trials update frequently)
    def search_trials(self, drug_name: str, condition: str) -> list[dict]:
        """
        Search ClinicalTrials.gov for trials of this drug in this condition.
        Sends a User-Agent header to avoid silent rate-limiting during batch runs.
        """
        url = f"{CLINICALTRIALS_BASE}/studies"
        params = {
            "query.intr": drug_name,
            "query.cond": condition,
            "fields": "NCTId,Phase,OverallStatus,BriefTitle,StudyType,PrimaryCompletionDate",
            "pageSize": 50,
            "format": "json",
        }
        try:
            r = requests.get(
                url,
                params=params,
                headers=CLINICALTRIALS_HEADERS,  # polite User-Agent
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            studies = data.get("studies", [])
            trials = []
            for s in studies:
                proto = s.get("protocolSection", {})
                id_module = proto.get("identificationModule", {})
                status_module = proto.get("statusModule", {})
                design_module = proto.get("designModule", {})
                trials.append({
                    "nct_id": id_module.get("nctId"),
                    "title": id_module.get("briefTitle"),
                    "phase": design_module.get("phases", [""])[0] if design_module.get("phases") else "",
                    "status": status_module.get("overallStatus"),
                    "study_type": design_module.get("studyType"),
                })
            return trials
        except Exception as e:
            logger.error(f"ClinicalTrials search failed for {drug_name}/{condition}: {e}")
            return []

    def compute_evidence_score(self, drug_name: str, condition: str) -> int:
        """
        Compute clinical trial evidence score on the 0–5 scale.

        Scale:
            5 = Phase III completed
            4 = Phase II completed or Phase III in progress
            3 = Phase I completed or Phase II in progress
            2 = Observational / compassionate use study
            1 = Any registered trial
            0 = No trials found
        """
        trials = self.search_trials(drug_name, condition)
        if not trials:
            return 0

        best_score = 0
        for trial in trials:
            phase = (trial.get("phase") or "").upper()
            status = (trial.get("status") or "").upper()
            study_type = (trial.get("study_type") or "").upper()

            if "PHASE3" in phase or "PHASE 3" in phase or "III" in phase:
                score = 5 if "COMPLETED" in status else 4
            elif "PHASE2" in phase or "PHASE 2" in phase or "II" in phase:
                score = 4 if "COMPLETED" in status else 3
            elif "PHASE1" in phase or "PHASE 1" in phase or "I" in phase:
                score = 3
            elif "OBSERVATIONAL" in study_type:
                score = 2
            else:
                score = 1

            best_score = max(best_score, score)

        return best_score


class LiteratureLayer(BaseLayer):
    """
    Layer 5 — Literature and clinical evidence scoring.

    Scores:
        pair.scores.pubmed_cooccurrence_score
        pair.scores.clinical_trial_evidence   (0–5)
        pair.scores.case_report_count

    Setup:
        Add NCBI_API_KEY to .env (register at ncbi.nlm.nih.gov/account).
        Update CLINICALTRIALS_HEADERS with your real contact email.
        No keys needed for ClinicalTrials.
    """

    layer_name = "layer5_literature"
    version = "1.1"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.pubmed = PubMedClient()
        self.ct = ClinicalTrialsClient()

    def score(self, pair: CandidatePair) -> CandidatePair:
        # ── PubMed co-occurrence ──────────────────────────────────────────
        cooc = self.pubmed.cooccurrence_score(pair.drug_name, pair.disease_name)
        pair.scores.pubmed_cooccurrence_score = cooc

        # ── Case reports ──────────────────────────────────────────────────
        case_count = self.pubmed.count_case_reports(pair.drug_name, pair.disease_name)
        pair.scores.case_report_count = case_count

        # ── ClinicalTrials.gov evidence ───────────────────────────────────
        ct_score = self.ct.compute_evidence_score(pair.drug_name, pair.disease_name)
        pair.scores.clinical_trial_evidence = ct_score

        logger.debug(
            f"[{self.layer_name}] {pair.drug_name}×{pair.disease_name}: "
            f"cooccurrence={cooc:.1f}, case_reports={case_count}, ct_evidence={ct_score}"
        )

        return pair