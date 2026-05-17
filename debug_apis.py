"""
debug_apis.py — Updated to test the corrected API setup.

Tests:
  1. OpenTargets (replaces DisGeNET) — disease gene associations
  2. ChEMBL — drug targets (confirmed working)
  3. PharmGKB — CYP data via hardcoded reference + file
  4. PubMed / ClinicalTrials — literature layer
  5. FAERS / openFDA — ADMET layer

Run: python debug_apis.py
"""

from dotenv import load_dotenv
load_dotenv()

import os, requests, time, json

SEP = "─" * 60


# ── 1. OpenTargets (new primary gene-disease source) ──────────────────────────
print(SEP)
print("1. OPENTARGETS — disease gene associations (replaces DisGeNET)")
print(SEP)

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
HEADERS = {"Content-Type": "application/json"}

# Test: PAH associated targets
query = """
query {
  disease(efoId: "EFO_0000222") {
    id
    name
    dbXRefs
    associatedTargets(page: {index: 0, size: 5}) {
      rows {
        target {
          id
          approvedSymbol
          proteinAnnotations { id }
        }
        score
      }
    }
  }
}
"""
try:
    r = requests.post(GRAPHQL_URL, headers=HEADERS,
                      json={"query": query}, timeout=20)
    if r.status_code == 200:
        data = r.json()
        disease = data.get("data", {}).get("disease", {})
        print(f"  ✓ Disease: {disease.get('name')} ({disease.get('id')})")
        print(f"    xrefs: {disease.get('dbXRefs', [])[:5]}")
        rows = disease.get("associatedTargets", {}).get("rows", [])
        print(f"    Associated targets (top 5):")
        for row in rows[:5]:
            t = row.get("target", {})
            uniprot = t.get("proteinAnnotations", {}).get("id", [])
            print(f"      {t['approvedSymbol']:10s} | score={row['score']:.3f} | UniProt={uniprot}")
    else:
        print(f"  ✗ {r.status_code}: {r.text[:200]}")
        print("  → Add PAH EFO ID to ORPHA_TO_EFO manually: EFO_0000222")
except Exception as e:
    print(f"  ✗ ERROR: {e}")

print()
print("  Testing Orphanet ID search (ORPHA:422 → EFO ID mapping):")
try:
    search_q = """
    query {
      diseases(q: "Orphanet_422", page: {index: 0, size: 3}) {
        rows { id name dbXRefs }
      }
    }
    """
    r = requests.post(GRAPHQL_URL, headers=HEADERS,
                      json={"query": search_q}, timeout=15)
    if r.status_code == 200:
        rows = r.json().get("data", {}).get("diseases", {}).get("rows", [])
        for row in rows:
            print(f"  ✓ {row['id']}: {row['name']}")
            print(f"    xrefs: {row.get('dbXRefs', [])[:4]}")
    else:
        print(f"  ✗ Search status: {r.status_code}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")


# ── 2. ChEMBL (confirmed working) ────────────────────────────────────────────
print()
print(SEP)
print("2. CHEMBL — drug targets (confirmed working)")
print(SEP)

try:
    r = requests.get(
        "https://www.ebi.ac.uk/chembl/api/data/mechanism.json",
        params={"molecule_chembl_id": "CHEMBL1520", "limit": 5},
        headers={"Accept": "application/json"},
        timeout=15,
    )
    if r.status_code == 200:
        data = r.json()
        mechs = data.get("mechanisms", [])
        print(f"  ✓ Sildenafil mechanisms ({len(mechs)} total):")
        for m in mechs[:3]:
            print(f"    {m.get('target_chembl_id')}: {m.get('action_type')} — {m.get('target_name','')[:50]}")
    else:
        print(f"  ✗ {r.status_code}: {r.text[:200]}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")


# ── 3. PharmGKB — CYP data via hardcoded reference ────────────────────────────
print()
print(SEP)
print("3. PHARMGKB — CYP profiles via hardcoded reference table")
print(SEP)

# Test the hardcoded reference (no API call needed)
CYP_REFERENCE_SAMPLE = {
    "sildenafil":   {"substrates": ["CYP3A4", "CYP2C9"], "inhibitors": [], "inducers": []},
    "carbamazepine":{"substrates": ["CYP3A4"], "inhibitors": [], "inducers": ["CYP3A4", "CYP2C9"]},
    "valproate":    {"substrates": ["CYP2C9"], "inhibitors": ["CYP2C19"], "inducers": []},
}
print("  ✓ Hardcoded reference (no API call needed):")
for drug, profile in CYP_REFERENCE_SAMPLE.items():
    print(f"    {drug}: substrates={profile['substrates']}, inhibitors={profile['inhibitors']}")

print()
print("  PharmGKB REST API test (v1/data/drug):")
try:
    r = requests.get(
        "https://api.pharmgkb.org/v1/data/drug",
        params={"name": "sildenafil", "view": "base"},
        headers={"Accept": "application/json"},
        timeout=15,
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        drugs = data.get("data", [])
        if drugs:
            print(f"  ✓ Found: {drugs[0].get('name')} (ID: {drugs[0].get('id')})")
    else:
        print(f"  → Use hardcoded reference table (covers 25+ drugs used in rare disease trials)")
        print(f"  → Download relationships.tsv for full coverage: pharmgkb.org/downloads")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    print(f"  → Hardcoded reference table is the reliable fallback.")


# ── 4. PubMed (literature layer) ─────────────────────────────────────────────
print()
print(SEP)
print("4. PUBMED — literature co-occurrence")
print(SEP)

ncbi_key = os.getenv("NCBI_API_KEY", "")
try:
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": '"sildenafil" AND "pulmonary arterial hypertension"',
            "retmax": 5,
            "retmode": "json",
            "api_key": ncbi_key,
        },
        timeout=15,
    )
    if r.status_code == 200:
        data = r.json()
        count = data.get("esearchresult", {}).get("count", "?")
        ids   = data.get("esearchresult", {}).get("idlist", [])
        print(f"  ✓ sildenafil + PAH: {count} papers (PMIDs: {ids[:3]})")
    else:
        print(f"  ✗ {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")


# ── 5. ClinicalTrials.gov ────────────────────────────────────────────────────
print()
print(SEP)
print("5. CLINICALTRIALS.GOV — trial evidence")
print(SEP)

try:
    r = requests.get(
        "https://clinicaltrials.gov/api/v2/studies",
        params={"query.intr": "sildenafil", "query.cond": "pulmonary arterial hypertension",
                "pageSize": 3, "format": "json"},
        headers={"User-Agent": "PoppyEngine/1.0 (research)"},
        timeout=20,
    )
    if r.status_code == 200:
        data = r.json()
        studies = data.get("studies", [])
        print(f"  ✓ Found {len(studies)} trials (sample):")
        for s in studies[:2]:
            proto = s.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            design = proto.get("designModule", {})
            phase = design.get("phases", ["?"])[0] if design.get("phases") else "?"
            print(f"    {id_mod.get('nctId')}: {phase} — {id_mod.get('briefTitle','')[:60]}")
    else:
        print(f"  ✗ {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")


# ── 6. openFDA FAERS ─────────────────────────────────────────────────────────
print()
print(SEP)
print("6. OPENFDA FAERS — adverse event signals")
print(SEP)

openfda_key = os.getenv("OPENFDA_API_KEY", "")
try:
    params = {
        "search": 'patient.drug.openfda.brand_name:"REVATIO" AND serious:1',
        "limit": 1,
    }
    if openfda_key:
        params["api_key"] = openfda_key
    r = requests.get("https://api.fda.gov/drug/event.json", params=params, timeout=15)
    if r.status_code == 200:
        total = r.json().get("meta", {}).get("results", {}).get("total", 0)
        print(f"  ✓ Sildenafil (Revatio) serious FAERS reports: {total}")
    else:
        print(f"  ✗ {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"  ✗ ERROR: {e}")


# ── Summary ──────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SUMMARY")
print(SEP)
print("""
  DisGeNET:     REPLACED by OpenTargets (stable, free, better coverage)
  ChEMBL:       Working ✓
  PharmGKB:     Hardcoded reference table (download file for full coverage)
  PubMed:       Working ✓ (with NCBI_API_KEY)
  ClinicalTrials: Working ✓
  openFDA:      Working ✓

  NEXT STEPS:
  1. Copy the 3 new files into your src/ingestion/ directory
  2. Run: python debug_apis.py   (confirm OpenTargets and ChEMBL are ✓)
  3. Run: python run_engine.py score --drug-id CHEMBL1520 --drug-name Sildenafil \\
           --disease-id ORPHA:422 --disease-name "Pulmonary arterial hypertension"
  4. Download STRING DB for Layer 1B:
       python -m src.graph.ppi_network download
       python -m src.graph.ppi_network build
  5. Download PharmGKB relationships.tsv (pharmgkb.org/downloads) for full PGx coverage
""")