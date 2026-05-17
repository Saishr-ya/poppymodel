"""
debug_apis2.py — Run from poppymodel root.
Finds the working DisGeNET endpoint and confirms ChEMBL/PharmGKB data.

Usage: python debug_apis2.py
"""

from dotenv import load_dotenv
load_dotenv()

import os, requests, time, json

SEP = "─" * 60
key = os.getenv("DISGENET_API_KEY", "")
headers_json = {"Authorization": f"Bearer {key}", "Accept": "application/json"}

# ── 1. DisGeNET new API — test every plausible endpoint ──────
print(SEP)
print("1. DISGENET.COM — finding the working endpoint")
print(SEP)

candidates = [
    ("GET /api/v1/gda/disease/ORPHA:422",
     "https://api.disgenet.com/api/v1/gda/disease/ORPHA:422", {}),
    ("GET /api/v1/gda?disease_id=ORPHA:422",
     "https://api.disgenet.com/api/v1/gda", {"disease_id": "ORPHA:422", "limit": 5}),
    ("GET /api/v1/gda?disease_id=C0020542 (UMLS CUI for PAH)",
     "https://api.disgenet.com/api/v1/gda", {"disease_id": "C0020542", "limit": 5}),
    ("GET /api/v1/entity/disease/ORPHA:422/genes",
     "https://api.disgenet.com/api/v1/entity/disease/ORPHA:422/genes", {}),
    ("GET /api/v1/disease/ORPHA:422/gda",
     "https://api.disgenet.com/api/v1/disease/ORPHA:422/gda", {"limit": 5}),
    ("GET /api/v1/gene-disease-associations?disease=ORPHA:422",
     "https://api.disgenet.com/api/v1/gene-disease-associations",
     {"disease": "ORPHA:422", "limit": 5}),
]

working_endpoint = None
for label, url, params in candidates:
    try:
        r = requests.get(url, headers=headers_json, params=params, timeout=10)
        ct = r.headers.get("Content-Type", "")
        is_json = "json" in ct or (r.text.strip().startswith("[") or r.text.strip().startswith("{"))
        body_preview = r.text[:120].replace("\n", " ")
        status = "✓ JSON!" if (r.status_code == 200 and is_json) else f"✗ {r.status_code}"
        print(f"  {status}  {label}")
        if r.status_code == 200 and is_json:
            working_endpoint = (url, params)
            try:
                data = r.json()
                count = len(data) if isinstance(data, list) else data.get("total", "?")
                print(f"         → {count} results")
                if isinstance(data, list) and data:
                    first = data[0]
                    print(f"         → first keys: {list(first.keys())[:6]}")
            except Exception:
                print(f"         → body: {body_preview}")
        elif r.status_code not in (404, 405):
            print(f"         → {body_preview}")
        time.sleep(0.3)
    except Exception as e:
        print(f"  ✗ ERROR  {label}: {e}")

if not working_endpoint:
    print("\n  No endpoint worked. Trying root to see what's available...")
    r = requests.get("https://api.disgenet.com/api/v1", headers=headers_json, timeout=10)
    print(f"  Root status: {r.status_code}")
    print(f"  Root body:   {r.text[:300]}")

# ── 2. ChEMBL activity — real binding data for Sildenafil ────
print()
print(SEP)
print("2. CHEMBL ACTIVITY — real binding data for CHEMBL1520 (Sildenafil)")
print(SEP)
r = requests.get(
    "https://www.ebi.ac.uk/chembl/api/data/activity.json",
    params={
        "molecule_chembl_id": "CHEMBL1520",
        "standard_type__in": "Ki,IC50,Kd",
        "standard_value__lte": 100,   # nM — only potent interactions
        "assay_type": "B",            # binding assays only
        "limit": 10,
    },
    timeout=20
)
data = r.json()
print(f"  total_count: {data['page_meta']['total_count']}")
print(f"  (binding assays with IC50/Ki/Kd ≤ 100nM)")
seen = set()
for a in data.get("activities", []):
    tid = a.get("target_chembl_id")
    if tid and tid not in seen:
        seen.add(tid)
        print(f"  → target={tid}, type={a.get('standard_type')}, "
              f"value={a.get('standard_value')} {a.get('standard_units')}")

# Same for Imatinib
print()
print("  CHEMBL192 (Imatinib) binding data:")
r2 = requests.get(
    "https://www.ebi.ac.uk/chembl/api/data/activity.json",
    params={
        "molecule_chembl_id": "CHEMBL192",
        "standard_type__in": "Ki,IC50,Kd",
        "standard_value__lte": 100,
        "assay_type": "B",
        "limit": 10,
    },
    timeout=20
)
data2 = r2.json()
print(f"  total_count: {data2['page_meta']['total_count']}")
seen2 = set()
for a in data2.get("activities", []):
    tid = a.get("target_chembl_id")
    if tid and tid not in seen2:
        seen2.add(tid)
        print(f"  → target={tid}, type={a.get('standard_type')}, "
              f"value={a.get('standard_value')} {a.get('standard_units')}")

# ── 3. PharmGKB — CYP data for PA451346 (Sildenafil) ────────
print()
print(SEP)
print("3. PHARMGKB — CYP relationships for PA451346 (Sildenafil)")
print(SEP)
r = requests.get(
    "https://api.pharmgkb.org/v1/data/drug/PA451346",
    params={"view": "max"},
    timeout=15
)
if r.status_code == 200:
    data = r.json().get("data", {})
    # look for CYP-related pathways or labels
    pathways = data.get("pathways", [])
    print(f"  Pathways: {len(pathways)}")
    for p in pathways[:3]:
        print(f"    → {p.get('name')}")

    # try the relationships endpoint
    print()
    r2 = requests.get(
        "https://api.pharmgkb.org/v1/data/drug/PA451346/relationships",
        params={"view": "max", "limit": 20},
        timeout=15
    )
    print(f"  Relationships status: {r2.status_code}")
    if r2.status_code == 200:
        rels = r2.json().get("data", [])
        print(f"  Total relationships: {len(rels)}")
        for rel in rels[:10]:
            entity = rel.get("entity1") or rel.get("entity2") or {}
            name = entity.get("name", "")
            rtype = rel.get("relationType", "")
            if "CYP" in name.upper() or "metaboli" in rtype.lower():
                print(f"    CYP → {name}: {rtype}")
else:
    # Try the drug-label endpoint which has CYP data
    print(f"  PA451346 direct: {r.status_code}")
    print("  Trying drug label endpoint...")
    r3 = requests.get(
        "https://api.pharmgkb.org/v1/data/drugLabel",
        params={"drug": "PA451346", "view": "max", "limit": 5},
        timeout=15
    )
    print(f"  Drug labels status: {r3.status_code}")
    if r3.status_code == 200:
        labels = r3.json().get("data", [])
        print(f"  Labels: {len(labels)}")
        for label in labels[:3]:
            genes = label.get("relatedGenes", [])
            print(f"    → {label.get('name')}: genes={[g.get('symbol') for g in genes[:5]]}")

print()
print(SEP)
print("DONE")
print(SEP)