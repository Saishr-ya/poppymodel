"""
scripts/validate_ppi.py

Validates that the PPI network is correctly built and labeled with UniProt IDs.
Run after any rebuild of data/processed/ppi_network.pkl.

Usage:
    python scripts/validate_ppi.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.ppi_network import load_graph
from src.layers.layer1b_network_proximity import compute_network_proximity

from src.graph.ppi_network import load_graph
from src.layers.layer1b_network_proximity import compute_network_proximity

G = load_graph()
print(f"\nGraph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Check UniProt IDs are present ─────────────────────────────────────────────
print("\n--- UniProt ID presence check ---")
test_ids = {
    "O76074": "PDE5A (Sildenafil target)",
    "Q13873": "BMPR2 (PAH causal gene)",
    "P37023": "ACVRL1 (PAH causal gene)",
    "P04637": "TP53 (sanity check — should always be present)",
    "P00533": "EGFR (sanity check — should always be present)",
}
for uid, label in test_ids.items():
    present = uid in G.nodes
    status = "✓" if present else "✗ MISSING"
    print(f"  {status}  {uid}  ({label})")

# ── Proximity test: known positive pair ──────────────────────────────────────
print("\n--- Proximity test: Sildenafil x PAH (known positive) ---")
drug_targets = {"O76074"}           # PDE5A
disease_genes = {"Q13873", "P37023"}  # BMPR2, ACVRL1

proximity = compute_network_proximity(drug_targets, disease_genes, G)
print(f"  Proximity score: {proximity}")
if proximity is None:
    print("  ✗ FAIL — returned None. UniProt IDs not in graph or no path exists.")
elif proximity < 2.0:
    print("  ✓ STRONG signal (< 2.0 hops)")
elif proximity < 3.0:
    print("  ✓ MODERATE signal (2.0–3.0 hops)")
elif proximity < 4.0:
    print("  ~ WEAK signal (3.0–4.0 hops) — acceptable for this pair")
else:
    print("  ✗ FAIL — proximity too high (> 4.0). Check ID mapping.")

# ── Proximity test: known negative pair ──────────────────────────────────────
print("\n--- Proximity test: Imatinib x Microcephaly (known negative) ---")
# Imatinib targets: ABL1 (P00519), KIT (P10721)
# Microcephaly genes: ASPM (Q8IZT6), CDK5RAP2 (Q96SN8)
drug_targets_neg = {"P00519", "P10721"}
disease_genes_neg = {"Q8IZT6", "Q96SN8"}

proximity_neg = compute_network_proximity(drug_targets_neg, disease_genes_neg, G)
print(f"  Proximity score: {proximity_neg}")
if proximity_neg is None:
    print("  ~ None — IDs not in graph (acceptable for negative control)")
elif proximity_neg > 2.0:
    print(f"  ✓ High distance ({proximity_neg:.2f}) — correctly distant for negative pair")
else:
    print(f"  ~ Unexpectedly close ({proximity_neg:.2f}) — review this pair")

print("\n--- Done ---")