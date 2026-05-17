"""
scripts/debug_graph_nodes.py

Diagnoses what format the PPI graph nodes are actually stored in.
Run this when validate_ppi.py shows UniProt IDs as MISSING.

Usage:
    python scripts/debug_graph_nodes.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.ppi_network import load_graph

G = load_graph()
print(f"\nGraph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Sample 20 nodes to see what format they are ──────────────────────────────
nodes = list(G.nodes)
print(f"\n--- First 20 nodes (to check ID format) ---")
for n in nodes[:20]:
    print(f"  {n}")

# ── Search for known proteins by partial ID ───────────────────────────────────
print(f"\n--- Searching for EGFR (canonical: P00533) ---")
egfr_hits = [n for n in nodes if "533" in str(n)]
print(f"  Nodes containing '533': {egfr_hits[:10]}")

print(f"\n--- Searching for PDE5A (canonical: O76074) ---")
pde5_hits = [n for n in nodes if "76074" in str(n)]
print(f"  Nodes containing '76074': {pde5_hits[:10]}")

print(f"\n--- Searching for BMPR2 (canonical: Q13873) ---")
bmpr2_hits = [n for n in nodes if "13873" in str(n)]
print(f"  Nodes containing '13873': {bmpr2_hits[:10]}")

print(f"\n--- Searching for TP53 (canonical: P04637) ---")
tp53_hits = [n for n in nodes if "04637" in str(n)]
print(f"  Nodes containing '04637': {tp53_hits[:10]}")

# ── Check what the aliases file actually mapped ───────────────────────────────
import json
map_path = "data/processed/string_to_uniprot.json"
if os.path.exists(map_path):
    with open(map_path) as f:
        uniprot_map = json.load(f)
    print(f"\n--- Sample of string_to_uniprot.json (first 20 entries) ---")
    for k, v in list(uniprot_map.items())[:20]:
        print(f"  {k} → {v}")
else:
    print(f"\n--- string_to_uniprot.json not found at {map_path} ---")

print("\n--- Done ---")