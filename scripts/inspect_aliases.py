"""
scripts/inspect_aliases.py

Inspects the STRING aliases file to find what source types are available.
Run once to diagnose the correct source string to use for canonical UniProt mapping.

Usage:
    python scripts/inspect_aliases.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gzip
from collections import Counter

aliases_path = "data/raw/string_db/9606.protein.aliases.v12.0.txt.gz"

print(f"Inspecting: {aliases_path}\n")

source_counter = Counter()
uniprot_examples = {}   # source_type → example (string_id, alias)
line_count = 0

with gzip.open(aliases_path, "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        string_id = parts[0]
        alias = parts[1]
        sources = parts[2]

        for source in sources.split():
            source_counter[source] += 1
            # Capture examples for sources that look UniProt-related
            if "uniprot" in source.lower() or "swiss" in source.lower():
                if source not in uniprot_examples:
                    uniprot_examples[source] = (string_id, alias)

        line_count += 1
        if line_count > 2_000_000:   # cap at 2M lines for speed
            break

print(f"Lines inspected: {line_count:,}")
print(f"\n--- All source types found (sorted by frequency) ---")
for source, count in source_counter.most_common():
    print(f"  {count:>8,}  {source}")

print(f"\n--- UniProt-related sources with examples ---")
if uniprot_examples:
    for source, (sid, alias) in uniprot_examples.items():
        print(f"  {source}")
        print(f"    example: {sid} → {alias}")
else:
    print("  None found — checking all sources with 'AC' in name:")
    for source, count in source_counter.most_common():
        if "AC" in source or "ac" in source:
            print(f"  {count:>8,}  {source}")

print("\n--- Done ---")