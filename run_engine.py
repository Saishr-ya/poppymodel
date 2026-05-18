#!/usr/bin/env python3
"""
run_engine.py — Main CLI for the drug repurposing engine.

Fix #14: cmd_validate previously disabled Layer 1B (network proximity) via
         enable_layer1b_network_proximity=False. This meant every validate run
         was measuring a deliberately crippled engine, so the AUROC baseline was
         artificially low. The flag has been removed so validate uses the same
         layer configuration as a real batch run.

Usage examples:

  # Score a single pair
  python run_engine.py score \
      --drug-id CHEMBL1520 --drug-name Sildenafil \
      --disease-id ORPHA:422 --disease-name "Pulmonary arterial hypertension"

  # Score all pairs in a config file
  python run_engine.py batch \
      --input data/target_pairs.json \
      --output results/run_2025_q1/

  # Run validation against ground truth
  python run_engine.py validate

  # Generate a candidate report for existing results
  python run_engine.py report --results results/run_2025_q1/scored_pairs.json
"""

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("engine")


def cmd_score(args):
    """Score a single drug-disease pair and print the result."""
    from src.scoring.engine import ScoringEngine, EngineConfig

    config = EngineConfig(
        disgenet_api_key=os.getenv("DISGENET_API_KEY", ""),
        openfda_api_key=os.getenv("OPENFDA_API_KEY", ""),
    )
    engine = ScoringEngine.build(config)

    pair = engine.score_pair(
        drug_id=args.drug_id,
        drug_name=args.drug_name,
        disease_id=args.disease_id,
        disease_name=args.disease_name,
    )

    result      = pair.to_dict()
    explanation = engine.scorer.score_explanation(pair)

    print("\n" + "=" * 60)
    print(f"RESULT: {pair.drug_name} × {pair.disease_name}")
    print("=" * 60)
    print(f"Composite score:  {pair.composite_score:.4f}")
    print(f"Business total:   {pair.scores.business_total}/30")
    print(f"Disqualified:     {pair.flags.is_disqualified}")
    if pair.flags.is_disqualified:
        print(f"Reason:           {pair.flags.disqualify_reason}")

    print("\nScore components:")
    for key, comp in explanation.get("components", {}).items():
        norm   = comp.get("normalized")
        contrib = comp.get("contribution")
        if norm is not None:
            print(f"  {key:<40} norm={norm:.3f}  contrib={contrib:.4f}")

    if args.json:
        print("\nFull JSON output:")
        print(json.dumps(result, indent=2, default=str))


def cmd_batch(args):
    """Score all pairs in an input JSON file."""
    from src.scoring.engine import ScoringEngine, EngineConfig

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path) as f:
        drug_disease_pairs = json.load(f)

    logger.info(f"Loaded {len(drug_disease_pairs)} pairs from {input_path}")

    config = EngineConfig(
        disgenet_api_key=os.getenv("DISGENET_API_KEY", ""),
        openfda_api_key=os.getenv("OPENFDA_API_KEY", ""),
    )
    engine = ScoringEngine.build(config)

    run_id        = str(uuid.uuid4())[:8]
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run ID: {run_id}")

    ranked_pairs = engine.score_batch(drug_disease_pairs)

    results_path = output_dir / f"scored_pairs_{run_timestamp}.json"
    results      = [p.to_dict() for p in ranked_pairs]
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    report_path = output_dir / f"report_{run_timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(engine.report(ranked_pairs))
    logger.info(f"Report saved to {report_path}")

    print(engine.report(ranked_pairs))


def cmd_validate(args):
    """
    Run validation metrics against ground truth pairs.

    Fix #14: Removed enable_layer1b_network_proximity=False. The previous stub
    measured an artificially weakened engine — the real AUROC baseline requires
    all production layers to be active. Layer 1B will be skipped automatically
    if the PPI graph file hasn't been built yet (no crash, just a warning).
    """
    from src.scoring.engine import ScoringEngine, EngineConfig
    from src.validation.ground_truth import load_ground_truth, ValidationMetrics

    # Fix #14: no longer disabling Layer 1B here
    config = EngineConfig(
        disgenet_api_key=os.getenv("DISGENET_API_KEY", ""),
    )
    engine = ScoringEngine.build(config)

    gt_pairs = load_ground_truth()
    logger.info(f"Loaded {len(gt_pairs)} ground truth pairs")

    inputs = [
        {
            "drug_id":      p.drug_id,
            "drug_name":    p.drug_name,
            "disease_id":   p.disease_id,
            "disease_name": p.disease_name,
        }
        for p in gt_pairs
    ]
    ranked = engine.score_batch(inputs)

    try:
        auroc = ValidationMetrics.auroc(ranked, gt_pairs)
        print(f"\nAUROC: {auroc:.4f}  (target: > 0.75)")
    except ValueError as e:
        print(f"\nAUROC: could not compute — {e}")

    p_at_20 = ValidationMetrics.precision_at_k(ranked, gt_pairs, k=20)
    print(f"Precision@20: {p_at_20:.4f}  (target: > 0.40)")

    fn_analysis = ValidationMetrics.false_negative_analysis(ranked, gt_pairs)
    if fn_analysis:
        print(f"\nFalse negatives (known positives in bottom 50%): {len(fn_analysis)}")
        for fn in fn_analysis:
            print(
                f"  [{fn['rank']:4d}] {fn['drug']} × {fn['disease']} "
                f"(score={fn['composite_score']:.3f})"
            )
            print(f"         Evidence: {fn['evidence_source']}")


def cmd_report(args):
    """Generate report from existing results JSON."""
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    print(f"Report for: {results_path}")
    print(f"Total pairs: {len(results)}")
    print(f"Disqualified: {sum(1 for r in results if r.get('is_disqualified'))}")
    print(f"\nTop 20 by composite score:")
    eligible = [r for r in results if not r.get("is_disqualified")]
    eligible.sort(key=lambda x: x.get("composite_score") or 0, reverse=True)
    for i, r in enumerate(eligible[:20], 1):
        print(
            f"  #{i:2d} [{r.get('composite_score', 0):.4f}] "
            f"{r.get('drug_name')} × {r.get('disease_name')} "
            f"(biz={r.get('business_total')}/30)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Drug Repurposing Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    score_parser = subparsers.add_parser("score", help="Score a single drug-disease pair")
    score_parser.add_argument("--drug-id",      required=True, help="ChEMBL ID (e.g., CHEMBL192)")
    score_parser.add_argument("--drug-name",    required=True)
    score_parser.add_argument("--disease-id",   required=True, help="ORPHA:XXXXX or OMIM:XXXXXX")
    score_parser.add_argument("--disease-name", required=True)
    score_parser.add_argument("--json", action="store_true", help="Output full JSON")

    batch_parser = subparsers.add_parser("batch", help="Score all pairs in a JSON file")
    batch_parser.add_argument("--input",  required=True, help="Path to input JSON")
    batch_parser.add_argument("--output", required=True, help="Output directory")

    subparsers.add_parser("validate", help="Validate against ground truth")

    report_parser = subparsers.add_parser("report", help="Generate report from results JSON")
    report_parser.add_argument("--results", required=True, help="Path to scored_pairs.json")

    args = parser.parse_args()

    if args.command == "score":
        cmd_score(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()