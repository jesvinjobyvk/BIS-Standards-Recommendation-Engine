#!/usr/bin/env python3
"""
eval_script.py — BIS Hackathon Evaluation Script
Computes: Hit Rate @3, MRR @5, and Average Latency

Usage:
    python eval_script.py --predictions results.json --ground_truth ground_truth.json

Predictions JSON (output of inference.py):
    [{"id": "q1", "retrieved_standards": ["IS 12269", "IS 8112"], "latency_seconds": 0.04}]

Ground Truth JSON:
    [{"id": "q1", "relevant_standards": ["IS 12269", "IS 4031"]}]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def load_json(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_code(code: str) -> str:
    """Normalise IS code for comparison: lowercase, strip spaces."""
    return code.lower().replace(" ", "").replace("-", "").strip()


def hit_at_k(retrieved: List[str], relevant: List[str], k: int = 3) -> int:
    """Return 1 if any relevant standard appears in top-k retrieved."""
    retrieved_norm = [normalize_code(r) for r in retrieved[:k]]
    relevant_norm = [normalize_code(r) for r in relevant]
    return int(any(r in retrieved_norm for r in relevant_norm))


def reciprocal_rank_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
    """Return 1/rank of first relevant result in top-k, or 0."""
    relevant_norm = set(normalize_code(r) for r in relevant)
    for rank, code in enumerate(retrieved[:k], start=1):
        if normalize_code(code) in relevant_norm:
            return 1.0 / rank
    return 0.0


def evaluate(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Compute Hit Rate @3, MRR @5, and Average Latency."""
    gt_map: Dict[str, List[str]] = {
        item["id"]: item.get("relevant_standards", [])
        for item in ground_truth
    }

    hit_scores: List[int] = []
    rr_scores: List[float] = []
    latencies: List[float] = []
    missing_ids: List[str] = []

    for pred in predictions:
        qid = pred.get("id")
        retrieved = pred.get("retrieved_standards", [])
        latency = pred.get("latency_seconds", 0.0)

        if qid not in gt_map:
            missing_ids.append(qid)
            continue

        relevant = gt_map[qid]
        latencies.append(float(latency))

        if not relevant:
            # No ground truth for this query — skip metric computation
            continue

        hit_scores.append(hit_at_k(retrieved, relevant, k=3))
        rr_scores.append(reciprocal_rank_at_k(retrieved, relevant, k=5))

    n = len(hit_scores)
    if n == 0:
        print("[WARN] No evaluable queries found.", file=sys.stderr)
        return {"hit_rate_at_3": 0.0, "mrr_at_5": 0.0, "avg_latency_seconds": 0.0, "n_queries": 0}

    hit_rate = round(sum(hit_scores) / n * 100, 2)
    mrr = round(sum(rr_scores) / n, 4)
    avg_latency = round(sum(latencies) / max(len(latencies), 1), 4)

    if missing_ids:
        print(f"[WARN] {len(missing_ids)} prediction IDs not found in ground truth: {missing_ids[:5]}")

    return {
        "hit_rate_at_3": hit_rate,        # percentage, e.g. 85.0
        "mrr_at_5": mrr,                  # 0.0 – 1.0
        "avg_latency_seconds": avg_latency,
        "n_queries_evaluated": n,
        "total_predictions": len(predictions),
    }


def print_report(metrics: Dict) -> None:
    print("\n" + "=" * 50)
    print("  BIS HACKATHON — EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Queries evaluated   : {metrics['n_queries_evaluated']}")
    print(f"  Hit Rate @3         : {metrics['hit_rate_at_3']:.2f}%   (target > 80%)")
    print(f"  MRR @5              : {metrics['mrr_at_5']:.4f}     (target > 0.70)")
    print(f"  Avg Latency         : {metrics['avg_latency_seconds']:.4f}s  (target < 5s)")
    print("=" * 50)

    # Highlight pass/fail
    hr_pass = metrics["hit_rate_at_3"] > 80
    mrr_pass = metrics["mrr_at_5"] > 0.70
    lat_pass = metrics["avg_latency_seconds"] < 5.0

    print(f"  Hit Rate @3 : {'PASS ✓' if hr_pass else 'FAIL ✗'}")
    print(f"  MRR @5      : {'PASS ✓' if mrr_pass else 'FAIL ✗'}")
    print(f"  Latency     : {'PASS ✓' if lat_pass else 'FAIL ✗'}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="BIS Hackathon Evaluation Script")
    parser.add_argument("--predictions", required=True, help="Path to inference.py output JSON")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    predictions = load_json(args.predictions)
    ground_truth = load_json(args.ground_truth)

    metrics = evaluate(predictions, ground_truth)
    print_report(metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Metrics saved to {args.output}")

    return metrics


if __name__ == "__main__":
    main()
