#!/usr/bin/env python3
"""
inference.py — BIS Standards Recommendation Engine
Hackathon submission entry point.

Usage:
    python inference.py --input queries.json --output results.json

Input JSON schema (list of objects):
    [{"id": "q1", "query": "Ordinary Portland Cement 53 grade ..."}]

Output JSON schema (strict — do not change key names):
    [
      {
        "id": "q1",
        "retrieved_standards": ["IS 12269", "IS 8112", "IS 456"],
        "latency_seconds": 0.042
      }
    ]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add src/ to path so imports resolve regardless of working directory
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine import retrieve  # type: ignore


def run_inference(input_path: str, output_path: str, top_k: int = 5) -> None:
    """
    Read queries from input_path, run RAG pipeline on each,
    and write results to output_path in the required JSON schema.
    """
    # ── 1. Load input ────────────────────────────────────────────────────────
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        try:
            queries = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse input JSON: {e}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(queries, list):
        print("[ERROR] Input JSON must be a list of query objects.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loaded {len(queries)} queries from {input_path}")

    # ── 2. Run pipeline ──────────────────────────────────────────────────────
    results = []
    total_latency = 0.0

    for i, item in enumerate(queries):
        query_id = item.get("id", str(i))
        query_text = item.get("query", "").strip()

        if not query_text:
            print(f"[WARN] Empty query for id={query_id}, skipping.", file=sys.stderr)
            results.append({
                "id": query_id,
                "retrieved_standards": [],
                "latency_seconds": 0.0,
            })
            continue

        # Time the retrieval
        t_start = time.perf_counter()
        retrieved = retrieve(query_text, top_k=top_k)
        t_end = time.perf_counter()
        latency = round(t_end - t_start, 4)
        total_latency += latency

        # Extract IS codes in ranked order (required output field)
        standard_codes = [r["code"] for r in retrieved]

        results.append({
            "id": query_id,
            "retrieved_standards": standard_codes,
            "latency_seconds": latency,
        })

        # Progress logging every 10 queries
        if (i + 1) % 10 == 0 or (i + 1) == len(queries):
            avg_lat = round(total_latency / (i + 1), 4)
            print(f"[INFO] Processed {i + 1}/{len(queries)} queries | avg latency: {avg_lat}s")

    # ── 3. Save output ───────────────────────────────────────────────────────
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_latency = round(total_latency / max(len(queries), 1), 4)
    print(f"[INFO] Done. Results saved to {output_path}")
    print(f"[INFO] Total queries: {len(queries)} | Avg latency: {avg_latency}s")


def main():
    parser = argparse.ArgumentParser(
        description="BIS Standards Recommendation Engine — Inference Script"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file containing list of {id, query} objects",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write output JSON results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top standards to retrieve per query (default: 5)",
    )
    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
