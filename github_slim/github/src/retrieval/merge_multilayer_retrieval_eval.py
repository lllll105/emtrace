#!/usr/bin/env python3
"""Merge three completed shards of the 11-class multilayer retrieval evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "outputs/retrieval_eval/multilayer_11class_fullbank_20260918"
METHODS = ("dual_concat", "emotion_only", "emotion_first_fact_rerank_lambda_0p1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-count", type=int, default=3)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records), encoding="utf-8")


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["condition"], record["target_emotion"])].append(record)
    output = []
    for (condition, emotion), group in sorted(grouped.items()):
        output.append({"scope": "per_emotion", "condition": condition, "emotion": emotion, "n": len(group), "emotion_accuracy": sum(row["emotion_accuracy"] for row in group) / len(group), "mean_target_emotion_score": sum(row["target_emotion_score"] for row in group) / len(group), "mean_emotion_gain": sum(row["emotion_gain_vs_baseline"] for row in group) / len(group), "retrieval_emotion_match": sum(row["retrieval_emotion_match"] for row in group) / len(group)})
    for condition in ("baseline", "steered_configured_multilayer"):
        group = [record for record in records if record["condition"] == condition]
        output.append({"scope": "micro", "condition": condition, "emotion": "all", "n": len(group), "emotion_accuracy": sum(row["emotion_accuracy"] for row in group) / len(group), "mean_target_emotion_score": sum(row["target_emotion_score"] for row in group) / len(group), "mean_emotion_gain": sum(row["emotion_gain_vs_baseline"] for row in group) / len(group), "retrieval_emotion_match": sum(row["retrieval_emotion_match"] for row in group) / len(group)})
    return output


def main() -> None:
    args = parse_args()
    for method in METHODS:
        combined = []
        for shard in range(args.shard_count):
            path = args.output_dir / f"shard_{shard:02d}_of_{args.shard_count:02d}" / method / "scored_results.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"missing completed shard: {path}")
            combined.extend(rows(path))
        expected = 2 * 1103
        if len(combined) != expected:
            raise ValueError(f"{method}: expected {expected} scored rows, got {len(combined)}")
        pairs = {(row["memory_id"], row["condition"]) for row in combined}
        if len(pairs) != expected:
            raise ValueError(f"{method}: duplicate or missing query-condition rows")
        method_dir = args.output_dir / "merged" / method
        write_jsonl(method_dir / "scored_results.jsonl", combined)
        report = summarize(combined)
        with (method_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[0]))
            writer.writeheader(); writer.writerows(report)
    print(json.dumps({"status": "merged", "output": str(args.output_dir / "merged"), "methods": METHODS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
