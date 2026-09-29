#!/usr/bin/env python3
"""Merge completed NPTI generation shards and recompute summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("dual_concat", "emotion_only", "emotion_first_fact_rerank_lambda_0p1")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/npti/multilayer_11class_fullbank_20260918")
    parser.add_argument("--shard-count", type=int, default=3)
    setting = parser.parse_args()
    merged = setting.output_dir / "merged"
    for method in METHODS:
        steered: list[dict[str, Any]] = []
        baselines: list[dict[str, Any]] = []
        for shard in range(setting.shard_count):
            prefix = setting.output_dir / f"shard_{shard:02d}_of_{setting.shard_count:02d}"
            steered.extend(rows(prefix / method / "scored_results.jsonl"))
            baselines.extend(rows(prefix / "baseline_scored.jsonl"))
        if len(steered) != 1103 or len(baselines) != 1103:
            raise ValueError(f"{method}: expected 1103 rows after merging shards, got steered={len(steered)}, baseline={len(baselines)}")
        if len({row["memory_id"] for row in steered}) != 1103 or len({row["memory_id"] for row in baselines}) != 1103:
            raise ValueError(f"{method}: duplicate shard rows")
        base = {row["memory_id"]: row for row in baselines}
        for row in steered:
            row["emotion_gain_vs_baseline"] = row["target_emotion_score"] - base[row["memory_id"]]["target_emotion_score"]
        combined = baselines + steered
        write(merged / method / "baseline_scored.jsonl", baselines)
        write(merged / method / "scored_results.jsonl", combined)
        report = []
        for condition, group in (("baseline", baselines), ("npti_all_layers", steered)):
            report.append({"scope": "micro", "condition": condition, "n": len(group), "emotion_accuracy": sum(row["emotion_accuracy"] for row in group) / len(group), "target_score": sum(row["target_emotion_score"] for row in group) / len(group), "emotion_gain": sum(row.get("emotion_gain_vs_baseline", 0.0) for row in group) / len(group), "retrieval_match": sum(row["retrieval_emotion_match"] for row in group) / len(group)})
        with (merged / method / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[0])); writer.writeheader(); writer.writerows(report)
    print(json.dumps({"status": "merged", "output": str(merged), "methods": METHODS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
