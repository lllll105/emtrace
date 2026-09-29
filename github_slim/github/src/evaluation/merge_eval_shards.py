#!/usr/bin/env python3
"""Merge independently evaluated JSONL shards into one evaluation output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_generation_emotion_fact_metrics import summarize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", action="append", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for directory in a.input_dir:
        rows.extend(json.loads(line) for line in (directory / "per_generation_metrics.jsonl").read_text().splitlines() if line.strip())
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "per_generation_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    summarize(rows, a.output_dir)
    (a.output_dir / "config.json").write_text(json.dumps({"rows": len(rows), "input_dirs": [str(x) for x in a.input_dir]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
