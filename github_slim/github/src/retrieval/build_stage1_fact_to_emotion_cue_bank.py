#!/usr/bin/env python3
"""Build Stage-1 of the two-stage RAG memory system.

Stage 1 is a fact-to-emotion-cue lookup table:

    key   : L2-normalized Qwen3 fact embedding (1024 dimensions)
    value : L2-normalized ModernBERT layer-22 emotion-cue embedding (768)

It preserves every actual row from the current regenerated 11-class memory
bank: 6,927 accepted training memories and 1,103 test-memory keys (8,030
rows).  It deliberately does *not* construct class centers or store steering
values; those belong to Stage 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "use/build/multilayer_11class_trainplus_test_memory"
    / "qwen_multilayer_memory_bank_regenerated_classified_6927.pt"
)
DEFAULT_OUTPUT = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    source_path = args.source_bank.resolve()
    output_dir = args.output_dir.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    fact_vectors = source["fact_vectors"].float().contiguous()
    emotion_cue_vectors = source["emotion_cue_vectors"].float().contiguous()
    records = source["records"]
    if fact_vectors.shape != (len(records), 1024):
        raise ValueError(f"unexpected fact vectors shape: {tuple(fact_vectors.shape)}")
    if emotion_cue_vectors.shape != (len(records), 768):
        raise ValueError(f"unexpected emotion cue vectors shape: {tuple(emotion_cue_vectors.shape)}")
    expected_row_ids = list(range(len(records)))
    observed_row_ids = [record.get("bank_row_id") for record in records]
    if observed_row_ids != expected_row_ids:
        raise ValueError("source records do not have contiguous bank_row_id values")
    for name, vectors in (("fact_vectors", fact_vectors), ("emotion_cue_vectors", emotion_cue_vectors)):
        norms = torch.linalg.vector_norm(vectors, dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=2e-3):
            raise ValueError(f"{name} are not L2-normalized")

    source_counts = Counter(record["memory_source"] for record in records)
    labels = list(source["manifest"]["labels"])
    label_counts = Counter(record["emotion_label"] for record in records)
    if set(label_counts) != set(labels):
        raise ValueError("records and source manifest disagree on label set")

    stage1_records = []
    for record in records:
        stage1_records.append({
            "stage1_row_id": record["bank_row_id"],
            "source_bank_row_id": record["bank_row_id"],
            "memory_id": record["memory_id"],
            "emotion_label": record["emotion_label"],
            "memory_source": record["memory_source"],
            "factual_memory": record["factual_memory"],
            "original_text": record["original_text"],
        })

    artifact = {
        "fact_keys": fact_vectors.to(torch.float16),
        "emotion_cue_values": emotion_cue_vectors.to(torch.float16),
        "records": stage1_records,
        "manifest": {
            "stage": 1,
            "name": "fact_to_emotion_cue",
            "retrieval": "Top-1 cosine similarity over fact_keys",
            "key": {
                "name": "fact_keys",
                "shape": [len(records), 1024],
                "encoder": "Qwen3-Embedding-0.6B",
                "normalized": True,
            },
            "value": {
                "name": "emotion_cue_values",
                "shape": [len(records), 768],
                "encoder": "ModernBERT-GoEmotions layer-22 masked mean",
                "normalized": True,
            },
            "row_count": len(records),
            "memory_source_counts": dict(sorted(source_counts.items())),
            "labels": labels,
            "label_counts": dict(sorted(label_counts.items())),
            "source_bank": str(source_path),
            "source_bank_sha256": sha256(source_path),
            "source_manifest_num_rows": source["manifest"].get("num_rows"),
            "source_actual_num_rows": len(records),
            "source_manifest_row_count_matches_actual": source["manifest"].get("num_rows") == len(records),
            "stage2_status": "not constructed",
        },
    }
    artifact_path = output_dir / "stage1_fact_to_emotion_cue_bank.pt"
    torch.save(artifact, artifact_path)
    write_jsonl(output_dir / "records.jsonl", stage1_records)
    (output_dir / "manifest.json").write_text(
        json.dumps(artifact["manifest"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "built", "artifact": str(artifact_path), **artifact["manifest"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
