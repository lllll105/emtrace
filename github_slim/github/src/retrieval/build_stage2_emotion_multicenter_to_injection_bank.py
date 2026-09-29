#!/usr/bin/env python3
"""Build a multi-prototype Stage-2 emotion library.

Each emotion class is partitioned with deterministic spherical k-means.  Each
partition is represented by its Euclidean geometric median (Weiszfeld), then
L2-normalized for cosine retrieval.  All prototypes of one class share that
class's single existing configured multilayer injection value.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
STAGE1 = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt"
SOURCE = ROOT / "use/build/multilayer_11class_trainplus_test_memory/qwen_multilayer_memory_bank_regenerated_classified_6927.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prototypes-per-class", type=int, required=True)
    p.add_argument("--stage1-bank", type=Path, default=STAGE1)
    p.add_argument("--source-bank", type=Path, default=SOURCE)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--max-kmeans-iterations", type=int, default=100)
    p.add_argument("--max-weiszfeld-iterations", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=1e-7)
    return p.parse_args()


def geometric_median(points: torch.Tensor, max_iterations: int, tolerance: float) -> tuple[torch.Tensor, int]:
    points = points.double()
    center = points.mean(0)
    for iteration in range(1, max_iterations + 1):
        distance = torch.linalg.vector_norm(points - center, dim=1)
        exact = torch.nonzero(distance <= 1e-12, as_tuple=False)
        if len(exact):
            return points[int(exact[0, 0])].float(), iteration
        weight = distance.reciprocal()
        updated = (points * weight[:, None]).sum(0) / weight.sum()
        if float(torch.linalg.vector_norm(updated - center)) <= tolerance * max(1.0, float(torch.linalg.vector_norm(center))):
            return updated.float(), iteration
        center = updated
    return center.float(), max_iterations


def spherical_kmeans(points: torch.Tensor, k: int, seed: int, max_iterations: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return per-point assignments and unit spherical k-means centers."""
    if not 1 <= k <= len(points):
        raise ValueError(f"invalid k={k} for {len(points)} points")
    points = F.normalize(points.float(), p=2, dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = points[torch.randperm(len(points), generator=generator)[:k]].clone()
    previous: torch.Tensor | None = None
    for iteration in range(1, max_iterations + 1):
        assignment = (points @ centers.T).argmax(1)
        if previous is not None and torch.equal(assignment, previous):
            return assignment, centers, iteration
        updated = []
        similarity = points @ centers.T
        for cluster in range(k):
            members = points[assignment == cluster]
            if len(members):
                updated.append(F.normalize(members.mean(0), p=2, dim=0))
            else:
                # Deterministically seed an empty cluster with the least well
                # represented point under the previous centers.
                index = int(similarity.max(1).values.argmin())
                updated.append(points[index])
        centers = torch.stack(updated)
        previous = assignment
    return assignment, centers, max_iterations


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    a = parse_args()
    if a.prototypes_per_class < 2:
        raise ValueError("this builder is for multi-prototype experiments; use at least 2")
    output = (a.output_dir or ROOT / f"use_2rag/build/stage2_emotion_multicenter_geometric_m{a.prototypes_per_class}").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    s1 = torch.load(a.stage1_bank, map_location="cpu", weights_only=False)
    source = torch.load(a.source_bank, map_location="cpu", weights_only=False)
    cues, records, labels = s1["emotion_cue_values"].float(), s1["records"], list(s1["manifest"]["labels"])
    if cues.shape != (8030, 768) or len(records) != 8030:
        raise ValueError("Stage-1 must contain all 8,030 768-D cue values")
    injection = source["injection_values"]
    if injection["test_data_used"] or list(injection["labels"]) != labels:
        raise ValueError("unexpected injection-value provenance")

    prototype_keys, prototype_labels, prototype_records = [], [], []
    for class_index, label in enumerate(labels):
        rows = [i for i, r in enumerate(records) if r["emotion_label"] == label]
        assignment, _, kmeans_iterations = spherical_kmeans(cues[rows], a.prototypes_per_class, a.seed + class_index, a.max_kmeans_iterations)
        for cluster in range(a.prototypes_per_class):
            local = torch.nonzero(assignment == cluster, as_tuple=False).flatten()
            if not len(local):
                raise RuntimeError(f"empty final cluster: {label}/{cluster}")
            center, median_iterations = geometric_median(cues[torch.tensor(rows)[local]], a.max_weiszfeld_iterations, a.tolerance)
            prototype_keys.append(F.normalize(center, p=2, dim=0))
            prototype_labels.append(class_index)
            prototype_records.append({
                "prototype_row_id": len(prototype_records), "emotion_label": label, "class_index": class_index,
                "prototype_index_within_class": cluster, "member_count": int(len(local)),
                "member_source_counts": dict(sorted(Counter(records[rows[int(i)]]["memory_source"] for i in local.tolist()).items())),
                "partition_method": "spherical_kmeans", "kmeans_iterations": kmeans_iterations,
                "center_method": "Euclidean geometric median via Weiszfeld", "weiszfeld_iterations": median_iterations,
            })
    keys = torch.stack(prototype_keys).float()
    values = {label: {"layers": injection["values"][label]["layers"], "layer_weights": injection["values"][label]["layer_weights"], "configured_layer_vectors": injection["values"][label]["configured_layer_vectors"].float().contiguous()} for label in labels}
    manifest = {
        "stage": 2, "name": "emotion_multicenter_geometric_to_injection",
        "retrieval": "class score is maximum cosine over that class's prototypes",
        "labels": labels, "class_count": len(labels), "prototype_count": len(prototype_records),
        "prototypes_per_class": a.prototypes_per_class, "key_shape": list(keys.shape),
        "key_construction": "spherical k-means partitions followed by per-cluster Euclidean geometric median and L2 normalization",
        "member_pool": "all 8,030 Stage-1 rows", "stage1_memory_source_counts": dict(sorted(Counter(r["memory_source"] for r in records).items())),
        "value_construction": "existing training-only configured_layer_vectors copied without modification", "test_data_used_for_values": False,
        "seed": a.seed,
    }
    torch.save({"emotion_prototype_keys": keys, "prototype_class_indices": torch.tensor(prototype_labels), "configured_injection_values": values, "records": prototype_records, "manifest": manifest}, output / "stage2_emotion_multicenter_to_injection_bank.pt")
    write_jsonl(output / "records.jsonl", prototype_records)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "built", "output": str(output), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
