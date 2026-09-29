#!/usr/bin/env python3
"""Build the train-only m=2 Stage-2 prototype bank for leakage sensitivity.

This intentionally changes only the member pool used to construct cue keys.
Stage-1 retrieval, labels, clustering, geometric-median construction, and
configured injection values are otherwise pinned to the validated full bank.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
PLUS = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt"
FULL_STAGE2 = ROOT / "use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt"


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
    points = F.normalize(points.float(), p=2, dim=1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = points[torch.randperm(len(points), generator=generator)[:k]].clone()
    previous: torch.Tensor | None = None
    for iteration in range(1, max_iterations + 1):
        assignment = (points @ centers.T).argmax(1)
        if previous is not None and torch.equal(assignment, previous):
            return assignment, centers, iteration
        similarity = points @ centers.T
        updated = []
        for cluster in range(k):
            members = points[assignment == cluster]
            if len(members):
                updated.append(F.normalize(members.mean(0), p=2, dim=0))
            else:
                updated.append(points[int(similarity.max(1).values.argmin())])
        centers = torch.stack(updated)
        previous = assignment
    return assignment, centers, max_iterations


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PLUS / "results/train_only_prototype_sensitivity/train_only_prototypes")
    p.add_argument("--stage1-bank", type=Path, default=STAGE1)
    p.add_argument("--full-stage2-bank", type=Path, default=FULL_STAGE2)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--max-kmeans-iterations", type=int, default=100)
    p.add_argument("--max-weiszfeld-iterations", type=int, default=1000)
    p.add_argument("--tolerance", type=float, default=1e-7)
    a = p.parse_args(); output = a.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    stage1 = torch.load(a.stage1_bank, map_location="cpu", weights_only=False)
    full = torch.load(a.full_stage2_bank, map_location="cpu", weights_only=False)
    cues = stage1["emotion_cue_values"].float(); records = stage1["records"]
    labels = list(stage1["manifest"]["labels"])
    if cues.shape != (8030, 768) or len(records) != 8030:
        raise ValueError("expected the validated 8,030-row Stage-1 bank")
    if full["manifest"].get("prototypes_per_class") != 2 or list(full["manifest"]["labels"]) != labels:
        raise ValueError("full Stage-2 reference must be the validated m=2 bank")
    train_rows = [i for i, r in enumerate(records) if r["memory_source"] == "train_successful_steering"]
    if len(train_rows) != 6927:
        raise ValueError(f"expected 6,927 train rows; found {len(train_rows)}")

    keys: list[torch.Tensor] = []; class_indices: list[int] = []; prototype_records: list[dict[str, Any]] = []
    for class_index, label in enumerate(labels):
        rows = [i for i in train_rows if records[i]["emotion_label"] == label]
        if len(rows) < 2:
            raise ValueError(f"too few train rows for {label}: {len(rows)}")
        assignment, _, kmeans_iterations = spherical_kmeans(cues[rows], 2, a.seed + class_index, a.max_kmeans_iterations)
        row_tensor = torch.tensor(rows, dtype=torch.long)
        for cluster in range(2):
            local = torch.nonzero(assignment == cluster, as_tuple=False).flatten()
            if not len(local):
                raise RuntimeError(f"empty final cluster: {label}/{cluster}")
            center, median_iterations = geometric_median(cues[row_tensor[local]], a.max_weiszfeld_iterations, a.tolerance)
            keys.append(F.normalize(center, p=2, dim=0)); class_indices.append(class_index)
            prototype_records.append({
                "prototype_row_id": len(prototype_records), "emotion_label": label, "class_index": class_index,
                "prototype_index_within_class": cluster, "member_count": int(len(local)),
                "member_source_counts": dict(Counter(records[rows[int(i)]]["memory_source"] for i in local.tolist())),
                "partition_method": "spherical_kmeans", "kmeans_iterations": kmeans_iterations,
                "center_method": "Euclidean geometric median via Weiszfeld", "weiszfeld_iterations": median_iterations,
            })

    manifest = {
        "stage": 2, "name": "emotion_multicenter_geometric_to_injection_train_only_sensitivity",
        "retrieval": "class score is maximum cosine over that class's prototypes",
        "labels": labels, "class_count": len(labels), "prototype_count": len(prototype_records),
        "prototypes_per_class": 2, "key_shape": [len(keys), 768],
        "key_construction": "spherical k-means partitions followed by per-cluster Euclidean geometric median and L2 normalization",
        "member_pool": "train_successful_steering only (6,927 Stage-1 rows)",
        "member_source_counts": {"train_successful_steering": len(train_rows)},
        "test_data_used_for_prototype_keys": False,
        "value_construction": "copied unchanged from validated full-memory m=2 bank; unused in this label-only sensitivity analysis",
        "test_data_used_for_values": bool(full["manifest"].get("test_data_used_for_values", False)),
        "seed": a.seed, "max_kmeans_iterations": a.max_kmeans_iterations,
        "max_weiszfeld_iterations": a.max_weiszfeld_iterations, "tolerance": a.tolerance,
        "reference_full_stage2_bank": str(a.full_stage2_bank.resolve()),
    }
    torch.save({
        "emotion_prototype_keys": torch.stack(keys).float(),
        "prototype_class_indices": torch.tensor(class_indices),
        "configured_injection_values": full["configured_injection_values"],
        "records": prototype_records, "manifest": manifest,
    }, output / "stage2_emotion_multicenter_to_injection_bank.pt")
    write_jsonl(output / "records.jsonl", prototype_records)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "built", "output": str(output), "prototype_count": len(keys), "train_rows": len(train_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
