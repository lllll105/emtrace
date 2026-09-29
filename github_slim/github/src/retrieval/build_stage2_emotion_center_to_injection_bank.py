#!/usr/bin/env python3
"""Build Stage-2 of the two-stage RAG memory system.

Stage 2 contains exactly one record per supported emotion class:

    key   : L2-normalized geometric median of that class's Stage-1 emotion cues
    value : the existing, preweighted multilayer steering vectors for that class

All 8,030 Stage-1 rows (6,927 training and 1,103 test-memory keys) contribute
to geometric-center estimation.  Steering values are copied from the existing
training-only injection-value artifact and are never recomputed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1 = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt"
DEFAULT_SOURCE_BANK = (
    ROOT
    / "use/build/multilayer_11class_trainplus_test_memory"
    / "qwen_multilayer_memory_bank_regenerated_classified_6927.pt"
)
DEFAULT_OUTPUTS = {
    "geometric_median": ROOT / "use_2rag/build/stage2_emotion_geometric_centers",
    "arithmetic_mean": ROOT / "use_2rag/build/stage2_emotion_arithmetic_means",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-bank", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--source-bank", type=Path, default=DEFAULT_SOURCE_BANK)
    parser.add_argument("--center-method", choices=("geometric_median", "arithmetic_mean"), default="geometric_median")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-iterations", type=int, default=1_000)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geometric_median(points: torch.Tensor, max_iterations: int, tolerance: float, epsilon: float) -> tuple[torch.Tensor, int, bool, float]:
    """Return the Euclidean geometric median via Weiszfeld iterations."""
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("points must be a non-empty [n, dimension] tensor")
    points = points.to(dtype=torch.float64)
    center = points.mean(dim=0)
    for iteration in range(1, max_iterations + 1):
        distances = torch.linalg.vector_norm(points - center, dim=1)
        exact = torch.nonzero(distances <= epsilon, as_tuple=False)
        if len(exact):
            center = points[int(exact[0, 0])]
            return center.float(), iteration, True, float(distances.sum())
        weights = distances.reciprocal()
        updated = (points * weights[:, None]).sum(dim=0) / weights.sum()
        step = torch.linalg.vector_norm(updated - center)
        center = updated
        if float(step) <= tolerance * max(1.0, float(torch.linalg.vector_norm(center))):
            objective = torch.linalg.vector_norm(points - center, dim=1).sum()
            return center.float(), iteration, True, float(objective)
    objective = torch.linalg.vector_norm(points - center, dim=1).sum()
    return center.float(), max_iterations, False, float(objective)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    stage1_path, source_path = args.stage1_bank.resolve(), args.source_bank.resolve()
    output_dir = (args.output_dir or DEFAULT_OUTPUTS[args.center_method]).resolve()
    if not stage1_path.is_file() or not source_path.is_file():
        raise FileNotFoundError(f"missing input: stage1={stage1_path.is_file()} source={source_path.is_file()}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
    cues = stage1["emotion_cue_values"].float()
    records = stage1["records"]
    labels = list(stage1["manifest"]["labels"])
    if cues.shape != (len(records), 768) or len(records) != 8030:
        raise ValueError(f"expected 8,030 Stage-1 rows with 768-D cues, got {tuple(cues.shape)} and {len(records)} records")
    if not torch.allclose(torch.linalg.vector_norm(cues, dim=1), torch.ones(len(cues)), atol=2e-3):
        raise ValueError("Stage-1 emotion cues are not L2-normalized")
    row_labels = [record["emotion_label"] for record in records]
    if set(row_labels) != set(labels):
        raise ValueError("Stage-1 records and manifest disagree on labels")

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    injection_values = source["injection_values"]
    if injection_values.get("test_data_used") is not False:
        raise ValueError("steering values must remain training-only")
    if list(injection_values["labels"]) != labels:
        raise ValueError("injection-value labels do not match Stage-1 labels")

    centers: list[torch.Tensor] = []
    stage2_records: list[dict[str, Any]] = []
    all_counts = Counter(row_labels)
    source_counts = Counter(record["memory_source"] for record in records)
    for class_id, label in enumerate(labels):
        indices = [index for index, row_label in enumerate(row_labels) if row_label == label]
        if args.center_method == "geometric_median":
            center, iterations, converged, objective = geometric_median(cues[indices], args.max_iterations, args.tolerance, args.epsilon)
            construction = "per-class Euclidean geometric median via Weiszfeld, then L2 normalization"
        else:
            center = cues[indices].mean(dim=0)
            iterations, converged = 0, True
            objective = float(torch.linalg.vector_norm(cues[indices] - center, dim=1).sum())
            construction = "per-class arithmetic mean, then L2 normalization"
        normalized_center = F.normalize(center, p=2, dim=0)
        value = injection_values["values"][label]
        vectors = value["configured_layer_vectors"].float().contiguous()
        layers = [int(layer) for layer in value["layers"]]
        if vectors.ndim != 2 or vectors.shape[0] != len(layers):
            raise ValueError(f"invalid configured vectors for {label}")
        centers.append(normalized_center)
        stage2_records.append({
            "stage2_row_id": class_id,
            "emotion_label": label,
            "center_method": args.center_method,
            "center_construction": construction,
            "member_count": len(indices),
            "member_source_counts": dict(sorted(Counter(records[index]["memory_source"] for index in indices).items())),
            "weiszfeld_iterations": iterations,
            "converged": converged,
            "sum_euclidean_distances": objective,
            "layers": layers,
            "layer_weights": {str(layer): float(weight) for layer, weight in value["layer_weights"].items()},
            "configured_vector_shape": list(vectors.shape),
            "steering_value_provenance": "existing training-only configured_layer_vectors",
        })

    center_keys = torch.stack(centers).to(torch.float32).contiguous()
    if not torch.allclose(torch.linalg.vector_norm(center_keys, dim=1), torch.ones(len(labels)), atol=1e-6):
        raise ValueError("failed to normalize Stage-2 geometric centers")
    stage2_values = {
        record["emotion_label"]: {
            "layers": record["layers"],
            "layer_weights": injection_values["values"][record["emotion_label"]]["layer_weights"],
            "configured_layer_vectors": injection_values["values"][record["emotion_label"]]["configured_layer_vectors"].float().contiguous(),
        }
        for record in stage2_records
    }
    manifest = {
        "stage": 2,
        "name": f"emotion_{args.center_method}_to_injection",
        "retrieval": "Top-1 cosine similarity over 11 class-center keys",
        "key": {
            "name": "emotion_class_center_keys",
            "shape": [len(labels), 768],
            "construction": construction,
            "member_pool": "all 8,030 Stage-1 rows",
            "normalized": True,
        },
        "value": {
            "name": "configured_multilayer_injection_values",
            "construction": "copied without modification from existing training-only injection_values",
            "test_data_used_for_values": False,
        },
        "row_count": len(labels),
        "labels": labels,
        "stage1_row_count": len(records),
        "stage1_label_counts": dict(sorted(all_counts.items())),
        "stage1_memory_source_counts": dict(sorted(source_counts.items())),
        "stage1_bank": str(stage1_path),
        "stage1_bank_sha256": sha256(stage1_path),
        "source_bank": str(source_path),
        "source_bank_sha256": sha256(source_path),
        "weiszfeld": {"max_iterations": args.max_iterations, "tolerance": args.tolerance, "epsilon": args.epsilon},
    }
    artifact = {"emotion_class_center_keys": center_keys, "configured_injection_values": stage2_values, "records": stage2_records, "manifest": manifest}
    artifact_path = output_dir / "stage2_emotion_center_to_injection_bank.pt"
    torch.save(artifact, artifact_path)
    write_jsonl(output_dir / "records.jsonl", stage2_records)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "built", "artifact": str(artifact_path), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
