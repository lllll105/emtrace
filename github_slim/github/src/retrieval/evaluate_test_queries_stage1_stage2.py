#!/usr/bin/env python3
"""Evaluate strict two-stage retrieval on held-out ``evaluation_query`` text.

No Qwen generation model or steering hook is loaded.  Each eligible test
question is embedded by Qwen3-Embedding-0.6B, retrieves one of the 8,030
Stage-1 fact keys, then retrieves a class from the 11 Stage-2 geometric
centers using the retrieved row's stored emotion-cue value.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
QWEN_EMBED = ROOT / "model/Qwen3-Embedding-0.6B"
QUERIES = ROOT / "data/test_create_bygpt/goemotions_test_queries_by_gpt.jsonl"
STAGE1 = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt"
STAGE2 = ROOT / "use_2rag/build/stage2_emotion_geometric_centers/stage2_emotion_center_to_injection_bank.pt"
OUTPUT = ROOT / "use_2rag/evaluation/test_query_qwen3_stage1_stage2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--stage1-bank", type=Path, default=STAGE1)
    parser.add_argument("--stage2-bank", type=Path, default=STAGE2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--query-fact-vectors", type=Path, default=None, help="reuse saved Qwen3 query vectors instead of re-embedding")
    parser.add_argument("--query-metadata", type=Path, default=None, help="metadata paired with --query-fact-vectors")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool(torch.all(attention_mask[:, -1])):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), sequence_lengths]


def embed_questions(texts: list[str], device: torch.device, batch_size: int, max_length: int) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(QWEN_EMBED, padding_side="left", local_files_only=True)
    model = AutoModel.from_pretrained(QWEN_EMBED, torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            embedding = F.normalize(last_token_pool(model(**batch).last_hidden_state, batch["attention_mask"]), p=2, dim=1)
            chunks.append(embedding.to(dtype=torch.float16).cpu())
            completed = min(start + batch_size, len(texts))
            if completed % (batch_size * 10) == 0 or completed == len(texts):
                print(f"embedded {completed}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return torch.cat(chunks).contiguous()


def metric(records: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    n = len(records)
    result = {
        "n": n,
        "stage1_retrieved_emotion_match_at1": sum(row["stage1_retrieved_emotion_match"] for row in records) / n,
    }
    for k in (1, 3, 5):
        result[f"stage2_emotion_accuracy_at{k}"] = sum(row["target_emotion"] in row["stage2_ranked_emotions"][:k] for row in records) / n
    result["labels"] = labels
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    stage1 = torch.load(args.stage1_bank, map_location="cpu", weights_only=False)
    stage2 = torch.load(args.stage2_bank, map_location="cpu", weights_only=False)
    labels = list(stage2["manifest"]["labels"])
    if list(stage1["manifest"]["labels"]) != labels:
        raise ValueError("Stage-1 and Stage-2 label order differs")
    fact_keys = stage1["fact_keys"].float()
    cue_values = stage1["emotion_cue_values"].float()
    center_keys = stage2.get("emotion_class_center_keys", stage2.get("emotion_geometric_center_keys"))
    if center_keys is None:
        raise KeyError("Stage-2 bank has no class-center key tensor")
    center_keys = center_keys.float()
    if fact_keys.shape != (8030, 1024) or cue_values.shape != (8030, 768) or center_keys.shape != (11, 768):
        raise ValueError("unexpected two-stage bank shapes")

    if (args.query_fact_vectors is None) != (args.query_metadata is None):
        raise ValueError("--query-fact-vectors and --query-metadata must be supplied together")
    if args.query_fact_vectors is not None:
        rows = [
            {"memory_id": row["memory_id"], "emotion_label": row["target_emotion"], "evaluation_query": row["evaluation_query"]}
            for row in read_jsonl(args.query_metadata)
        ]
        query_vectors = torch.load(args.query_fact_vectors, map_location="cpu", weights_only=True).float()
    else:
        rows = [
            row for row in read_jsonl(args.queries)
            if row.get("query_generation_status") == "ok" and row.get("emotion_label") in labels and isinstance(row.get("evaluation_query"), str) and row["evaluation_query"].strip()
        ]
    if len(rows) != 1103:
        raise ValueError(f"expected 1,103 eligible 11-class questions, got {len(rows)}")
    if len({row["memory_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate memory_id in selected queries")

    if args.query_fact_vectors is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for strict Qwen3 query embedding")
        query_vectors = embed_questions([row["evaluation_query"] for row in rows], device, args.batch_size, args.max_length)
    if query_vectors.shape != (len(rows), 1024):
        raise ValueError(f"unexpected query vector shape: {tuple(query_vectors.shape)}")
    torch.save(query_vectors.to(torch.float16), output_dir / "query_fact_vectors_qwen3_fp16.pt")
    write_jsonl(output_dir / "query_metadata.jsonl", [
        {"query_row_id": index, "memory_id": row["memory_id"], "target_emotion": row["emotion_label"], "evaluation_query": row["evaluation_query"]}
        for index, row in enumerate(rows)
    ])

    fact_scores = query_vectors.float() @ fact_keys.T
    stage1_scores, stage1_indices = fact_scores.max(dim=1)
    retrieved_cues = cue_values[stage1_indices]
    stage2_scores = retrieved_cues @ center_keys.T
    ranked_scores, ranked_indices = torch.topk(stage2_scores, k=len(labels), dim=1)
    results: list[dict[str, Any]] = []
    for index, (query, stage1_index, stage1_score, class_indices, class_scores) in enumerate(zip(rows, stage1_indices.tolist(), stage1_scores.tolist(), ranked_indices.tolist(), ranked_scores.tolist())):
        retrieved = stage1["records"][stage1_index]
        ranked_emotions = [labels[class_index] for class_index in class_indices]
        results.append({
            "query_row_id": index,
            "memory_id": query["memory_id"],
            "target_emotion": query["emotion_label"],
            "evaluation_query": query["evaluation_query"],
            "stage1_top1_row_id": stage1_index,
            "stage1_top1_memory_id": retrieved["memory_id"],
            "stage1_top1_memory_source": retrieved["memory_source"],
            "stage1_top1_retrieved_emotion": retrieved["emotion_label"],
            "stage1_fact_cosine_similarity": stage1_score,
            "stage1_retrieved_emotion_match": int(retrieved["emotion_label"] == query["emotion_label"]),
            "stage2_predicted_emotion": ranked_emotions[0],
            "stage2_ranked_emotions": ranked_emotions,
            "stage2_ranked_cosine_scores": class_scores,
        })
    write_jsonl(output_dir / "per_query_retrieval.jsonl", results)

    summary = metric(results, labels)
    summary.update({
        "protocol": "evaluation_query Qwen3 fact embedding -> Stage-1 Top-1 over 8,030 fact keys -> retrieved cue -> Stage-2 Top-1 over 11 class centers",
        "stage2_center_construction": stage2["manifest"]["key"]["construction"],
        "query_count": len(rows),
        "stage1_candidate_count": len(stage1["records"]),
        "stage2_candidate_count": len(labels),
        "query_encoder": "Qwen3-Embedding-0.6B last-token pool, L2 normalized",
        "stage1_similarity": "cosine",
        "stage2_similarity": "cosine",
        "no_generation_or_steering": True,
        "query_vector_source": "reused saved Qwen3 query vectors" if args.query_fact_vectors is not None else "encoded in this run by Qwen3-Embedding-0.6B",
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    per_emotion: list[dict[str, Any]] = []
    for label in labels:
        group = [row for row in results if row["target_emotion"] == label]
        values = metric(group, labels)
        per_emotion.append({"emotion": label, **{key: value for key, value in values.items() if key != "labels"}})
    with (output_dir / "per_emotion.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_emotion[0]))
        writer.writeheader()
        writer.writerows(per_emotion)

    confusion = Counter((row["target_emotion"], row["stage2_predicted_emotion"]) for row in results)
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_emotion", *labels])
        writer.writeheader()
        for target in labels:
            writer.writerow({"target_emotion": target, **{predicted: confusion[(target, predicted)] for predicted in labels}})
    print(json.dumps({"status": "complete", "output": str(output_dir), **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
