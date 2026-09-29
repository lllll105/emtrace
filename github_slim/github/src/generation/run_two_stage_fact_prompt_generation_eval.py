#!/usr/bin/env python3
"""Generate and score the 11-class test set with two-stage retrieval.

Protocol (strict / non-oracle): saved Qwen3 embeddings of ``evaluation_query``
search all 8,030 Stage-1 fact keys.  The Top-1 record's factual memory is added
to the generation prompt.  Its cue then selects a class by maximum cosine over
the two geometric-median cue prototypes for that class.  The selected class's
preconfigured multilayer vectors are added directly to the Qwen residual stream.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "code" / "phase3_test_evaluation" / "retrieval_eval"
sys.path.insert(0, str(LEGACY))
from run_qwen_multilayer_11class_retrieval_eval import (  # noqa: E402
    MultilayerSteering, append_jsonl, configured_vectors, load_classifier,
    load_qwen, read_jsonl, token_inputs, write_jsonl,
)

STAGE1 = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt"
STAGE2 = ROOT / "use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt"
QUERY_VECTORS = ROOT / "use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt"
QUERY_METADATA = ROOT / "use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl"
OUTPUT = ROOT / "use_2rag/evaluation/generation_two_stage_fact_prompt_m2_20260922"
CONDITIONS = ("bare_query", "retrieved_fact_prompt", "retrieved_fact_prompt_plus_stage2_steering")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--gpu", default="0")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--classifier-batch-size", type=int, default=64)
    p.add_argument("--stage1-bank", type=Path, default=STAGE1)
    p.add_argument("--stage2-bank", type=Path, default=STAGE2)
    p.add_argument("--query-vectors", type=Path, default=QUERY_VECTORS)
    p.add_argument("--query-metadata", type=Path, default=QUERY_METADATA)
    return p.parse_args()


def prompt_messages(query: str, retrieved_fact: str | None) -> list[dict[str, str]]:
    if retrieved_fact is None:
        content = f"User query:\n{query}\n\nGenerate a helpful response."
    else:
        content = (
            "Retrieved factual memory (use it as context when relevant; do not "
            "mention this retrieval process):\n"
            f"{retrieved_fact}\n\nUser query:\n{query}\n\nGenerate a helpful response."
        )
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def generate(tokenizer: Any, model: Any, controller: MultilayerSteering, query: str,
             retrieved_fact: str | None, vectors: dict[int, torch.Tensor] | None,
             max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = token_inputs(tokenizer, prompt_messages(query, retrieved_fact), device)
    stop_ids = {x for x in (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id) if x is not None}
    with controller.use(vectors):
        output = model(**inputs, use_cache=True)
    cache, next_token = output.past_key_values, torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
    result: list[int] = []
    while len(result) < max_new_tokens:
        token = int(next_token.item())
        if token in stop_ids:
            break
        result.append(token)
        with controller.use(vectors):
            output = model(input_ids=next_token.to(device), past_key_values=cache, use_cache=True)
        cache, next_token = output.past_key_values, torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
    return tokenizer.decode(result, skip_special_tokens=True).strip().strip("`").strip()


def prepare_retrieval(a: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stage1 = torch.load(a.stage1_bank, map_location="cpu", weights_only=False)
    stage2 = torch.load(a.stage2_bank, map_location="cpu", weights_only=False)
    query = torch.load(a.query_vectors, map_location="cpu", weights_only=True).float()
    metadata = read_jsonl(a.query_metadata)
    labels = list(stage2["manifest"]["labels"])
    if len(metadata) != 1103 or query.shape != (1103, 1024):
        raise ValueError(f"expected saved strict query input [1103,1024], got {query.shape} / {len(metadata)} metadata")
    if len(stage1["records"]) != 8030 or stage1["fact_keys"].shape != (8030, 1024):
        raise ValueError("Stage-1 bank must contain all 8,030 fact keys")
    if stage2["manifest"].get("prototypes_per_class") != 2:
        raise ValueError("this runner is pinned to the validated two-prototype Stage-2 bank")
    fact_score, stage1_index = (query @ stage1["fact_keys"].float().T).max(dim=1)
    cue = stage1["emotion_cue_values"].float()[stage1_index]
    prototype_score = cue @ stage2["emotion_prototype_keys"].float().T
    class_score = torch.full((len(metadata), len(labels)), float("-inf"))
    for prototype_id, class_id in enumerate(stage2["prototype_class_indices"].tolist()):
        class_score[:, class_id] = torch.maximum(class_score[:, class_id], prototype_score[:, prototype_id])
    ranked_score, ranked_class = class_score.topk(len(labels), dim=1)
    all_rows: list[dict[str, Any]] = []
    for row_id, (meta, record_id, score, ranks, scores) in enumerate(zip(metadata, stage1_index.tolist(), fact_score.tolist(), ranked_class.tolist(), ranked_score.tolist())):
        retrieved = stage1["records"][record_id]
        ranked_labels = [labels[index] for index in ranks]
        all_rows.append({
            "query_row_id": row_id, "memory_id": meta["memory_id"], "target_emotion": meta["target_emotion"],
            "evaluation_query": meta["evaluation_query"], "stage1_top1_memory_id": retrieved["memory_id"],
            "stage1_top1_memory_source": retrieved["memory_source"], "stage1_top1_retrieved_emotion": retrieved["emotion_label"],
            "stage1_retrieved_factual_memory": retrieved["factual_memory"], "stage1_fact_cosine_similarity": float(score),
            "stage1_retrieved_emotion_match": int(retrieved["emotion_label"] == meta["target_emotion"]),
            "stage2_predicted_emotion": ranked_labels[0], "stage2_ranked_emotions": ranked_labels,
            "stage2_ranked_class_scores": [float(x) for x in scores],
            "stage2_emotion_match": int(ranked_labels[0] == meta["target_emotion"]),
        })
    if not 0 <= a.shard_index < a.shard_count:
        raise ValueError("invalid shard selection")
    config = {"stage1_bank": str(a.stage1_bank), "stage2_bank": str(a.stage2_bank),
              "query_vectors": str(a.query_vectors), "query_vector_source": "Qwen3-Embedding-0.6B evaluation_query vectors saved before generation",
              "stage1_protocol": "all 8030 normalized fact keys, Top-1 cosine; retrieved factual_memory is prompt context",
              "stage2_protocol": "retrieved Stage-1 cue -> max cosine over 2 geometric-median prototypes per class",
              "injection_protocol": "existing configured_layer_vectors directly added at configured residual layers; no second alpha", "conditions": list(CONDITIONS),
              "total_queries": len(all_rows), "shard_index": a.shard_index, "shard_count": a.shard_count, "stage2_manifest": stage2["manifest"]}
    return all_rows[a.shard_index::a.shard_count], {"config": config, "stage2": stage2}


def score(rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    tokenizer, model, labels = load_classifier(device)
    for start in range(0, len(rows), batch_size):
        block = rows[start:start + batch_size]
        encoded = tokenizer([x["generated_text"] for x in block], padding=True, truncation=True, max_length=512, return_tensors="pt")
        probabilities = torch.sigmoid(model(**{k: v.to(device) for k, v in encoded.items()}).logits).cpu()
        for row, p in zip(block, probabilities):
            target = labels.index(row["target_emotion"]); predicted = int(p.argmax())
            row.update(modernbert_predicted_label=labels[predicted], modernbert_top_score=float(p[predicted]),
                       target_emotion_score=float(p[target]), emotion_accuracy=int(labels[predicted] == row["target_emotion"]))
    del model
    torch.cuda.empty_cache()
    return rows


def summarize(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    baseline = {x["memory_id"]: x for x in rows if x["condition"] == "bare_query"}
    for row in rows:
        row["emotion_gain_vs_bare_query"] = row["target_emotion_score"] - baseline[row["memory_id"]]["target_emotion_score"]
    write_jsonl(run_dir / "scored_results.jsonl", rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[(row["condition"], row["target_emotion"])].append(row)
    report = []
    for condition in CONDITIONS:
        for emotion in ["all", *sorted({x["target_emotion"] for x in rows})]:
            group = [x for x in rows if x["condition"] == condition and (emotion == "all" or x["target_emotion"] == emotion)]
            if group:
                report.append({"scope": "micro" if emotion == "all" else "per_emotion", "condition": condition, "emotion": emotion, "n": len(group),
                               "emotion_accuracy": sum(x["emotion_accuracy"] for x in group) / len(group),
                               "mean_target_emotion_score": sum(x["target_emotion_score"] for x in group) / len(group),
                               "mean_emotion_gain_vs_bare_query": sum(x["emotion_gain_vs_bare_query"] for x in group) / len(group),
                               "stage1_emotion_match_at1": sum(x["stage1_retrieved_emotion_match"] for x in group) / len(group),
                               "stage2_emotion_match_at1": sum(x["stage2_emotion_match"] for x in group) / len(group)})
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0])); w.writeheader(); w.writerows(report)


def main() -> None:
    a = parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    rows, assets = prepare_retrieval(a)
    run_dir = a.output_dir / f"shard_{a.shard_index:02d}_of_{a.shard_count:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(assets["config"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(run_dir / "retrieval.jsonl", rows)
    print(json.dumps({"status": "retrieval_prepared", "queries": len(rows), "run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
    tokenizer, model = load_qwen(torch.device("cuda"))
    # The validated bank stores emotion entries directly; the legacy helper
    # expects them under a ``values`` wrapper.
    values = {"values": assets["stage2"]["configured_injection_values"]}
    layers = {layer for item in values["values"].values() for layer in item["layers"]}
    controller = MultilayerSteering(model, layers)
    for condition in CONDITIONS:
        path = run_dir / f"{condition}_generations.jsonl"; existing = {x["memory_id"] for x in read_jsonl(path)}
        for number, row in enumerate(rows, start=1):
            if row["memory_id"] not in existing:
                fact = row["stage1_retrieved_factual_memory"] if condition != "bare_query" else None
                emotion = row["stage2_predicted_emotion"]
                vectors = configured_vectors(values, emotion) if condition == "retrieved_fact_prompt_plus_stage2_steering" else None
                item = values["values"][emotion] if vectors is not None else None
                append_jsonl(path, {**row, "condition": condition, "prompt_includes_retrieved_fact": fact is not None,
                                    "injected_emotion": emotion if vectors is not None else None,
                                    "injected_layers": item["layers"] if item else [], "layer_weights": item["layer_weights"] if item else {},
                                    "generated_text": generate(tokenizer, model, controller, row["evaluation_query"], fact, vectors, a.max_new_tokens)})
            if number % 10 == 0 or number == len(rows): print(json.dumps({"condition": condition, "completed": number, "total": len(rows)}, ensure_ascii=False), flush=True)
    controller.close(); del model; torch.cuda.empty_cache()
    all_generated = [x for c in CONDITIONS for x in read_jsonl(run_dir / f"{c}_generations.jsonl")]
    if len(all_generated) != len(rows) * len(CONDITIONS): raise RuntimeError("incomplete generation files")
    scored_path = run_dir / "scored_results.jsonl"; scored = read_jsonl(scored_path)
    if len(scored) != len(all_generated): scored = score(all_generated, torch.device("cuda"), a.classifier_batch_size)
    summarize(run_dir, scored)
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "generated": len(scored)}, ensure_ascii=False))


if __name__ == "__main__": main()
