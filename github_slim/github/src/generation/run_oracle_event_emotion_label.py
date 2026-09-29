#!/usr/bin/env python3
"""P1-1 explicit-label controls with oracle event factual memory.

Produces five Oracle-Event + Normal-Emotion-Label conditions and one shared
Full-Oracle-Label condition. Existing saved normal retrieval decisions are
reused exactly; no retrieval is recomputed.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/phase3_test_evaluation/retrieval_eval"))
from run_qwen_multilayer_11class_retrieval_eval import append_jsonl, load_qwen, read_jsonl, token_inputs

STAGE1 = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt"
META = ROOT / "use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl"
NORMAL = ROOT / "outputs/normal_retrieval_vector_steering_20260923"
METHOD_PATHS = {
    "fact_only": "fact_only/retrieval.jsonl",
    "emotion_only": "emotion_only/retrieval.jsonl",
    "concat_sum": "concat_sum/retrieval.jsonl",
    "similarity_product": "similarity_product/retrieval.jsonl",
    "two_stage_fact_then_cue_prototype": "two_stage_fact_then_cue_prototype/retrieval.jsonl",
}

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    return p.parse_args()

def messages(query, fact, emotion):
    return [{"role": "user", "content": (
        "Retrieved factual memory (use it as context when relevant; do not mention this retrieval process):\n"
        f"{fact}\n\nEmotional state:\n{emotion}\n\nUser query:\n{query}\n\nGenerate a helpful response."
    )}]

@torch.inference_mode()
def generate(tokenizer, model, query, fact, emotion, max_new_tokens):
    device = next(model.parameters()).device
    inputs = token_inputs(tokenizer, messages(query, fact, emotion), device)
    stop = {x for x in (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id) if x is not None}
    output = model(**inputs, use_cache=True)
    cache = output.past_key_values
    next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    ids = []
    while len(ids) < max_new_tokens:
        token = int(next_token.item())
        if token in stop:
            break
        ids.append(token)
        output = model(input_ids=next_token.to(device), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    return tokenizer.decode(ids, skip_special_tokens=True).strip().strip("`").strip()

def validate(rows, name):
    # Retrieval JSONL may repeat baseline/steered rows. Deduplicate only after
    # confirming that every repeated query has the same retrieval decision.
    identity_fields = (
        "memory_id", "retrieved_memory_id", "retrieved_emotion",
        "memory_exact_top1", "retrieval_emotion_match",
    )
    by_query = {}
    for row in rows:
        query_id = int(row["query_row_id"])
        if query_id in by_query:
            previous = by_query[query_id]
            inconsistent = [
                field for field in identity_fields
                if row.get(field) != previous.get(field)
            ]
            if inconsistent:
                raise ValueError(
                    f"{name}: inconsistent duplicate query_row_id={query_id}: "
                    f"{inconsistent}"
                )
        else:
            by_query[query_id] = row
    if set(by_query) != set(range(1103)):
        missing = sorted(set(range(1103)) - set(by_query))
        extra = sorted(set(by_query) - set(range(1103)))
        raise ValueError(
            f"{name}: expected 1,103 unique query IDs; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return [by_query[index] for index in range(1103)]

def main():
    a = parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    meta = validate(read_jsonl(META), "metadata")
    stage1 = torch.load(STAGE1, map_location="cpu", weights_only=False)
    targets = {r["memory_id"]: r for r in stage1["records"] if r["memory_source"] == "test_memory_key_only"}
    if len(targets) != 1103:
        raise ValueError(f"expected 1,103 target memories, got {len(targets)}")
    retrieval = {m: validate(read_jsonl(NORMAL / path), m) for m, path in METHOD_PATHS.items()}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_qwen(torch.device("cuda"))

    for method, rows in retrieval.items():
        out = a.output_dir / "oracle_event_normal_emotion_label" / method
        out.mkdir(parents=True, exist_ok=True)
        path = out / "generations.jsonl"
        done = {x["memory_id"] for x in read_jsonl(path)}
        for number, (m, retrieved) in enumerate(zip(meta, rows), 1):
            target = targets[m["memory_id"]]
            emotion = retrieved["retrieved_emotion"]
            row = {**m, "p1_1_condition": "oracle_event_normal_emotion_label",
                   "retrieval_method": method, "event_source": "oracle_target_memory",
                   "emotion_source": "normal_retrieval",
                   "retrieved_memory_id": retrieved["retrieved_memory_id"],
                   "retrieved_emotion": emotion, "prompt_emotion": emotion,
                   "prompt_memory_id": target["memory_id"],
                   "prompt_factual_memory": target["factual_memory"],
                   "memory_exact_top1": retrieved["memory_exact_top1"],
                   "retrieval_emotion_match": retrieved["retrieval_emotion_match"],
                   "prompt_includes_retrieved_fact": True,
                   "prompt_includes_retrieved_emotion_label": True}
            if m["memory_id"] not in done:
                append_jsonl(path, {**row, "generated_text": generate(tokenizer, model, m["evaluation_query"], target["factual_memory"], emotion, a.max_new_tokens)})
            if number % 10 == 0 or number == 1103:
                print(json.dumps({"condition": row["p1_1_condition"], "method": method, "completed": number, "total": 1103}), flush=True)

    out = a.output_dir / "full_oracle_emotion_label"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "generations.jsonl"
    done = {x["memory_id"] for x in read_jsonl(path)}
    for number, m in enumerate(meta, 1):
        target = targets[m["memory_id"]]
        row = {**m, "p1_1_condition": "full_oracle_emotion_label",
               "retrieval_method": "full_oracle", "event_source": "oracle_target_memory",
               "emotion_source": "oracle", "prompt_emotion": m["target_emotion"],
               "prompt_memory_id": target["memory_id"],
               "prompt_factual_memory": target["factual_memory"],
               "prompt_includes_retrieved_fact": True,
               "prompt_includes_retrieved_emotion_label": True}
        if m["memory_id"] not in done:
            append_jsonl(path, {**row, "generated_text": generate(tokenizer, model, m["evaluation_query"], target["factual_memory"], m["target_emotion"], a.max_new_tokens)})
        if number % 10 == 0 or number == 1103:
            print(json.dumps({"condition": row["p1_1_condition"], "method": "full_oracle", "completed": number, "total": 1103}), flush=True)

    (a.output_dir / "config.json").write_text(json.dumps({
        "protocol": "P1-1 explicit emotion-label prompt",
        "conditions": ["oracle_event_normal_emotion_label", "full_oracle_emotion_label"],
        "retrieval_source": str(NORMAL), "methods": list(METHOD_PATHS),
        "queries": 1103, "max_new_tokens": a.max_new_tokens,
        "decode": "greedy", "prompt_style": "helpful"
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    print(json.dumps({"status": "complete", "output": str(a.output_dir)}))

if __name__ == "__main__":
    main()
