#!/usr/bin/env python3
"""Evaluate the regenerated 11-class multilayer memory bank on held-out questions.

    The retrieval bank contains 6,927 accepted training memories plus 1,103 test
    memory keys.  A test key's fact vector is the Qwen3 embedding of its factual
    memory, and its emotion-cue vector is the ModernBERT layer-22 embedding of its
    original text.  At evaluation time, the test question is independently
    encoded by Qwen3 and ModernBERT, then the full 8,030-row bank is searched.

Steering vectors are read from ``configured_layer_vectors``.  Those vectors
already include their per-layer alpha weights, so the forward hooks add them
directly and must never apply a second global alpha multiplier.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[3]
COMMON = ROOT / "code" / "common"
sys.path.insert(0, str(COMMON))
from goemotions import GOEMOTIONS_LABELS  # noqa: E402


QWEN = "./local/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"
QWEN_EMBED = ROOT / "model" / "Qwen3-Embedding-0.6B"
MODERNBERT = "./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487"
BUILD = ROOT / "use/build/multilayer_11class_trainplus_test_memory"
BANK = BUILD / "qwen_multilayer_memory_bank_regenerated_classified_6927.pt"
QUERIES = ROOT / "data/test_create_bygpt/goemotions_test_queries_by_gpt.jsonl"
OUTPUT = ROOT / "outputs/retrieval_eval/multilayer_11class_fullbank_20260918"
METHODS = ("dual_concat", "emotion_only", "emotion_first_fact_rerank_lambda_0p1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-embed-batch-size", type=int, default=32)
    parser.add_argument("--retrieval-max-length", type=int, default=512)
    parser.add_argument("--rerank-candidate-k", type=int, default=50)
    parser.add_argument("--rerank-fact-weight", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def load_qwen(device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True)
    model.to(device).eval()
    return tokenizer, model


def load_classifier(device: torch.device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODERNBERT, local_files_only=True)
    except ValueError:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(Path(MODERNBERT) / "tokenizer.json"), unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]")
    model = AutoModelForSequenceClassification.from_pretrained(MODERNBERT, local_files_only=True).to(device).eval()
    labels = [str(model.config.id2label.get(index, f"label_{index}")).lower() for index in range(model.config.num_labels)]
    if all(label.startswith("label_") for label in labels):
        labels = GOEMOTIONS_LABELS[: model.config.num_labels]
    return tokenizer, model, labels


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if bool(torch.all(attention_mask[:, -1])):
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), sequence_lengths]


def encode_retrieval_questions(texts: list[str], device: torch.device, batch_size: int, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy helper retained for ad hoc question-only retrieval checks."""
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_EMBED, padding_side="left", local_files_only=True)
    qwen_model = AutoModel.from_pretrained(QWEN_EMBED, torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    fact_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = qwen_tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            fact_chunks.append(F.normalize(last_token_pool(qwen_model(**encoded).last_hidden_state, encoded["attention_mask"]), p=2, dim=1).float().cpu())
    del qwen_model
    torch.cuda.empty_cache()

    try:
        cue_tokenizer = AutoTokenizer.from_pretrained(MODERNBERT, local_files_only=True)
    except ValueError:
        cue_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(Path(MODERNBERT) / "tokenizer.json"), unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]")
    cue_model = AutoModel.from_pretrained(MODERNBERT, local_files_only=True).to(device).eval()
    cue_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = cue_tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
            hidden = cue_model(**encoded, output_hidden_states=True, return_dict=True).hidden_states[22]
            mask = encoded["attention_mask"].to(dtype=hidden.dtype).unsqueeze(-1)
            cue_chunks.append(F.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0), p=2, dim=1).float().cpu())
    del cue_model
    torch.cuda.empty_cache()
    return torch.cat(fact_chunks), torch.cat(cue_chunks)


def prompt_messages(query: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"User query:\n{query}\n\nGenerate a response."}]


def token_inputs(tokenizer: Any, message_list: list[dict[str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    payload = tokenizer.apply_chat_template(message_list, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    if isinstance(payload, torch.Tensor):
        return {"input_ids": payload.to(device)}
    return {key: value.to(device) for key, value in payload.items() if torch.is_tensor(value)}


@dataclass
class SteeringState:
    vectors: dict[int, torch.Tensor] | None = None


class MultilayerSteering:
    """Add preweighted vectors at their configured one-based transformer layers."""

    def __init__(self, model: Any, layers: set[int]):
        self.state = SteeringState()
        self.handles = [model.model.layers[layer - 1].register_forward_hook(self._hook(layer)) for layer in sorted(layers)]

    def _hook(self, layer: int):
        def apply(_module: Any, _inputs: Any, output: Any):
            if self.state.vectors is None or layer not in self.state.vectors:
                return output
            hidden, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, None)
            changed = hidden.clone()
            changed[:, -1, :] += self.state.vectors[layer].to(device=changed.device, dtype=changed.dtype)
            return changed if rest is None else (changed,) + rest
        return apply

    @contextmanager
    def use(self, vectors: dict[int, torch.Tensor] | None) -> Iterator[None]:
        self.state = SteeringState(vectors=vectors)
        try:
            yield
        finally:
            self.state = SteeringState()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def generate(tokenizer: Any, model: Any, controller: MultilayerSteering, query: str, vectors: dict[int, torch.Tensor] | None, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = token_inputs(tokenizer, prompt_messages(query), device)
    stop_ids = {token for token in (tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id) if token is not None}
    with controller.use(vectors):
        outputs = model(**inputs, use_cache=True)
    cache, next_token = outputs.past_key_values, torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated: list[int] = []
    while len(generated) < max_new_tokens:
        token = int(next_token.item())
        if token in stop_ids:
            break
        generated.append(token)
        with controller.use(vectors):
            outputs = model(input_ids=next_token.to(device), past_key_values=cache, use_cache=True)
        cache, next_token = outputs.past_key_values, torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    return tokenizer.decode(generated, skip_special_tokens=True).strip().strip("`").strip()


def prepare_retrieval(args: argparse.Namespace, device: torch.device) -> dict[str, list[dict[str, Any]]]:
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    records = bank["records"]
    train_indices = [index for index, row in enumerate(records) if row["memory_source"] == "train_successful_steering"]
    test_indices = [index for index, row in enumerate(records) if row["memory_source"] == "test_memory_key_only"]
    if len(train_indices) != 6927 or len(test_indices) != 1103:
        raise ValueError(f"unexpected bank split: train={len(train_indices)}, test={len(test_indices)}")
    labels = set(bank["manifest"]["labels"])
    if {records[index]["emotion_label"] for index in test_indices} != labels:
        raise ValueError("test records do not exactly cover the configured 11 labels")
    query_by_id = {
        row["memory_id"]: row for row in read_jsonl(QUERIES)
        if row.get("query_generation_status") == "ok" and row.get("emotion_label") in labels
    }
    selected = [index for index in test_indices if records[index]["memory_id"] in query_by_id]
    if len(selected) != 1103:
        raise ValueError(f"missing evaluation queries: expected 1103, got {len(selected)}")
    if args.limit is not None:
        selected = selected[:args.limit]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    selected = selected[args.shard_index::args.shard_count]
    # Encode the test question at evaluation time, then search the complete
    # bank.  The precomputed test vectors remain candidate keys; they are not
    # substituted for the query vectors.
    candidate_indices = torch.arange(len(records))
    bank_fact = bank["fact_vectors"][candidate_indices].float()
    bank_cue = bank["emotion_cue_vectors"][candidate_indices].float()
    question_texts = [query_by_id[records[index]["memory_id"]]["evaluation_query"] for index in selected]
    query_fact, query_cue = encode_retrieval_questions(question_texts, device, args.retrieval_embed_batch_size, args.retrieval_max_length)
    for name, vectors in (("bank_fact", bank_fact), ("bank_cue", bank_cue), ("query_fact", query_fact), ("query_cue", query_cue)):
        if not torch.allclose(torch.linalg.vector_norm(vectors, dim=1), torch.ones(len(vectors)), atol=2e-3):
            raise ValueError(f"{name} is not L2-normalized")
    fact_scores, emotion_scores = query_fact @ bank_fact.T, query_cue @ bank_cue.T
    rerank_k = min(args.rerank_candidate_k, len(train_indices))
    emotion_candidates, candidate_indices = torch.topk(emotion_scores, k=rerank_k, dim=1)
    candidate_fact = fact_scores.gather(1, candidate_indices)
    rerank_values, rerank_local_indices = (emotion_candidates + args.rerank_fact_weight * candidate_fact).max(dim=1)
    rerank_indices = candidate_indices.gather(1, rerank_local_indices[:, None]).squeeze(1)
    selections = {
        "dual_concat": (fact_scores + emotion_scores).max(dim=1),
        "emotion_only": emotion_scores.max(dim=1),
        "emotion_first_fact_rerank_lambda_0p1": (rerank_values, rerank_indices),
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for method, (values, local_indices) in selections.items():
        prepared = []
        for query_index, score, local_index, fact_score, emotion_score in zip(selected, values, local_indices, fact_scores.gather(1, local_indices[:, None]).squeeze(1), emotion_scores.gather(1, local_indices[:, None]).squeeze(1)):
            # All selection methods return indices in the complete bank.  The
            # rerank path gathers global indices from its Top-K shortlist;
            # dual/emotion paths already produce global indices directly.
            query_record, retrieved = records[query_index], records[int(local_index)]
            query = query_by_id[query_record["memory_id"]]
            prepared.append({
                "memory_id": query_record["memory_id"], "test_bank_row_id": query_record["bank_row_id"],
                "target_emotion": query_record["emotion_label"], "evaluation_query": query["evaluation_query"],
                "query_factual_memory": query_record["factual_memory"], "query_original_text": query_record["original_text"],
                "retrieval_method": method, "retrieved_bank_row_id": retrieved["bank_row_id"],
                "retrieved_memory_id": retrieved["memory_id"], "retrieved_emotion": retrieved["emotion_label"],
                "retrieved_memory_source": retrieved["memory_source"], "retrieved_factual_memory": retrieved["factual_memory"], "retrieved_similarity": float(score),
                "fact_score": float(fact_score), "emotion_score": float(emotion_score),
                "retrieval_emotion_match": int(query_record["emotion_label"] == retrieved["emotion_label"]),
            })
        output[method] = prepared
    return output


def configured_vectors(values: dict[str, Any], emotion: str) -> dict[int, torch.Tensor]:
    item = values["values"][emotion]
    layers, vectors = item["layers"], item["configured_layer_vectors"]
    if len(layers) != len(vectors) or vectors.ndim != 2:
        raise ValueError(f"invalid configured vector artifact for {emotion}")
    return {int(layer): vectors[index] for index, layer in enumerate(layers)}


def score_records(records: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    tokenizer, model, labels = load_classifier(device)
    for start in range(0, len(records), batch_size):
        block = records[start:start + batch_size]
        encoded = tokenizer([row["generated_text"] for row in block], padding=True, truncation=True, max_length=512, return_tensors="pt")
        probabilities = torch.sigmoid(model(**{key: value.to(device) for key, value in encoded.items()}).logits).cpu()
        for row, probability in zip(block, probabilities):
            target_index = labels.index(row["target_emotion"])
            top = int(probability.argmax().item())
            row["modernbert_predicted_label"] = labels[top]
            row["modernbert_top_score"] = float(probability[top])
            row["target_emotion_score"] = float(probability[target_index])
            row["emotion_accuracy"] = int(labels[top] == row["target_emotion"])
    del model
    torch.cuda.empty_cache()
    return records


def report(method_dir: Path, records: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["condition"], record["target_emotion"])].append(record)
    report_rows: list[dict[str, Any]] = []
    for (condition, emotion), group in sorted(groups.items()):
        report_rows.append({"scope": "per_emotion", "condition": condition, "emotion": emotion, "n": len(group), "emotion_accuracy": sum(row["emotion_accuracy"] for row in group) / len(group), "mean_target_emotion_score": sum(row["target_emotion_score"] for row in group) / len(group), "mean_emotion_gain": sum(row["emotion_gain_vs_baseline"] for row in group) / len(group), "retrieval_emotion_match": sum(row["retrieval_emotion_match"] for row in group) / len(group)})
    for condition in ("baseline", "steered_configured_multilayer"):
        group = [row for row in records if row["condition"] == condition]
        report_rows.append({"scope": "micro", "condition": condition, "emotion": "all", "n": len(group), "emotion_accuracy": sum(row["emotion_accuracy"] for row in group) / len(group), "mean_target_emotion_score": sum(row["target_emotion_score"] for row in group) / len(group), "mean_emotion_gain": sum(row["emotion_gain_vs_baseline"] for row in group) / len(group), "retrieval_emotion_match": sum(row["retrieval_emotion_match"] for row in group) / len(group)})
    with (method_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader(); writer.writerows(report_rows)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda")
    retrieval = prepare_retrieval(args, device)
    retrieval = {method: retrieval[method] for method in args.methods}
    run_dir = args.output_dir / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for method, rows in retrieval.items():
        write_jsonl(run_dir / method / "retrieval.jsonl", rows)
    config = {"bank": str(BANK), "injection_values": "embedded in memory bank", "retrieval_records": 8030, "train_records": 6927, "retrievable_test_fact_records": 1103, "test_queries": 1103, "query_fact_source": "evaluation_query encoded at evaluation time by Qwen3-Embedding-0.6B", "query_emotion_source": "evaluation_query encoded at evaluation time by ModernBERT layer-22 masked mean", "methods": args.methods, "rerank_protocol": "emotion Top-K followed by emotion + beta * fact", "rerank_candidate_k": args.rerank_candidate_k, "rerank_fact_weight": args.rerank_fact_weight, "injection": "configured_layer_vectors added directly at each specified residual layer; no second alpha multiplication", "effective_global_alpha": 1.0, "max_new_tokens": args.max_new_tokens, "shard_index": args.shard_index, "shard_count": args.shard_count}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "retrieval_prepared", "queries_in_shard": len(next(iter(retrieval.values()))), **config}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    values = bank["injection_values"]
    if values["test_data_used"] or values["num_accepted_train_records"] != 6927:
        raise ValueError("unexpected value library provenance")
    tokenizer, model = load_qwen(device)
    layers = {layer for item in values["values"].values() for layer in item["layers"]}
    controller = MultilayerSteering(model, layers)
    baseline_path = run_dir / "baseline_generations.jsonl"
    baseline = {row["memory_id"]: row for row in read_jsonl(baseline_path)}
    canonical_queries = next(iter(retrieval.values()))
    for index, row in enumerate(canonical_queries, start=1):
        if row["memory_id"] not in baseline:
            baseline_row = {**row, "condition": "baseline", "injected_layers": [], "layer_weights": {}, "generated_text": generate(tokenizer, model, controller, row["evaluation_query"], None, args.max_new_tokens)}
            append_jsonl(baseline_path, baseline_row)
            baseline[row["memory_id"]] = baseline_row
        if index % 10 == 0 or index == len(canonical_queries):
            print(json.dumps({"condition": "baseline", "completed": index, "total": len(canonical_queries)}, ensure_ascii=False), flush=True)
    for method, retrieval_rows in retrieval.items():
        generation_path = run_dir / method / "steered_generations.jsonl"
        existing = {row["memory_id"] for row in read_jsonl(generation_path)}
        for index, row in enumerate(retrieval_rows, start=1):
            if row["memory_id"] not in existing:
                item = values["values"][row["retrieved_emotion"]]
                vectors = configured_vectors(values, row["retrieved_emotion"])
                append_jsonl(generation_path, {**row, "condition": "steered_configured_multilayer", "injected_layers": item["layers"], "layer_weights": item["layer_weights"], "generated_text": generate(tokenizer, model, controller, row["evaluation_query"], vectors, args.max_new_tokens)})
            if index % 10 == 0 or index == len(retrieval_rows):
                print(json.dumps({"method": method, "condition": "steered", "completed": index, "total": len(retrieval_rows)}, ensure_ascii=False), flush=True)
    controller.close(); del model; torch.cuda.empty_cache()
    baseline_scored_path = run_dir / "baseline_scored.jsonl"
    baseline_scored = read_jsonl(baseline_scored_path)
    if len(baseline_scored) != len(canonical_queries):
        baseline_scored = score_records(list(baseline.values()), device, args.classifier_batch_size)
        write_jsonl(baseline_scored_path, baseline_scored)
    baseline_by_id = {row["memory_id"]: row for row in baseline_scored}
    for method in args.methods:
        method_dir = run_dir / method
        scored_path = method_dir / "scored_results.jsonl"
        steered = read_jsonl(method_dir / "steered_generations.jsonl")
        if len(read_jsonl(scored_path)) != 2 * len(steered):
            steered = score_records(steered, device, args.classifier_batch_size)
            combined = []
            for row in steered:
                base = dict(baseline_by_id[row["memory_id"]])
                base["retrieval_method"] = method
                base["retrieval_emotion_match"] = row["retrieval_emotion_match"]
                base["emotion_gain_vs_baseline"] = 0.0
                row["emotion_gain_vs_baseline"] = row["target_emotion_score"] - base["target_emotion_score"]
                combined.extend((base, row))
            write_jsonl(scored_path, combined)
        report(method_dir, read_jsonl(scored_path))
    print(json.dumps({"status": "complete", "output": str(run_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
