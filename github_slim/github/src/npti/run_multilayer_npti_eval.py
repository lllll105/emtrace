#!/usr/bin/env python3
"""Formal 11-class all-layer NPTI evaluation.

Queries are encoded from evaluation_query at evaluation time.  The precomputed
test vectors are candidate-bank vectors only; they are never used as queries.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[3]
COMMON = ROOT / "code/common"
import sys
sys.path.insert(0, str(COMMON))
from goemotions import GOEMOTIONS_LABELS  # noqa: E402

BUILD = ROOT / "use/build/multilayer_11class_trainplus_test_memory"
BANK = BUILD / "qwen_multilayer_memory_bank_regenerated_classified_6927.pt"
NPTI_SET = ROOT / "use/npti_set/current_bank_11class/emotion_neuron_sets_11class.json"
QUERIES = ROOT / "data/test_create_bygpt/goemotions_test_queries_by_gpt.jsonl"
QWEN = "./local/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"
QWEN_EMBED = ROOT / "model/Qwen3-Embedding-0.6B"
MODERNBERT = "./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487"
OUTPUT = ROOT / "outputs/npti/multilayer_11class_fullbank_20260918"
METHODS = ("fact_only", "dual_concat", "emotion_only", "emotion_first_fact_rerank_lambda_0p1")
DEFAULT_METHODS = ("dual_concat", "emotion_only", "emotion_first_fact_rerank_lambda_0p1")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=OUTPUT)
    p.add_argument("--neuron-set", type=Path, default=NPTI_SET)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    p.add_argument("--retrieval-embed-batch-size", type=int, default=32)
    p.add_argument("--retrieval-max-length", type=int, default=512)
    p.add_argument("--rerank-candidate-k", type=int, default=50)
    p.add_argument("--rerank-fact-weight", type=float, default=0.1)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--classifier-batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def last_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(torch.all(mask[:, -1])):
        return hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]


def cue_tokenizer() -> Any:
    try:
        return AutoTokenizer.from_pretrained(MODERNBERT, local_files_only=True)
    except ValueError:
        return PreTrainedTokenizerFast(tokenizer_file=str(Path(MODERNBERT) / "tokenizer.json"), unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]")


@torch.inference_mode()
def encode_queries(texts: list[str], device: torch.device, batch_size: int, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    qt = AutoTokenizer.from_pretrained(QWEN_EMBED, padding_side="left", local_files_only=True)
    qm = AutoModel.from_pretrained(QWEN_EMBED, torch_dtype=torch.float16, local_files_only=True).to(device).eval()
    fact = []
    for start in range(0, len(texts), batch_size):
        enc = qt(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        fact.append(F.normalize(last_pool(qm(**enc).last_hidden_state, enc["attention_mask"]), p=2, dim=1).float().cpu())
    del qm, qt
    torch.cuda.empty_cache()
    ct, cm = cue_tokenizer(), AutoModel.from_pretrained(MODERNBERT, local_files_only=True).to(device).eval()
    cue = []
    for start in range(0, len(texts), batch_size):
        enc = ct(texts[start:start + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in {"input_ids", "attention_mask"}}
        hidden = cm(**enc, output_hidden_states=True, return_dict=True).hidden_states[22]
        mask = enc["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        cue.append(F.normalize((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0), p=2, dim=1).float().cpu())
    del cm, ct
    torch.cuda.empty_cache()
    return torch.cat(fact), torch.cat(cue)


def prepare_retrieval(setting: argparse.Namespace, device: torch.device, run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    records = bank["records"]
    if len(records) != 8030:
        raise ValueError(f"expected 8030 candidate records, got {len(records)}")
    labels = set(bank["manifest"]["labels"])
    if len(labels) != 11:
        raise ValueError(f"expected 11 labels, got {sorted(labels)}")
    query_by_id = {row["memory_id"]: row for row in read_jsonl(QUERIES) if row.get("query_generation_status") == "ok" and row.get("emotion_label") in labels}
    test_indices = [i for i, row in enumerate(records) if row["memory_source"] == "test_memory_key_only" and row["memory_id"] in query_by_id]
    if len(test_indices) != 1103:
        raise ValueError(f"expected 1103 test queries, got {len(test_indices)}")
    if setting.limit is not None:
        test_indices = test_indices[:setting.limit]
    test_indices = test_indices[setting.shard_index::setting.shard_count]
    texts = [query_by_id[records[i]["memory_id"]]["evaluation_query"] for i in test_indices]
    query_fact, query_cue = encode_queries(texts, device, setting.retrieval_embed_batch_size, setting.retrieval_max_length)
    bank_fact, bank_cue = bank["fact_vectors"].float(), bank["emotion_cue_vectors"].float()
    fact_scores, emotion_scores = query_fact @ bank_fact.T, query_cue @ bank_cue.T
    rerank_k = min(setting.rerank_candidate_k, len(records))
    top_emotion, candidate_indices = torch.topk(emotion_scores, k=rerank_k, dim=1)
    candidate_fact = fact_scores.gather(1, candidate_indices)
    rerank_scores = top_emotion + setting.rerank_fact_weight * candidate_fact
    rerank_values, local = rerank_scores.max(dim=1)
    selected = {
        "fact_only": fact_scores.max(dim=1),
        "dual_concat": ((fact_scores + emotion_scores).max(dim=1)),
        "emotion_only": emotion_scores.max(dim=1),
        "emotion_first_fact_rerank_lambda_0p1": (rerank_values, candidate_indices.gather(1, local[:, None]).squeeze(1)),
    }
    output = {}
    for method, result in selected.items():
        values, indices = result
        rows = []
        for q_index, score, bank_index in zip(test_indices, values, indices):
            query_record, query = records[q_index], query_by_id[records[q_index]["memory_id"]]
            retrieved = records[int(bank_index)]
            rows.append({
                "memory_id": query_record["memory_id"], "test_bank_row_id": query_record["bank_row_id"],
                "target_emotion": query_record["emotion_label"], "evaluation_query": query["evaluation_query"],
                "query_factual_memory": query_record["factual_memory"], "query_original_text": query_record["original_text"],
                "retrieval_method": method, "retrieved_bank_row_id": retrieved["bank_row_id"],
                "retrieved_memory_id": retrieved["memory_id"], "retrieved_emotion": retrieved["emotion_label"],
                "retrieved_memory_source": retrieved["memory_source"], "retrieved_factual_memory": retrieved["factual_memory"],
                "retrieved_similarity": float(score), "retrieval_emotion_match": int(query_record["emotion_label"] == retrieved["emotion_label"]),
                "fact_score": float(fact_scores[test_indices.index(q_index), int(bank_index)]),
                "emotion_score": float(emotion_scores[test_indices.index(q_index), int(bank_index)]),
            })
        output[method] = rows
        write_jsonl(run_dir / method / "retrieval.jsonl", rows)
    return output


def load_qwen(device: torch.device):
    tok = AutoTokenizer.from_pretrained(QWEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    return tok, model


def token_inputs(tok: Any, query: str, device: torch.device) -> dict[str, torch.Tensor]:
    payload = tok.apply_chat_template([{"role": "user", "content": f"User query:\n{query}\n\nGenerate a response."}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    if isinstance(payload, torch.Tensor):
        return {"input_ids": payload.to(device)}
    return {key: value.to(device) for key, value in payload.items() if torch.is_tensor(value)}


def npti_weight(delta: float) -> float:
    return 1.0 / (1.0 + torch.exp(torch.tensor(-10.0 * (abs(delta) - 0.15))).item())


class NPTIController:
    def __init__(self, model: Any):
        self.active: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers, start=1):
            layer.mlp.forward = self.forward_for(layer_index).__get__(layer.mlp, type(layer.mlp))

    def forward_for(self, layer_index: int):
        owner = self
        def forward(module: Any, x: torch.Tensor):
            activation = module.act_fn(module.gate_proj(x))
            state = owner.active.get(layer_index)
            if state is not None:
                pos, values, neg = state
                if pos.numel():
                    activation[:, -1, pos.to(activation.device)] += values.to(activation.device, activation.dtype)
                if neg.numel():
                    idx = neg.to(activation.device)
                    activation[:, -1, idx] = torch.minimum(activation[:, -1, idx], torch.zeros((), device=activation.device, dtype=activation.dtype))
            return module.down_proj(activation * module.up_proj(x))
        return forward

    @contextmanager
    def use(self, state: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None):
        self.active = state or {}
        try:
            yield
        finally:
            self.active = {}


@torch.inference_mode()
def generate(tok: Any, model: Any, controller: NPTIController, query: str, state: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None, max_tokens: int) -> str:
    device = next(model.parameters()).device
    with controller.use(state):
        out = model(**token_inputs(tok, query, device), use_cache=True)
    cache, next_token = out.past_key_values, out.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = []
    stops = {x for x in (tok.eos_token_id, tok.pad_token_id, tok.bos_token_id) if x is not None}
    while len(generated) < max_tokens:
        token = int(next_token.item())
        if token in stops:
            break
        generated.append(token)
        with controller.use(state):
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        cache, next_token = out.past_key_values, out.logits[:, -1].argmax(dim=-1, keepdim=True)
    return tok.decode(generated, skip_special_tokens=True).strip()


def load_state(sets: dict[str, Any], emotion: str, gamma: float = 1.0) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if emotion not in sets:
        raise ValueError(f"missing NPTI set for retrieved emotion {emotion}")
    item = sets[emotion]
    state = {}
    for layer_name, positive in item["positive_neurons"].items():
        layer = int(layer_name.split("_")[1])
        negative = item["negative_neurons"][layer_name]
        pidx = torch.tensor([x["neuron_id"] for x in positive], dtype=torch.long)
        pval = torch.tensor([gamma * float(x["a95"]) * npti_weight(float(x["delta"])) for x in positive], dtype=torch.float32)
        nidx = torch.tensor([x["neuron_id"] for x in negative], dtype=torch.long)
        state[layer] = (pidx, pval, nidx)
    if len(state) != 28:
        raise ValueError(f"{emotion} has {len(state)} NPTI layers, expected 28")
    return state


def score_and_summarize(method_dir: Path, baseline_by_id: dict[str, dict[str, Any]], device: torch.device, batch_size: int) -> None:
    rows = read_jsonl(method_dir / "steered_generations.jsonl")
    tok = cue_tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(MODERNBERT, local_files_only=True).to(device).eval()
    labels = [str(model.config.id2label.get(i, f"label_{i}")).lower() for i in range(model.config.num_labels)]
    if all(x.startswith("label_") for x in labels): labels = GOEMOTIONS_LABELS[:model.config.num_labels]
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            block = rows[start:start + batch_size]
            enc = tok([x["generated_text"] for x in block], padding=True, truncation=True, max_length=512, return_tensors="pt")
            probs = torch.sigmoid(model(**{k: v.to(device) for k, v in enc.items()}).logits).cpu()
            for row, prob in zip(block, probs):
                target = labels.index(row["target_emotion"])
                top = int(prob.argmax())
                row.update(modernbert_predicted_label=labels[top], target_emotion_score=float(prob[target]), emotion_accuracy=int(labels[top] == row["target_emotion"]))
                row["emotion_gain_vs_baseline"] = row["target_emotion_score"] - baseline_by_id[row["memory_id"]]["target_emotion_score"]
    write_jsonl(method_dir / "scored_results.jsonl", rows)
    report = []
    for condition, group in (("baseline", list(baseline_by_id.values())), ("npti_all_layers", rows)):
        report.append({"scope": "micro", "condition": condition, "n": len(group), "emotion_accuracy": sum(x["emotion_accuracy"] for x in group) / len(group), "target_score": sum(x["target_emotion_score"] for x in group) / len(group), "emotion_gain": sum(x.get("emotion_gain_vs_baseline", 0.0) for x in group) / len(group), "retrieval_match": sum(x["retrieval_emotion_match"] for x in group) / len(group)})
    with (method_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0])); writer.writeheader(); writer.writerows(report)
    (method_dir / "summary.md").write_text("# 11-class All-layer NPTI\n\n" + "\n".join("| {condition} | {emotion_accuracy:.6f} | {target_score:.6f} | {emotion_gain:.6f} | {retrieval_match:.6f} |".format(**x) for x in report) + "\n", encoding="utf-8")


def main() -> None:
    setting = args()
    os.environ["CUDA_VISIBLE_DEVICES"] = setting.gpu
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if setting.shard_count < 1 or not 0 <= setting.shard_index < setting.shard_count: raise ValueError("invalid shard")
    device = torch.device("cuda")
    run_dir = setting.output_dir / f"shard_{setting.shard_index:02d}_of_{setting.shard_count:02d}"
    bank = torch.load(BANK, map_location="cpu", weights_only=False)
    sets = json.loads(setting.neuron_set.read_text(encoding="utf-8"))
    if set(sets) != set(bank["manifest"]["labels"]): raise ValueError("NPTI labels and bank labels differ")
    retrieval = prepare_retrieval(setting, device, run_dir)
    config = {"model": "Qwen2.5-7B-Instruct", "npti_set": str(setting.neuron_set), "candidate_bank": str(BANK), "candidate_count": 8030, "test_queries": 1103, "query_fact": "evaluation_query encoded by Qwen3-Embedding-0.6B", "query_emotion": "evaluation_query encoded by ModernBERT layer-22 masked mean", "activation_site": "FFN intermediate act_fn(gate_proj(x))", "injection": "all 28 layers; P+ sigmoid weighting and P- min(0,h)", "gamma": setting.gamma, "max_new_tokens": setting.max_new_tokens, "do_sample": False, "methods": setting.methods, "shard_index": setting.shard_index, "shard_count": setting.shard_count}
    run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if setting.prepare_only: return
    tok, model = load_qwen(device); controller = NPTIController(model)
    canonical = next(iter(retrieval.values()))
    baseline_path = run_dir / "baseline_generations.jsonl"
    baseline = {x["memory_id"]: x for x in read_jsonl(baseline_path)}
    for row in canonical:
        if row["memory_id"] not in baseline:
            text = generate(tok, model, controller, row["evaluation_query"], None, setting.max_new_tokens)
            baseline[row["memory_id"]] = {**row, "condition": "baseline", "generated_text": text, "target_emotion_score": 0.0, "emotion_accuracy": 0}
            append_jsonl(baseline_path, baseline[row["memory_id"]])
    for method in setting.methods:
        path = run_dir / method / "steered_generations.jsonl"; done = {x["memory_id"] for x in read_jsonl(path)}
        for row in retrieval[method]:
            if row["memory_id"] in done: continue
            state = load_state(sets, row["retrieved_emotion"], setting.gamma)
            append_jsonl(path, {**row, "condition": "npti_all_layers", "generated_text": generate(tok, model, controller, row["evaluation_query"], state, setting.max_new_tokens)})
    del controller, model, tok; gc.collect(); torch.cuda.empty_cache()
    # Score baselines once, then each method.  Baseline rows are shared within this shard.
    base_rows = list(baseline.values())
    btok = cue_tokenizer(); bmodel = AutoModelForSequenceClassification.from_pretrained(MODERNBERT, local_files_only=True).to(device).eval(); labels = [str(bmodel.config.id2label.get(i, f"label_{i}")).lower() for i in range(bmodel.config.num_labels)]; labels = GOEMOTIONS_LABELS[:bmodel.config.num_labels] if all(x.startswith("label_") for x in labels) else labels
    with torch.inference_mode():
        for start in range(0, len(base_rows), setting.classifier_batch_size):
            block = base_rows[start:start + setting.classifier_batch_size]; enc = btok([x["generated_text"] for x in block], padding=True, truncation=True, max_length=512, return_tensors="pt"); probs = torch.sigmoid(bmodel(**{k:v.to(device) for k,v in enc.items()}).logits).cpu()
            for row, prob in zip(block, probs): row.update(target_emotion_score=float(prob[labels.index(row["target_emotion"])]), emotion_accuracy=int(labels[int(prob.argmax())] == row["target_emotion"]))
    del bmodel, btok; torch.cuda.empty_cache(); write_jsonl(run_dir / "baseline_scored.jsonl", base_rows); baseline = {x["memory_id"]: x for x in base_rows}
    for method in setting.methods: score_and_summarize(run_dir / method, baseline, device, setting.classifier_batch_size)


if __name__ == "__main__": main()
