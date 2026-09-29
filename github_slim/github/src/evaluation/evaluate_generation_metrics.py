#!/usr/bin/env python3
"""Evaluate completed generation JSONL files with emotion and NLI metrics.

Emotion metrics use ModernBERT-GoEmotions.  Fact consistency uses the local
DeBERTa-v3 MNLI model with premise=generated text and hypothesis=the matching
test sample's original_text.  This script only evaluates existing generations;
it does not generate text or apply steering.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer, PreTrainedTokenizerFast)

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "use/build/multilayer_11class_trainplus_test_memory/qwen_multilayer_memory_bank_regenerated_classified_6927.pt"
MODERNBERT = "./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487"
NLI = "./local/.cache/modelscope/hub/models/Xenova/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
NLI_ONNX_RUNNER = ROOT / "use_2rag/code/run_xenova_nli_onnx.js"
ONNXRUNTIME_NODE = "./local/.vscode-server/extensions/visualstudioexptteam.vscodeintellicode-completions-2.0.1/dist/node_modules"
ELEVEN = ("admiration", "amusement", "curiosity", "joy", "confusion", "sadness", "disappointment", "surprise", "excitement", "desire", "embarrassment")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, action="append", default=[], help="Generation output directory; repeat for multiple methods.")
    p.add_argument("--input-file", type=Path, action="append", default=[], help="Generation JSONL file; repeat for multiple files.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--condition-field", default="condition", help="Row field used as the summary condition.")
    p.add_argument("--skip-condition", action="append", default=[], help="Input condition to exclude, e.g. a duplicate baseline.")
    p.add_argument("--only-condition", action="append", default=[], help="If set, evaluate only these input conditions.")
    p.add_argument("--max-rows", type=int, default=None, help="Stop after this many rows, for a quick diagnostic run.")
    p.add_argument("--fact-reference", choices=("original_text", "factual_memory"), default="original_text",
                   help="Reference used as the NLI hypothesis; emotion similarity always uses original_text.")
    p.add_argument("--reuse-emotion-metrics", action="store_true",
                   help="Reuse existing ModernBERT scores when rescoring an evaluated JSONL.")
    p.add_argument("--bank", type=Path, default=BANK)
    p.add_argument("--device", default="cuda")
    p.add_argument("--emotion-batch-size", type=int, default=32)
    p.add_argument("--nli-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def tokenizer(path: str):
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except ValueError:
        return PreTrainedTokenizerFast(tokenizer_file=str(Path(path) / "tokenizer.json"), unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]")


def labels_from_model(model: Any) -> list[str]:
    return [str(model.config.id2label.get(i, f"label_{i}")).lower() for i in range(model.config.num_labels)]


def emotion_score(rows: list[dict[str, Any]], device: torch.device, batch_size: int, max_length: int) -> None:
    tok = tokenizer(MODERNBERT)
    model = AutoModelForSequenceClassification.from_pretrained(MODERNBERT, local_files_only=True).to(device).eval()
    labels = labels_from_model(model)
    for start in range(0, len(rows), batch_size):
        block = rows[start:start + batch_size]
        encoded = tok([x["generated_text"] for x in block], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.inference_mode():
            probability = torch.sigmoid(model(**{k: v.to(device) for k, v in encoded.items()}).logits).cpu()
        for row, p in zip(block, probability):
            target = row["target_emotion"].lower()
            if target not in labels:
                raise ValueError(f"target emotion {target!r} is absent from ModernBERT labels")
            target_index = labels.index(target)
            predicted_index = int(p.argmax().item())
            reference_probability = row.pop("_reference_emotion_probability", None)
            row["modernbert_labels"] = labels
            row["generated_emotion_probabilities"] = [float(x) for x in p]
            row["modernbert_predicted_label"] = labels[predicted_index]
            row["emotion_accuracy"] = int(labels[predicted_index] == target)
            row["target_emotion_score"] = float(p[target_index])
            row["emotion_score_1"] = float(p[target_index])
    # The reference distribution is encoded in a second pass by reference_score.
    del model
    torch.cuda.empty_cache()


def reference_score(rows: list[dict[str, Any]], device: torch.device, batch_size: int, max_length: int) -> None:
    tok = tokenizer(MODERNBERT)
    model = AutoModelForSequenceClassification.from_pretrained(MODERNBERT, local_files_only=True).to(device).eval()
    labels = labels_from_model(model)
    for start in range(0, len(rows), batch_size):
        block = rows[start:start + batch_size]
        encoded = tok([x["reference_original_text"] for x in block], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.inference_mode():
            probability = torch.sigmoid(model(**{k: v.to(device) for k, v in encoded.items()}).logits).cpu()
        for row, p in zip(block, probability):
            generated = torch.tensor(row["generated_emotion_probabilities"], dtype=torch.float32)
            row["reference_emotion_probabilities"] = [float(x) for x in p]
            row["emotion_similarity_1"] = float(F.cosine_similarity(generated.unsqueeze(0), p.unsqueeze(0)).item())
            row["emotion_similarity_full_distribution"] = row["emotion_similarity_1"]
            indices = [labels.index(label) for label in ELEVEN if label in labels]
            row["emotion_similarity_11class"] = float(F.cosine_similarity(generated[indices].unsqueeze(0), p[indices].unsqueeze(0)).item())
    del model
    torch.cuda.empty_cache()


def nli_score(rows: list[dict[str, Any]], device: torch.device, batch_size: int,
              max_length: int, reference_field: str = "reference_original_text") -> None:
    # Retrieval methods often produce the same deterministic answer for a
    # sample. NLI depends only on the text pair, so score each distinct pair
    # once and copy its probabilities back to every method-specific row.
    keyed = {(row["generated_text"], row[reference_field]): row for row in rows}
    distinct = list(keyed.values())
    print(json.dumps({"nli_rows": len(rows), "unique_text_pairs": len(distinct)}), flush=True)
    tok = tokenizer(NLI)
    config = json.loads((Path(NLI) / "config.json").read_text(encoding="utf-8"))
    labels = [str(config["id2label"][str(i)]).lower() for i in range(len(config["id2label"]))]
    normalized = {label.replace("_", "-"): i for i, label in enumerate(labels)}
    entailment = next((i for key, i in normalized.items() if "entail" in key), None)
    neutral = next((i for key, i in normalized.items() if "neutral" in key), None)
    contradiction = next((i for key, i in normalized.items() if "contrad" in key), None)
    if entailment is None:
        raise ValueError(f"could not identify entailment label in {labels}")
    env = {**os.environ, "NODE_PATH": ONNXRUNTIME_NODE}
    runner = subprocess.Popen(["node", str(NLI_ONNX_RUNNER), str(Path(NLI) / "onnx/model.onnx")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    try:
        for start in range(0, len(distinct), batch_size):
            block = distinct[start:start + batch_size]
            # NLI convention: premise=generated, hypothesis=selected test reference.
            encoded = tok([x["generated_text"] for x in block], [x[reference_field] for x in block], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            request = {"input_ids": encoded["input_ids"].tolist(), "attention_mask": encoded["attention_mask"].tolist()}
            assert runner.stdin is not None and runner.stdout is not None
            runner.stdin.write(json.dumps(request) + "\n"); runner.stdin.flush()
            response = json.loads(runner.stdout.readline())
            if "error" in response:
                raise RuntimeError(f"Xenova ONNX NLI error: {response['error']}")
            probability = torch.softmax(torch.tensor(response["logits"], dtype=torch.float32).reshape(len(block), len(labels)), dim=-1)
            for row, p in zip(block, probability):
                row["nli_model"] = NLI
                row["nli_runtime"] = "onnxruntime-node"
                row["nli_labels"] = labels
                row["nli_entailment_probability"] = float(p[entailment])
                row["nli_neutral_probability"] = float(p[neutral]) if neutral is not None else None
                row["nli_contradiction_probability"] = float(p[contradiction]) if contradiction is not None else None
                row["fact_consistency_score_1"] = float(p[entailment])
    finally:
        if runner.stdin is not None: runner.stdin.close()
        runner.wait(timeout=30)
    fields = ("nli_model", "nli_runtime", "nli_labels", "nli_entailment_probability",
              "nli_neutral_probability", "nli_contradiction_probability", "fact_consistency_score_1")
    for row in rows:
        source = keyed[(row["generated_text"], row[reference_field])]
        for field in fields:
            row[field] = source[field]


def summarize(rows: list[dict[str, Any]], output: Path) -> None:
    report: list[dict[str, Any]] = []
    conditions = sorted({str(x.get("condition", "unknown")) for x in rows})
    emotions = sorted({str(x["target_emotion"]) for x in rows})
    for condition in conditions:
        for emotion in ["all", *emotions]:
            group = [x for x in rows if x.get("condition", "unknown") == condition and (emotion == "all" or x["target_emotion"] == emotion)]
            if not group:
                continue
            report.append({
                "scope": "micro" if emotion == "all" else "per_emotion", "condition": condition, "emotion": emotion, "n": len(group),
                "emotion_accuracy": sum(x["emotion_accuracy"] for x in group) / len(group),
                "emotion_score_1": sum(x["emotion_score_1"] for x in group) / len(group),
                "emotion_similarity_1": sum(x["emotion_similarity_1"] for x in group) / len(group),
                "emotion_similarity_11class": sum(x["emotion_similarity_11class"] for x in group) / len(group),
                "fact_consistency_score_1": sum(x["fact_consistency_score_1"] for x in group) / len(group),
            })
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0])); writer.writeheader(); writer.writerows(report)


def main() -> None:
    a = parse_args()
    if not torch.cuda.is_available() and a.device.startswith("cuda"):
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly")
    device = torch.device(a.device)
    bank = torch.load(a.bank, map_location="cpu", weights_only=False)
    references = {row["memory_id"]: row for row in bank["records"]}
    files: list[Path] = list(a.input_file)
    for input_dir in a.input_dir:
        direct = input_dir / "scored_results.jsonl"
        if direct.exists():
            files.append(direct)
            continue
        generations = sorted(
            path for path in input_dir.rglob("*.jsonl")
            if path.name.endswith("_generations.jsonl") or path.name == "generations.jsonl"
        )
        if generations:
            # Prefer raw generation files so directories containing both
            # baseline generations and method-specific scored files do not
            # silently omit the baseline condition.
            files.extend(generations)
        else:
            files.extend(sorted(input_dir.rglob("scored_results.jsonl")))
    files = list(dict.fromkeys(files))
    if not files:
        raise FileNotFoundError("no input JSONL files found")
    rows: list[dict[str, Any]] = []
    for file in files:
        for row in read_jsonl(file):
            if "generated_text" not in row:
                continue
            # Baseline rows can lack the retrieval field. Filter them before
            # requiring the method-specific summary key.
            if row.get("condition") in a.skip_condition:
                continue
            if a.only_condition and row.get("condition") not in a.only_condition:
                continue
            condition = row.get(a.condition_field)
            if condition is None:
                raise KeyError(f"condition field {a.condition_field!r} missing from {file}")
            row["condition"] = str(condition)
            ref = references.get(row["memory_id"])
            if ref is None:
                raise KeyError(f"memory_id not found in evaluation bank: {row['memory_id']}")
            row["evaluation_source_file"] = str(file)
            row["reference_original_text"] = ref["original_text"]
            row["reference_factual_memory"] = ref["factual_memory"]
            row["fact_reference_source"] = f"matching test memory {a.fact_reference}"
            rows.append(row)
            if a.max_rows is not None and len(rows) >= a.max_rows:
                break
        if a.max_rows is not None and len(rows) >= a.max_rows:
            break
    if not rows:
        raise ValueError("input files contain no generated rows")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.reuse_emotion_metrics:
        required = ("emotion_accuracy", "emotion_score_1", "emotion_similarity_1", "emotion_similarity_11class")
        if any(any(row.get(key) is None for key in required) for row in rows):
            raise ValueError("--reuse-emotion-metrics requires complete existing emotion metrics")
    else:
        emotion_score(rows, device, a.emotion_batch_size, a.max_length)
        reference_score(rows, device, a.emotion_batch_size, a.max_length)
    if a.fact_reference == "factual_memory":
        for row in rows:
            if row.get("fact_consistency_score_1") is not None:
                row["fact_consistency_original_text"] = row["fact_consistency_score_1"]
    nli_score(rows, device, a.nli_batch_size, a.max_length,
              "reference_factual_memory" if a.fact_reference == "factual_memory" else "reference_original_text")
    write_jsonl(a.output_dir / "per_generation_metrics.jsonl", rows)
    summarize(rows, a.output_dir)
    (a.output_dir / "config.json").write_text(json.dumps({"emotion_model": MODERNBERT, "nli_model": NLI, "emotion_probability": "sigmoid ModernBERT GoEmotions output", "emotion_similarity": "cosine of full probability vectors; 11-class variant also saved", "fact_reference": f"matching test record {a.fact_reference}", "nli_direction": f"premise=generated_text, hypothesis=reference_{a.fact_reference}", "reuse_emotion_metrics": a.reuse_emotion_metrics, "skip_condition": a.skip_condition, "only_condition": a.only_condition, "max_rows": a.max_rows, "rows": len(rows), "input_files": [str(x) for x in files]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(a.output_dir), "files": [str(x) for x in files]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
