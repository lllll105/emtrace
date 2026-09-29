#!/usr/bin/env python3
"""Construct train-only configured-layer emotion values with the original gated flow.

This script deliberately reads only the training factual dataset.  It creates
neutral/target descriptions, retries ModernBERT gate failures, then averages
only the Qwen residual-state layers specified for each emotion in config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
SWEEP = ROOT / "ex/emotion_layer_every"
sys.path.insert(0, str(SWEEP))
from run_every_emotion_sweep import score_texts, usable_by_emotion  # noqa: E402
sys.path.insert(0, str(ROOT / "ex/emotion_steer_layer/code"))
from run_residual_layer_sweep import (  # noqa: E402
    classifier_label_names, emotion_messages, generate_text,
    last_response_token_hidden, load_causal_lm, load_modernbert_classifier,
    neutral_messages, set_seed,
)


CONFIG = ROOT / "use/build/multilayer_11class_trainplus_test_memory/multilayer_injection_config.json"
DEFAULT_INPUT = ROOT / "data/phase1/goemotions_train_factual_usable.jsonl"
DEFAULT_WORK = ROOT / "use/work/multilayer_value_construction"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--emotions", default=None, help="comma-separated subset; default is all config emotions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-gate-retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def gate_pair(row: dict[str, Any], emotion: str, tokenizer: Any, model: Any, cls_tokenizer: Any, classifier: Any, classifier_device: str, names: list[str], max_tokens: int, retries: int) -> dict[str, Any]:
    fact = row["factual_memory"].strip()
    attempts = []
    for attempt in range(1, retries + 1):
        try:
            neutral = generate_text(tokenizer, model, neutral_messages(fact), max_tokens)
            target = generate_text(tokenizer, model, emotion_messages(fact, emotion), max_tokens)
            neutral_score, target_score = score_texts(cls_tokenizer, classifier, classifier_device, names, [neutral, target])
            accepted = neutral_score["top1"] == "neutral" and target_score["top1"] == emotion
            attempts.append({"attempt": attempt, "neutral_top1": neutral_score["top1"], "target_top1": target_score["top1"], "error": None})
            if accepted:
                return {"memory_id": row["memory_id"], "emotion_label": emotion, "factual_memory": fact, "neutral_description": neutral, "emotion_description": target, "accepted": True, "attempts": attempts}
        except Exception as exc:
            attempts.append({"attempt": attempt, "neutral_top1": None, "target_top1": None, "error": repr(exc)})
            if torch.cuda.is_available() and classifier_device == "cuda":
                torch.cuda.empty_cache()
    return {"memory_id": row["memory_id"], "emotion_label": emotion, "factual_memory": fact, "accepted": False, "attempts": attempts}


def construct_emotion(emotion: str, config: dict[str, Any], args: argparse.Namespace, tokenizer: Any, model: Any, cls_tokenizer: Any, classifier: Any, classifier_device: str, names: list[str]) -> None:
    out = args.work_dir / emotion
    out.mkdir(parents=True, exist_ok=True)
    audit_path, state_path = out / "pair_gate_audit.jsonl", out / "delta_accumulator.pt"
    audit = read_jsonl(audit_path)
    completed = {row["memory_id"] for row in audit}
    candidates = usable_by_emotion(str(args.input_jsonl), emotion)
    for number, row in enumerate(candidates, 1):
        if row["memory_id"] in completed:
            continue
        result = gate_pair(row, emotion, tokenizer, model, cls_tokenizer, classifier, classifier_device, names, args.max_new_tokens, args.max_gate_retries)
        audit.append(result)
        write_jsonl(audit_path, audit)
        if number % 25 == 0 or result["accepted"]:
            accepted = sum(item["accepted"] for item in audit)
            print(json.dumps({"emotion": emotion, "gated": len(audit), "eligible": len(candidates), "accepted": accepted}, ensure_ascii=False), flush=True)
    accepted_rows = [row for row in audit if row["accepted"]]
    layer_weights = {int(layer): float(alpha) for layer, alpha in config["emotions"][emotion]["layer_weights"].items()}
    layers = list(layer_weights)
    state = torch.load(state_path, map_location="cpu", weights_only=False) if state_path.exists() else {"layers": layers, "sum": torch.zeros(len(layers), 3584), "processed_ids": []}
    if state["layers"] != layers:
        raise RuntimeError(f"{emotion}: saved accumulator layers differ from config")
    processed = set(state["processed_ids"])
    for number, row in enumerate(accepted_rows, 1):
        if row["memory_id"] in processed:
            continue
        fact = row["factual_memory"]
        all_layer_delta = last_response_token_hidden(tokenizer, model, emotion_messages(fact, emotion), row["emotion_description"]) - last_response_token_hidden(tokenizer, model, neutral_messages(fact), row["neutral_description"])
        selected_delta = torch.stack([all_layer_delta[layer - 1] for layer in layers])
        state["sum"] += selected_delta.float()
        state["processed_ids"].append(row["memory_id"])
        processed.add(row["memory_id"])
        if number % 10 == 0:
            torch.save(state, state_path)
            print(json.dumps({"emotion": emotion, "hidden_pairs": len(processed), "accepted": len(accepted_rows)}, ensure_ascii=False), flush=True)
    torch.save(state, state_path)
    if not state["processed_ids"]:
        raise RuntimeError(f"{emotion}: no ModernBERT-gated pair was accepted")
    raw = state["sum"] / len(state["processed_ids"])
    configured = torch.stack([raw[index] * layer_weights[layer] for index, layer in enumerate(layers)])
    torch.save({"emotion": emotion, "layers": layers, "raw_layer_vectors": raw, "configured_layer_vectors": configured, "layer_weights": layer_weights, "accepted_pair_count": len(state["processed_ids"]), "source": "training-only original neutral-target ModernBERT-gated flow", "test_data_used": False}, out / "configured_injection_value.pt")
    print(json.dumps({"emotion": emotion, "status": "complete", "accepted_pair_count": len(state["processed_ids"]), "layers": layer_weights}, ensure_ascii=False), flush=True)


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for value construction but is unavailable; refusing to fall back to CPU.")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    emotions = list(config["emotions"]) if args.emotions is None else [x.strip() for x in args.emotions.split(",") if x.strip()]
    if not set(emotions).issubset(config["emotions"]):
        raise ValueError("requested emotion is absent from multilayer config")
    set_seed(args.seed)
    tokenizer, model = load_causal_lm("./local/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct", args.device)
    cls_tokenizer, classifier, classifier_device = load_modernbert_classifier("./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487", args.classifier_device)
    names = classifier_label_names(classifier)
    for emotion in emotions:
        construct_emotion(emotion, config, args, tokenizer, model, cls_tokenizer, classifier, classifier_device, names)


if __name__ == "__main__":
    main()
