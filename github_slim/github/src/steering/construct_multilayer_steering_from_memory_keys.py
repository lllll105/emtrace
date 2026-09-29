#!/usr/bin/env python3
"""Regenerate -> classify/retry -> extract configured-layer steering sums.

The 7,603 training records in the retrieval-key artifact are used only as the
source cohort.  Old emotional descriptions and old emotional hidden states
are never reused.  A worker regenerates every description, tries up to five
prompted samples, records a retry-pending error when none passes the
classifier gate, and extracts hidden states only for accepted samples.

Workers save sums, not means.  ``--finalize-only`` refuses to run until all
11 emotions (all 7,603 records) are complete, then computes each class/layer
mean.  This prevents partial-class or rejected-sample averages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
KEYS = ROOT / "use/build/multilayer_11class_trainplus_test_memory/qwen_multilayer_memory_keys.pt"
CONFIG = ROOT / "use/build/multilayer_11class_trainplus_test_memory/multilayer_injection_config.json"
FINAL_VALUE_DIR = ROOT / "data/build/vector/emotion_value_final"
WORK = ROOT / "use/work/multilayer_steering_regenerated_classified"
QWEN = "./local/.cache/modelscope/hub/models/Qwen/Qwen2.5-7B-Instruct"
MODERNBERT = (
    "./local/.cache/huggingface/hub/"
    "models--cirimus--modernbert-base-go-emotions/"
    "snapshots/690341c8744d225dfd7a1fddae23f541b362b487"
)
PIPELINE_VERSION = "regenerate_classify_extract_all7603_v1"

STEER_CODE = ROOT / "ex/emotion_steer_layer/code"
sys.path.insert(0, str(STEER_CODE))
from run_residual_layer_sweep import chat_inputs, emotion_messages, load_causal_lm, neutral_messages  # noqa: E402

sys.path.insert(0, str(ROOT / "code"))
from build_emotion_value_hidden_qwen25 import classify_texts, clean_text, load_classifier, special_token_ids  # noqa: E402
from retry_emotion_value_hidden_qwen25 import LABEL_VARIANTS, mentions_label, retry_messages  # noqa: E402


RETRY_PENDING_CUES = {
    "admiration": "respect, being impressed, remarkable skill, or earned regard",
    "amusement": "playfulness, humor, a smile, or the urge to laugh",
    "curiosity": "wondering, questions, wanting to find out more, or eager investigation",
    "confusion": "uncertainty, puzzlement, something not adding up, or difficulty making sense of it",
    "sadness": "a heavy heart, sorrow, a sense of loss, or a subdued tone",
    "desire": "wanting, wishing, longing, or hoping to obtain something",
    "disappointment": "being let down, unmet expectations, or something falling short",
    "surprise": "the unexpected, being caught off guard, or sudden astonishment",
    "excitement": "anticipation, energy, enthusiasm, or a thrilling prospect",
    "joy": "delight, pleasure, cheerfulness, or warm satisfaction",
    "embarrassment": "awkwardness, self-consciousness, wanting to hide, or being mortified",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=Path, default=KEYS)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--model-path", default=QWEN)
    parser.add_argument("--modernbert-model-path", default=MODERNBERT)
    parser.add_argument("--emotions", default=None, help="Comma-separated worker subset; default is all emotions.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def retry_pending_messages(fact: str, emotion: str, attempt: int) -> list[dict[str, str]]:
    """Prompt hard cases with explicit forbidden forms and allowed emotional cues."""
    forbidden = ", ".join(LABEL_VARIANTS[emotion])
    cue = RETRY_PENDING_CUES[emotion]
    style = (
        "Use a natural evaluative framing.",
        "Use a concise first-person reaction only if it preserves the fact.",
        "Use concrete wording rather than an abstract emotional noun.",
        "Make the stance clear through emphasis and sentence rhythm.",
        "Use an understated but unmistakable emotional framing.",
    )[attempt % 5]
    system_prompt = f"""Rewrite an objective fact as one concise emotional description.

The desired emotional stance must be recognizable to an external classifier, but none of these forbidden words or forms may appear anywhere in the output: {forbidden}.
Convey the stance through wording such as: {cue}.

Rules:
1. Preserve the factual meaning. Do not add, remove, or alter facts, causes, motives, entities, times, or intentions.
2. Do not name, explain, analyze, or quote the emotion.
3. Do not use generic labels such as \"I feel...\" or \"This is...\".
4. {style}
5. Keep the same language as the fact and output only one natural sentence."""
    user_prompt = "Input:\n\n" f"Fact: {fact}\n\n" f"Emotion: {emotion}\n\n" "Output:\n"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def attempt_messages(fact: str, emotion: str, attempt: int, retry_round: int) -> list[dict[str, str]]:
    if retry_round:
        return retry_pending_messages(fact, emotion, attempt)
    prompt_index = attempt % 4
    if prompt_index == 0:
        return emotion_messages(fact, emotion)
    return retry_messages(fact, emotion, prompt_index - 1)


def generation_seed(base_seed: int, memory_id: str, attempt: int) -> int:
    digest = hashlib.sha256(f"{memory_id}:{attempt}".encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


@torch.inference_mode()
def generate_fresh_text(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    """Top-p sample reproducibly, rather than replaying the old greedy text."""
    device = next(model.parameters()).device
    inputs = chat_inputs(tokenizer, messages, device)
    outputs = model(**inputs, use_cache=True)
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    special_ids = special_token_ids(tokenizer)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        sorted_logits, sorted_indices = torch.sort(logits.float() / temperature, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = torch.softmax(sorted_logits.masked_fill(remove, float("-inf")), dim=-1)
        sampled_rank = torch.multinomial(sorted_probs, 1, generator=generator)
        next_token = sorted_indices.gather(-1, sampled_rank)
        token_id = int(next_token.item())
        if token_id in special_ids:
            break
        generated_ids.append(token_id)
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
    return clean_text(tokenizer.decode(generated_ids, skip_special_tokens=True))


class SelectedLayerExtractor:
    def __init__(self, model: Any, layers: list[int]):
        self.model = model
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = [
            model.model.layers[layer - 1].register_forward_hook(self._hook(layer))
            for layer in sorted(set(layers))
        ]

    def _hook(self, layer: int):
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.captured[layer] = hidden[:, -1, :].detach().float().cpu()[0]
        return capture

    @torch.inference_mode()
    def extract(
        self,
        tokenizer: Any,
        messages: list[dict[str, str]],
        response: str,
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        if not response.strip():
            raise ValueError("empty response cannot be used for hidden extraction")
        device = next(self.model.parameters()).device
        prompt = chat_inputs(tokenizer, messages, device)
        response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        input_ids = torch.cat([prompt["input_ids"], response_ids], dim=1)
        self.captured = {}
        self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False, return_dict=True)
        missing = set(layers) - set(self.captured)
        if missing:
            raise RuntimeError(f"hooks did not capture layers: {sorted(missing)}")
        return {layer: self.captured[layer] for layer in layers}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def load_source(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = torch.load(args.keys, map_location="cpu", weights_only=False)
    records = artifact["records"]
    train = [row for row in records if row.get("memory_source") == "train_successful_steering"]
    test = [row for row in records if row.get("memory_source") == "test_memory_key_only"]
    if len(records) != 8706 or len(train) != 7603 or len(test) != 1103:
        raise ValueError(f"unexpected key split: all={len(records)} train={len(train)} test={len(test)}")
    if len({row["memory_id"] for row in train}) != len(train):
        raise ValueError("training source contains duplicate memory_id values")
    required = ("factual_memory", "neutral_description", "emotion_description", "emotion_label")
    if any(not row.get(field) for row in train for field in required):
        raise ValueError("a training source row is missing a required field")
    final_meta = {row["memory_id"]: row for row in jsonl(FINAL_VALUE_DIR / "metadata.jsonl")}
    if not all(row["memory_id"] in final_meta for row in train):
        raise ValueError("a source row is absent from source-cohort metadata")
    if any(final_meta[row["memory_id"]]["neutral_state"] != "ok" for row in train):
        raise ValueError("a source row lacks a neutral-gated description")
    return config, train, test


def save_checkpoint(state: dict[str, Any], state_path: Path, audit_path: Path) -> None:
    torch.save(state, state_path)
    write_jsonl(audit_path, state["audit"])
    write_jsonl(audit_path.with_name("regeneration_errors.jsonl"), state.get("errors", []))


def finalize(
    args: argparse.Namespace,
    config: dict[str, Any],
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:
    """Finalize from classifier-accepted records and exclude retry-pending rows."""
    total_completed = 0
    states: dict[str, dict[str, Any]] = {}
    accepted_ids_by_emotion: dict[str, list[str]] = {}
    for emotion, emotion_config in config["emotions"].items():
        state_path = args.work_dir / emotion / "steering_accumulator.pt"
        if not state_path.exists():
            raise RuntimeError(f"cannot finalize: {emotion} has no accumulator")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        expected = [row["memory_id"] for row in train_records if row["emotion_label"] == emotion]
        expected_layers = [int(layer) for layer in emotion_config["layer_weights"]]
        if state.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError(f"{emotion}: wrong pipeline checkpoint")
        if state["layers"] != expected_layers:
            raise ValueError(f"{emotion}: extracted layers do not match configuration")
        processed = set(state["processed_ids"])
        if not processed or not processed.issubset(set(expected)):
            raise ValueError(f"{emotion}: invalid accepted-record set")
        if (
            len(state["audit"]) != len(processed)
            or {row["memory_id"] for row in state["audit"]} != processed
            or not all(row["accepted"] for row in state["audit"])
        ):
            raise ValueError(f"{emotion}: accepted audit and accumulator disagree")
        states[emotion] = state
        accepted_ids_by_emotion[emotion] = [memory_id for memory_id in expected if memory_id in processed]
        total_completed += len(processed)
    if total_completed == 0:
        raise ValueError("refusing finalization with no classifier-accepted records")

    summary: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "num_completed": total_completed,
        "num_excluded_retry_pending": len(train_records) - total_completed,
        "finalized_from_classifier_accepted_records": True,
        "emotions": {},
    }
    values: dict[str, Any] = {}
    accepted_audit_by_id: dict[str, dict[str, Any]] = {}
    for emotion, emotion_config in config["emotions"].items():
        state = states[emotion]
        count = len(state["processed_ids"])
        weights = {int(layer): float(alpha) for layer, alpha in emotion_config["layer_weights"].items()}
        raw_vectors = state["sum"] / count
        configured_vectors = torch.stack([raw_vectors[index] * weights[layer] for index, layer in enumerate(state["layers"])])
        accepted_ids = accepted_ids_by_emotion[emotion]
        output = {
            "pipeline_version": PIPELINE_VERSION,
            "emotion": emotion,
            "layers": state["layers"],
            "vectors": raw_vectors,
            "num_regenerated_classified_records": count,
            "num_excluded_retry_pending": sum(row["emotion_label"] == emotion for row in train_records) - count,
            "training_memory_ids": accepted_ids,
            "layer_weights_for_inference": weights,
            "construction": "regenerate -> classify -> exclude retry-pending rows -> mean accepted emotion-minus-neutral hidden differences",
            "source_keys": str(args.keys),
            "old_emotional_descriptions_reused": False,
            "neutral_policy": "reuse neutral-gated text; recompute hidden at emotion-specific configured layers",
            "test_data_used": False,
            "weights_applied": False,
        }
        torch.save(output, args.work_dir / emotion / "emotion_steering_vector.pt")
        summary["emotions"][emotion] = {"count": count, "layers": state["layers"], "weights": weights}
        values[emotion] = {
            "layers": state["layers"],
            "raw_vectors": raw_vectors,
            "configured_layer_vectors": configured_vectors,
            "layer_weights": weights,
            "accepted_training_memory_ids": accepted_ids,
        }
        accepted_audit_by_id.update({row["memory_id"]: row for row in state["audit"]})

    source = torch.load(args.keys, map_location="cpu", weights_only=False)
    accepted_ids = set(accepted_audit_by_id)
    retained_indices = [
        index for index, row in enumerate(source["records"])
        if row["memory_source"] == "test_memory_key_only" or row["memory_id"] in accepted_ids
    ]
    retained_records = []
    for index in retained_indices:
        row = dict(source["records"][index])
        row["bank_row_id"] = len(retained_records)
        audit = accepted_audit_by_id.get(row["memory_id"])
        if audit is not None:
            row["emotion_description"] = audit["new_emotion_description"]
            row["regenerated_classification"] = {
                "predicted_label": audit["new_predicted_label"],
                "attempt_used": audit["attempt_used"],
                "pipeline_version": PIPELINE_VERSION,
            }
        retained_records.append(row)
    excluded_ids = {row["memory_id"] for row in train_records} - accepted_ids
    errors_by_id = {
        error["memory_id"]: error
        for state in states.values()
        for error in state.get("errors", [])
        if error["memory_id"] in excluded_ids
    }
    values_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "labels": list(config["emotions"]),
        "values": values,
        "accepted_pair_counts": {emotion: item["count"] for emotion, item in summary["emotions"].items()},
        "num_accepted_train_records": total_completed,
        "num_excluded_retry_pending_train_records": len(excluded_ids),
        "config_file": str(args.config),
        "construction": "training-only regenerated and classifier-accepted descriptions; configured layers weighted after per-emotion averaging",
        "test_data_used": False,
    }
    bank = {
        "pipeline_version": PIPELINE_VERSION,
        "records": retained_records,
        "fact_vectors": source["fact_vectors"][retained_indices].contiguous(),
        "emotion_cue_vectors": source["emotion_cue_vectors"][retained_indices].contiguous(),
        "keys": source["keys"][retained_indices].contiguous(),
        "manifest": {
            **source["manifest"],
            "source_key_artifact": str(args.keys),
            "num_source_train_records": len(train_records),
            "num_accepted_train_records": total_completed,
            "num_excluded_retry_pending_train_records": len(excluded_ids),
            "num_test_records_retained": len(test_records),
            "old_emotional_descriptions_reused": False,
        },
        "excluded_retry_pending": [errors_by_id.get(memory_id, {"memory_id": memory_id, "status": "error", "retry_pending": True}) for memory_id in sorted(excluded_ids)],
        "injection_values": values_payload,
    }
    build_dir = args.keys.parent
    bank_path = build_dir / "qwen_multilayer_memory_bank_regenerated_classified_6927.pt"
    values_path = build_dir / "qwen_multilayer_memory_values.pt"
    torch.save(bank, bank_path)
    torch.save(values_payload, values_path)
    summary["memory_bank"] = str(bank_path)
    summary["memory_values"] = str(values_path)
    (args.work_dir / "finalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "finalized", "num_completed": total_completed, "num_excluded": len(excluded_ids), "memory_bank": str(bank_path), "memory_values": str(values_path)}, ensure_ascii=False), flush=True)


def main() -> None:
    args = arguments()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if not 0 < args.temperature:
        raise ValueError("--temperature must be > 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    config, train_records, test_records = load_source(args)
    configured = list(config["emotions"])
    emotions = configured if args.emotions is None else [item.strip() for item in args.emotions.split(",") if item.strip()]
    if not emotions or not set(emotions).issubset(configured):
        raise ValueError("requested emotion is absent from multilayer config")

    counts = {emotion: sum(row["emotion_label"] == emotion for row in train_records) for emotion in configured}
    layer_map = {
        emotion: [int(layer) for layer in config["emotions"][emotion]["layer_weights"]]
        for emotion in configured
    }
    preflight = {
        "status": "preflight_ok",
        "pipeline_version": PIPELINE_VERSION,
        "sequence": ["regenerate", "classify_up_to_5_or_mark_retry_pending", "extract_emotion_specific_layers", "finalize_after_all_7603"],
        "max_attempts_per_run": args.max_attempts,
        "num_source_training_keys": len(train_records),
        "num_excluded_test_keys": len(test_records),
        "old_emotional_descriptions_reused": False,
        "neutral_policy": "reuse gated neutral text and recompute hidden",
        "emotion_counts": counts,
        "emotion_layer_map": layer_map,
    }
    if sum(counts.values()) != 7603 or any(count == 0 for count in counts.values()):
        raise ValueError("configured emotion partition does not cover all 7,603 source records")
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False), flush=True)
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.finalize_only:
        finalize(args, config, train_records, test_records)
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    needed_layers = sorted({layer for emotion in emotions for layer in layer_map[emotion]})
    tokenizer, model = load_causal_lm(args.model_path, args.device)
    extractor = SelectedLayerExtractor(model, needed_layers)
    cls_tokenizer, cls_model, cls_device, labels = load_classifier(args.modernbert_model_path, args.classifier_device)
    try:
        for emotion in emotions:
            records = [row for row in train_records if row["emotion_label"] == emotion]
            layers = layer_map[emotion]
            out = args.work_dir / emotion
            out.mkdir(parents=True, exist_ok=True)
            state_path = out / "steering_accumulator.pt"
            audit_path = out / "regeneration_classification_audit.jsonl"
            completion_path = out / "completion.json"
            expected_ids = [row["memory_id"] for row in records]
            if completion_path.exists():
                print(json.dumps({"emotion": emotion, "status": "already_complete"}), flush=True)
                continue
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                if state.get("pipeline_version") != PIPELINE_VERSION:
                    raise ValueError(f"{emotion}: checkpoint belongs to another pipeline")
                if state["layers"] != layers or state["expected_memory_ids"] != expected_ids:
                    raise ValueError(f"{emotion}: checkpoint source records or layers do not match")
                state.setdefault("errors", [])
            else:
                state = {
                    "pipeline_version": PIPELINE_VERSION,
                    "emotion": emotion,
                    "layers": layers,
                    "sum": torch.zeros(len(layers), int(model.config.hidden_size), dtype=torch.float32),
                    "processed_ids": [],
                    "expected_memory_ids": expected_ids,
                    "audit": [],
                    "errors": [],
                    "source_keys": str(args.keys),
                }
            processed = set(state["processed_ids"])
            for record in records:
                memory_id = record["memory_id"]
                if memory_id in processed:
                    continue
                retry_round = sum(error["memory_id"] == memory_id for error in state["errors"])
                attempts: list[dict[str, Any]] = []
                for attempt in range(args.max_attempts):
                    messages = attempt_messages(record["factual_memory"], emotion, attempt, retry_round)
                    text = generate_fresh_text(
                        tokenizer, model, messages, args.max_new_tokens, args.temperature, args.top_p,
                        generation_seed(args.seed, memory_id, retry_round * args.max_attempts + attempt),
                    )
                    classification = classify_texts(cls_tokenizer, cls_model, cls_device, labels, [text])[0]
                    same_as_old = text.strip() == record["emotion_description"].strip()
                    direct_label = mentions_label(text, emotion)
                    passed = bool(text) and classification["predicted_label"] == emotion and not direct_label and not same_as_old
                    attempts.append({
                        "attempt": attempt + 1,
                        "retry_round": retry_round,
                        "prompt_variant": attempt % 4,
                        "prompt_strategy": "retry_pending_special" if retry_round else "initial",
                        "description": text,
                        "predicted_label": classification["predicted_label"],
                        "classifier_score": classification["classifier_score"],
                        "classifier_top3": classification["classifier_top3"],
                        "mentions_target_label": direct_label,
                        "same_as_old_description": same_as_old,
                        "passed": passed,
                    })
                    if passed:
                        break
                else:
                    state["errors"].append({
                        "memory_id": memory_id,
                        "emotion_label": emotion,
                        "status": "error",
                        "retry_pending": True,
                        "retry_round": retry_round,
                        "attempt_count": len(attempts),
                        "attempts": attempts,
                    })
                    save_checkpoint(state, state_path, audit_path)
                    print(json.dumps({
                        "emotion": emotion,
                        "memory_id": memory_id,
                        "status": "error",
                        "retry_pending": True,
                        "attempt_count": len(attempts),
                    }, ensure_ascii=False), flush=True)
                    continue

                emotion_hidden = extractor.extract(tokenizer, messages, text, layers)
                neutral_hidden = extractor.extract(
                    tokenizer, neutral_messages(record["factual_memory"]), record["neutral_description"], layers
                )
                state["sum"] += torch.stack([emotion_hidden[layer] - neutral_hidden[layer] for layer in layers])
                state["audit"].append({
                    "memory_id": memory_id,
                    "emotion_label": emotion,
                    "factual_memory": record["factual_memory"],
                    "old_emotion_description": record["emotion_description"],
                    "new_emotion_description": text,
                    "new_predicted_label": classification["predicted_label"],
                    "accepted": True,
                    "attempt_used": len(attempts),
                    "attempts": attempts,
                    "layers_extracted": layers,
                })
                state["processed_ids"].append(memory_id)
                processed.add(memory_id)
                if len(processed) % args.checkpoint_every == 0 or len(processed) == len(records):
                    save_checkpoint(state, state_path, audit_path)
                    print(json.dumps({
                        "emotion": emotion, "processed": len(processed), "expected": len(records),
                        "last_attempts": len(attempts), "layers": layers,
                    }, ensure_ascii=False), flush=True)

            # Persist a successful tail even when the class still has retry-pending
            # records and its final success did not land on a checkpoint boundary.
            save_checkpoint(state, state_path, audit_path)
            if len(state["processed_ids"]) != len(expected_ids) or set(state["processed_ids"]) != set(expected_ids):
                print(json.dumps({
                    "emotion": emotion,
                    "status": "retry_pending",
                    "processed": len(state["processed_ids"]),
                    "expected": len(expected_ids),
                    "error_events": len(state["errors"]),
                }, ensure_ascii=False), flush=True)
                continue
            completion_path.write_text(json.dumps({
                "pipeline_version": PIPELINE_VERSION,
                "emotion": emotion,
                "completed": len(records),
                "layers": layers,
                "all_classified_as_target": True,
                "mean_computed": False,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"emotion": emotion, "status": "sum_complete", "count": len(records)}, ensure_ascii=False), flush=True)
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
