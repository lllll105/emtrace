#!/usr/bin/env python3
"""Build the 11-emotion memory bank with train-only multilayer values.

The retrieval corpus combines successful training memories with test memory
facts/cues.  Test rows contribute keys only: every injected value is one of
the 11 prototypes extracted exclusively from the corresponding training-set
steering-vector artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
TRAIN_BANK = ROOT / "data/build/all/train/qwen_memory_bank.pt"
TEST_FACT = ROOT / "data/test_build/fact/goemotions_test_factual_qwen3_embedding_0.6b_fp16.pt"
TEST_FACT_META = ROOT / "data/test_build/fact/goemotions_test_factual_qwen3_embedding_0.6b_metadata.jsonl"
TEST_CUE = ROOT / "data/test_build/emotion_cue/goemotions_train_original_text_modernbert_goemotions_layer22_masked_mean_fp16.pt"
TEST_CUE_META = ROOT / "data/test_build/emotion_cue/goemotions_train_original_text_modernbert_goemotions_layer22_metadata.jsonl"
OUTPUT = ROOT / "use/build/multilayer_11class_trainplus_test_memory"

# Layer numbers are one-based Qwen transformer-block numbers.  The values
# below are the selected per-layer injection strengths, not a later global
# alpha.  Only the 11 selected classes exist in the resulting bank.
SPECS: dict[str, dict[str, Any]] = {
    "admiration": {"dir": "admiration1", "weights": {20: 0.75, 21: 0.75}, "status": "confirmed"},
    "amusement": {"dir": "amusement5", "weights": {20: 1.25}, "status": "confirmed"},
    "curiosity": {"dir": "curiosity6", "weights": {20: 1.0}, "status": "confirmed"},
    "joy": {"dir": "joy11", "weights": {20: 1.0}, "status": "confirmed"},
    "confusion": {"dir": "confusion12", "weights": {22: 0.6667, 23: 0.6667, 24: 0.6667}, "status": "balanced"},
    "sadness": {"dir": "sadness13", "weights": {21: 0.375, 22: 1.125}, "status": "confirmed"},
    "disappointment": {"dir": "disappointment14", "weights": {23: 2.5}, "status": "confirmed"},
    "surprise": {"dir": "surprise17", "weights": {20: 0.375, 22: 1.125}, "status": "balanced"},
    "excitement": {"dir": "excitement18", "weights": {19: 0.25, 20: 0.75, 22: 0.25}, "status": "balanced"},
    "desire": {"dir": "desire20", "weights": {21: 0.375, 22: 0.375, 23: 0.75}, "status": "provisional"},
    "embarrassment": {"dir": "embarrassment23", "weights": {21: 0.6375, 22: 1.0625}, "status": "provisional"},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def build_prototypes() -> tuple[list[str], torch.Tensor, list[dict[str, Any]]]:
    labels = list(SPECS)
    tensors: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for label in labels:
        spec = SPECS[label]
        source = ROOT / "ex/emotion_layer_every" / spec["dir"] / "steering_vector.pt"
        artifact = torch.load(source, map_location="cpu", weights_only=True)
        raw = artifact["vector"].float()
        assert raw.shape == (28, 3584), (label, raw.shape)
        weighted = torch.zeros_like(raw)
        for layer, alpha in spec["weights"].items():
            weighted[layer - 1] = raw[layer - 1] * alpha
        tensors.append(weighted)
        metadata.append({
            "value_label_index": len(metadata), "emotion_label": label,
            "source": str(source), "source_num_pairs": artifact["num_pairs"],
            "layer_weights": {str(layer): alpha for layer, alpha in spec["weights"].items()},
            "total_alpha": sum(spec["weights"].values()), "selection_status": spec["status"],
            "value_construction": "training-only steering_vector[layer-1] * selected layer weight",
        })
    return labels, torch.stack(tensors), metadata


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels, prototypes, prototype_meta = build_prototypes()
    label_to_index = {label: index for index, label in enumerate(labels)}

    train = torch.load(TRAIN_BANK, map_location="cpu", weights_only=False)
    selected_train = [record for record in train["records"] if record["emotion_label"] in label_to_index]
    selected_train_indices = torch.tensor([record["bank_row_id"] for record in selected_train])
    train_keys = train["keys"][selected_train_indices].float()
    assert train_keys.shape[1] == 1792

    fact = torch.load(TEST_FACT, map_location="cpu", weights_only=True).float()
    cue = torch.load(TEST_CUE, map_location="cpu", weights_only=True).float()
    fact_meta, cue_meta = read_jsonl(TEST_FACT_META), read_jsonl(TEST_CUE_META)
    assert len(fact) == len(cue) == len(fact_meta) == len(cue_meta) == 2540
    assert all(f["memory_id"] == c["memory_id"] for f, c in zip(fact_meta, cue_meta))
    test_indices = [index for index, row in enumerate(fact_meta) if row["emotion_label"] in label_to_index]
    test_fact = F.normalize(fact[test_indices], p=2, dim=1)
    test_cue = F.normalize(cue[test_indices], p=2, dim=1)
    test_keys = torch.cat([test_fact, test_cue], dim=1)
    assert test_keys.shape == (1103, 1792)

    records: list[dict[str, Any]] = []
    for bank_row_id, record in enumerate(selected_train):
        records.append({
            **record,
            "bank_row_id": bank_row_id,
            "memory_source": "train_successful_steering",
            "value_label_index": label_to_index[record["emotion_label"]],
            "value_source": "train_only_multilayer_prototype",
        })
    for offset, source_index in enumerate(test_indices, start=len(records)):
        fact_row, cue_row = fact_meta[source_index], cue_meta[source_index]
        label = fact_row["emotion_label"]
        records.append({
            "bank_row_id": offset,
            "memory_id": fact_row["memory_id"],
            "source_id": fact_row["source_id"],
            "source_row": fact_row["source_row"],
            "emotion_id": fact_row["emotion_id"],
            "emotion_label": label,
            "factual_memory": fact_row["factual_memory"],
            "original_text": cue_row["original_text"],
            "memory_source": "test_memory_key_only",
            "fact_row_id": fact_row["row_id"],
            "emotion_cue_row_id": cue_row["row_id"],
            "value_label_index": label_to_index[label],
            "value_source": "train_only_multilayer_prototype",
        })
    keys = torch.cat([train_keys, test_keys], dim=0).to(torch.float16)
    assert len(records) == len(keys) == 8706
    assert {record["memory_source"] for record in records} == {"train_successful_steering", "test_memory_key_only"}

    bank = {
        # The three vector components of this memory bank.  `keys` is kept as
        # the retrieval-ready concatenation of the first two components.
        "fact_vectors": keys[:, :1024].contiguous(),
        "emotion_cue_vectors": keys[:, 1024:].contiguous(),
        "injection_value_prototypes": prototypes,
        "injection_value_index": torch.tensor(
            [record["value_label_index"] for record in records], dtype=torch.int64
        ),
        "keys": keys,
        "value_labels": labels,
        "records": records,
        "manifest": {
            "num_rows": len(records), "key_shape": list(keys.shape),
            "fact_shape": list(keys[:, :1024].shape),
            "emotion_cue_shape": list(keys[:, 1024:].shape),
            "injection_value_prototype_shape": list(prototypes.shape),
            "key_method": "concat(L2-normalized fact embedding, L2-normalized emotion cue embedding)",
            "value_method": "11 train-only per-emotion multilayer steering prototypes",
            "test_value_policy": "test contributes fact/cue keys only; no test hidden state, steering vector, success filter, or averaging is used",
        },
    }
    torch.save(bank, OUTPUT / "qwen_multilayer_memory_bank.pt")
    print(json.dumps({"output": str(OUTPUT), "train_rows": len(selected_train), "test_rows": len(test_indices), "total_rows": len(records), "labels": labels}, ensure_ascii=False))


if __name__ == "__main__":
    main()
