#!/usr/bin/env python3
"""Build the retrieval-key half of the 11-emotion memory bank only.

No Qwen hidden state or injection value is read or written here.  Values are
constructed separately from training-only ModernBERT-gated pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "use/build/multilayer_11class_trainplus_test_memory/qwen_multilayer_memory_keys.pt"
CONFIG = ROOT / "use/build/multilayer_11class_trainplus_test_memory/multilayer_injection_config.json"
TRAIN_BANK = ROOT / "data/build/all/train/qwen_memory_bank.pt"
TEST_FACT = ROOT / "data/test_build/fact/goemotions_test_factual_qwen3_embedding_0.6b_fp16.pt"
TEST_FACT_META = ROOT / "data/test_build/fact/goemotions_test_factual_qwen3_embedding_0.6b_metadata.jsonl"
TEST_CUE = ROOT / "data/test_build/emotion_cue/goemotions_train_original_text_modernbert_goemotions_layer22_masked_mean_fp16.pt"
TEST_CUE_META = ROOT / "data/test_build/emotion_cue/goemotions_train_original_text_modernbert_goemotions_layer22_metadata.jsonl"


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    labels = list(json.loads(CONFIG.read_text(encoding="utf-8"))["emotions"])
    selected = set(labels)
    train = torch.load(TRAIN_BANK, map_location="cpu", weights_only=False)
    train_indices = [record["bank_row_id"] for record in train["records"] if record["emotion_label"] in selected]
    train_records = [record for record in train["records"] if record["emotion_label"] in selected]
    train_keys = train["keys"][torch.tensor(train_indices)].float()
    fact, cue = torch.load(TEST_FACT, map_location="cpu", weights_only=True).float(), torch.load(TEST_CUE, map_location="cpu", weights_only=True).float()
    fact_meta, cue_meta = rows(TEST_FACT_META), rows(TEST_CUE_META)
    assert len(fact) == len(cue) == len(fact_meta) == len(cue_meta) == 2540
    test_indices = [i for i, row in enumerate(fact_meta) if row["emotion_label"] in selected]
    test_fact, test_cue = F.normalize(fact[test_indices], p=2, dim=1), F.normalize(cue[test_indices], p=2, dim=1)
    fact_vectors = torch.cat([train_keys[:, :1024], test_fact], dim=0).to(torch.float16)
    cue_vectors = torch.cat([train_keys[:, 1024:], test_cue], dim=0).to(torch.float16)
    records = []
    for bank_row_id, record in enumerate(train_records):
        records.append({**record, "bank_row_id": bank_row_id, "memory_source": "train_successful_steering"})
    for bank_row_id, i in enumerate(test_indices, start=len(records)):
        f, c = fact_meta[i], cue_meta[i]
        records.append({"bank_row_id": bank_row_id, "memory_id": f["memory_id"], "source_id": f["source_id"], "source_row": f["source_row"], "emotion_id": f["emotion_id"], "emotion_label": f["emotion_label"], "factual_memory": f["factual_memory"], "original_text": c["original_text"], "memory_source": "test_memory_key_only", "fact_row_id": f["row_id"], "emotion_cue_row_id": c["row_id"]})
    keys = torch.cat([fact_vectors, cue_vectors], dim=1)
    assert len(records) == len(keys) == 8706
    torch.save({"fact_vectors": fact_vectors, "emotion_cue_vectors": cue_vectors, "keys": keys, "records": records, "manifest": {"labels": labels, "num_rows": len(records), "train_successful_memory_rows": len(train_records), "test_memory_key_rows": len(test_indices), "key_method": "concat(L2-normalized fact embedding, L2-normalized emotion cue embedding)", "value_status": "not constructed; no injection vector is present in this file"}}, OUT)
    print(json.dumps({"output": str(OUT), "fact_shape": list(fact_vectors.shape), "cue_shape": list(cue_vectors.shape), "key_shape": list(keys.shape), "train": len(train_records), "test": len(test_indices)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
