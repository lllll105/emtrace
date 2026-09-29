#!/usr/bin/env python3
"""Compare Stage-2 Full-memory and train-only m=2 prototype decisions.

No retrieval or text generation is modified: saved strict query vectors are
used once to reproduce the identical all-8,030 Stage-1 Top-1 result, and its
retrieved cue is scored against both prototype banks.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[4]
PLUS = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt"
FULL = ROOT / "use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt"
QUERY = ROOT / "use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt"
META = ROOT / "use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl"


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def score_classes(cues: torch.Tensor, bank: dict[str, Any], labels: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prototype_scores = cues @ bank["emotion_prototype_keys"].float().T
    class_scores = torch.full((len(cues), len(labels)), float("-inf"))
    for pidx, cidx in enumerate(bank["prototype_class_indices"].tolist()):
        class_scores[:, cidx] = torch.maximum(class_scores[:, cidx], prototype_scores[:, pidx])
    values, indices = class_scores.topk(len(labels), dim=1)
    return indices, values, class_scores


def metrics(rows: list[dict[str, Any]], version: str, labels: list[str]) -> dict[str, Any]:
    pred_key = "stage2_emotion_full" if version == "full_memory" else "stage2_emotion_train_only"
    n = len(rows)
    event_correct = [bool(r["memory_exact_top1"]) for r in rows]
    affect_correct = [r[pred_key] == r["target_emotion"] for r in rows]
    per_class = []
    for label in labels:
        group = [x for x in rows if x["target_emotion"] == label]
        per_class.append(sum(x[pred_key] == label for x in group) / len(group))
    cells = {
        "event_correct_affect_correct_n": sum(e and a for e, a in zip(event_correct, affect_correct)),
        "event_correct_affect_wrong_n": sum(e and not a for e, a in zip(event_correct, affect_correct)),
        "event_wrong_affect_correct_n": sum(not e and a for e, a in zip(event_correct, affect_correct)),
        "event_wrong_affect_wrong_n": sum(not e and not a for e, a in zip(event_correct, affect_correct)),
    }
    # The four-cell correction/corruption diagnostic is deliberately based on
    # Stage-1 *emotion* correctness (not exact event recall).  This matches
    # the protocol: did the prototype stage preserve, corrupt, or repair the
    # emotion label carried by the retrieved memory?
    affect_transition = Counter((int(r["stage1_correct"]), int(r[pred_key] == r["target_emotion"])) for r in rows)
    keep_correct = affect_transition[1, 1]
    corrupt = affect_transition[1, 0]
    correct = affect_transition[0, 1]
    remain_wrong = affect_transition[0, 0]
    return {
        "version": version, "n": n,
        "exact_recall_at1": sum(event_correct) / n,
        "affect_recovery_accuracy": sum(affect_correct) / n,
        "binding_at1": cells["event_correct_affect_correct_n"] / n,
        "uar_macro_recall_11class": sum(per_class) / len(per_class),
        "stage1_emotion_accuracy": sum(r["stage1_correct"] for r in rows) / n,
        **cells,
        "stage1_correct_to_stage2_correct_n": keep_correct,
        "stage1_correct_to_stage2_wrong_n": corrupt,
        "stage1_wrong_to_stage2_correct_n": correct,
        "stage1_wrong_to_stage2_wrong_n": remain_wrong,
        "correction_rate": correct / max(1, correct + remain_wrong),
        "corruption_rate": corrupt / max(1, keep_correct + corrupt),
        "stage2_label_change_rate_vs_full": 0.0 if version == "full_memory" else sum(r["stage2_label_changed"] for r in rows) / n,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    default = PLUS / "results/train_only_prototype_sensitivity"
    p.add_argument("--output-dir", type=Path, default=default)
    p.add_argument("--stage1-bank", type=Path, default=STAGE1)
    p.add_argument("--full-stage2-bank", type=Path, default=FULL)
    p.add_argument("--train-only-stage2-bank", type=Path, default=default / "train_only_prototypes/stage2_emotion_multicenter_to_injection_bank.pt")
    p.add_argument("--query-vectors", type=Path, default=QUERY)
    p.add_argument("--query-metadata", type=Path, default=META)
    a = p.parse_args(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)

    stage1 = torch.load(a.stage1_bank, map_location="cpu", weights_only=False)
    full = torch.load(a.full_stage2_bank, map_location="cpu", weights_only=False)
    train = torch.load(a.train_only_stage2_bank, map_location="cpu", weights_only=False)
    query = torch.load(a.query_vectors, map_location="cpu", weights_only=True).float(); meta = jsonl(a.query_metadata)
    labels = list(full["manifest"]["labels"])
    if len(meta) != 1103 or query.shape != (1103, 1024): raise ValueError("expected 1,103 saved strict queries of dimension 1,024")
    if list(train["manifest"]["labels"]) != labels or tuple(full["emotion_prototype_keys"].shape) != (22, 768) or tuple(train["emotion_prototype_keys"].shape) != (22, 768):
        raise ValueError("prototype banks must both be compatible m=2 11-class banks")

    stage1_score, stage1_idx = (query @ stage1["fact_keys"].float().T).max(1)
    cues = stage1["emotion_cue_values"].float()[stage1_idx]
    full_rank, full_values, _ = score_classes(cues, full, labels)
    train_rank, train_values, _ = score_classes(cues, train, labels)

    rows: list[dict[str, Any]] = []
    for i, m in enumerate(meta):
        r = stage1["records"][int(stage1_idx[i])]
        target = m["target_emotion"]; full_label = labels[int(full_rank[i, 0])]; train_label = labels[int(train_rank[i, 0])]
        rows.append({
            "query_row_id": i, "memory_id": m["memory_id"], "target_memory_id": m["memory_id"],
            "target_emotion": target, "retrieved_memory_id": r["memory_id"],
            "retrieved_memory_source": r["memory_source"], "stage1_emotion": r["emotion_label"],
            "memory_exact_top1": int(r["memory_id"] == m["memory_id"]),
            "stage1_correct": int(r["emotion_label"] == target),
            "stage1_fact_cosine_similarity": float(stage1_score[i]),
            "stage2_emotion_full": full_label, "stage2_full_correct": int(full_label == target),
            "stage2_full_ranked_emotions": [labels[int(x)] for x in full_rank[i].tolist()],
            "stage2_full_ranked_class_scores": [float(x) for x in full_values[i].tolist()],
            "stage2_emotion_train_only": train_label, "stage2_train_only_correct": int(train_label == target),
            "stage2_train_only_ranked_emotions": [labels[int(x)] for x in train_rank[i].tolist()],
            "stage2_train_only_ranked_class_scores": [float(x) for x in train_values[i].tolist()],
            "stage2_label_changed": int(full_label != train_label),
        })
    exact = sum(r["memory_exact_top1"] for r in rows)
    if exact != 790:
        raise AssertionError(f"Stage-1 changed unexpectedly: exact recall count={exact}, expected 790")
    write_jsonl(out / "two_stage_train_only_per_query.jsonl", rows)
    summary = [metrics(rows, "full_memory", labels), metrics(rows, "train_only", labels)]
    fields = list(summary[0])
    with (out / "full_vs_train_only_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)

    transitions: list[dict[str, Any]] = []
    for version, pred_key in (("full_memory", "stage2_emotion_full"), ("train_only", "stage2_emotion_train_only")):
        c = Counter((int(r["stage1_correct"]), int(r[pred_key] == r["target_emotion"])) for r in rows)
        for s1 in (1, 0):
            for s2 in (1, 0):
                n = c[s1, s2]
                transitions.append({"section": "stage1_to_stage2_correctness", "version": version, "stage1_correct": s1, "stage2_correct": s2, "full_label": "", "train_only_label": "", "n": n, "rate_of_all_queries": n / len(rows)})
    changed = Counter((r["stage2_emotion_full"], r["stage2_emotion_train_only"]) for r in rows)
    for (full_label, train_label), n in sorted(changed.items()):
        transitions.append({"section": "full_to_train_only_label", "version": "paired", "stage1_correct": "", "stage2_correct": "", "full_label": full_label, "train_only_label": train_label, "n": n, "rate_of_all_queries": n / len(rows)})
    with (out / "full_vs_train_only_transition.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(transitions[0])); w.writeheader(); w.writerows(transitions)

    report = {
        "protocol": "Saved evaluation_query vectors -> unchanged Stage-1 Top-1 over all 8,030 fact keys -> same retrieved cue scored by full-memory versus train-only m=2 prototype banks. No text generation or injection run.",
        "n": len(rows), "stage1_exact_recall_count": exact, "stage1_exact_recall_at1": exact / len(rows),
        "stage2_label_changed_n": sum(r["stage2_label_changed"] for r in rows),
        "stage2_label_changed_rate": sum(r["stage2_label_changed"] for r in rows) / len(rows),
        "full_bank": str(a.full_stage2_bank.resolve()), "train_only_bank": str(a.train_only_stage2_bank.resolve()),
    }
    (out / "README.md").write_text("# Train-only Two-stage Prototype Sensitivity\n\n" + json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", **report, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
