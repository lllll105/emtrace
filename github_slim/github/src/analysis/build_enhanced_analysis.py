#!/usr/bin/env python3
"""Build the enhanced EM-TRACE Results evidence package."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_results"
DERIVED = ROOT / "derived"
MEMORY_BANK = Path(
    "./use/build/"
    "multilayer_11class_trainplus_test_memory/"
    "qwen_multilayer_memory_bank_regenerated_classified_6927.pt"
)

LABELS = [
    "admiration", "amusement", "curiosity", "joy", "confusion",
    "sadness", "disappointment", "surprise", "excitement",
    "desire", "embarrassment",
]
METHODS = ["fact-only", "emotion-only", "concat-sum", "similarity-product", "two-stage"]
GROUPS = [
    "01_fact_prompt_only",
    "02_memory_retrieved_emotion_label",
    "03_memory_oracle_emotion_label",
    "04_normal_multilayer_steering",
    "05_oracle_multilayer_steering",
    "06_normal_npti",
    "07_oracle_npti",
]
GROUP_NAMES = {
    "01_fact_prompt_only": "Fact prompt only",
    "02_memory_retrieved_emotion_label": "Memory + Retrieved Emotion Label",
    "03_memory_oracle_emotion_label": "Memory + Oracle Emotion Label",
    "04_normal_multilayer_steering": "Normal + Multilayer Steering",
    "05_oracle_multilayer_steering": "Oracle + Multilayer Steering",
    "06_normal_npti": "Normal + NPTI",
    "07_oracle_npti": "Oracle + NPTI",
}
ALIASES = {
    "fact_only": "fact-only",
    "emotion_only": "emotion-only",
    "concat_sum": "concat-sum",
    "similarity_product": "similarity-product",
    "two_stage": "two-stage",
    "two_stage_fact_then_cue_prototype": "two-stage",
    "two_stage_fact_then_m2_prototype": "two-stage",
}


def norm_method(value: str | None) -> str:
    value = (value or "").strip()
    return ALIASES.get(value, value.replace("_", "-"))


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"Cannot infer fields for empty output: {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def usable_rows(group: str):
    for row in read_jsonl(SOURCE / group / "per_generation_metrics.jsonl"):
        if row.get("condition") == "baseline":
            continue
        method = norm_method(row.get("retrieval_method") or row.get("condition"))
        if method not in METHODS:
            continue
        yield method, row


def build_dataset_statistics():
    bank = torch.load(MEMORY_BANK, map_location="cpu", weights_only=False)
    records = bank["records"]
    source_counts = Counter(r.get("memory_source") for r in records)
    split_counts = {
        "train": source_counts["train_successful_steering"],
        "test": source_counts["test_memory_key_only"],
    }
    dist = {split: Counter() for split in split_counts}
    for row in records:
        source = row.get("memory_source")
        split = "train" if source == "train_successful_steering" else "test" if source == "test_memory_key_only" else None
        if split:
            dist[split][row.get("emotion_label")] += 1

    rows = []
    for label in LABELS:
        train, test = dist["train"][label], dist["test"][label]
        rows.append({
            "emotion": label, "train": train, "test": test, "total": train + test,
            "train_percent": train / split_counts["train"],
            "test_percent": test / split_counts["test"],
        })
    dest = DERIVED / "dataset_statistics"
    write_csv(dest / "emotion_distribution.csv", rows)
    summary = {
        "memory_bank": str(MEMORY_BANK),
        "total_memories": len(records),
        "train_memories": split_counts["train"],
        "test_memories": split_counts["test"],
        "memory_source_counts": dict(source_counts),
        "labels": LABELS,
        "manifest": bank.get("manifest", {}),
    }
    (dest / "memory_bank_statistics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    md = [
        "# Dataset and Memory-bank Statistics", "",
        f"- Total memory records: **{len(records):,}**",
        f"- Train memories: **{split_counts['train']:,}**",
        f"- Test memories: **{split_counts['test']:,}**",
        f"- Emotion classes: **{len(LABELS)}**", "",
        "| Emotion | Train | Train % | Test | Test % | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['emotion']} | {r['train']} | {r['train_percent']:.2%} | {r['test']} | {r['test_percent']:.2%} | {r['total']} |")
    (dest / "dataset_statistics.md").write_text("\n".join(md) + "\n")


def build_confusion_matrices():
    dest = DERIVED / "confusion_matrix"
    full_matrix_dir = dest / "matrices_full_prediction_labels"
    compact_matrix_dir = dest / "matrices_11class_plus_other"
    full_matrix_dir.mkdir(parents=True, exist_ok=True)
    compact_matrix_dir.mkdir(parents=True, exist_ok=True)
    long_rows, summary_rows = [], []
    representative = ["# Eleven-class Emotion Confusion Matrices", ""]
    all_buckets = {}
    observed_predictions = set()
    for group in GROUPS:
        buckets = defaultdict(list)
        for method, row in usable_rows(group):
            buckets[method].append(row)
            observed_predictions.add(row["modernbert_predicted_label"])
        all_buckets[group] = buckets
    prediction_labels = LABELS + sorted(observed_predictions - set(LABELS))
    for group in GROUPS:
        buckets = all_buckets[group]
        method_stats = []
        for method in METHODS:
            values = buckets[method]
            if len(values) != 1103:
                raise ValueError(f"{group}/{method}: expected 1103 rows, got {len(values)}")
            counts = Counter((r["target_emotion"], r["modernbert_predicted_label"]) for r in values)
            actual_counts = Counter(r["target_emotion"] for r in values)
            correct = sum(counts[(label, label)] for label in LABELS)
            off_diag = [(n, gold, pred) for (gold, pred), n in counts.items() if gold != pred]
            off_diag.sort(reverse=True)
            top = off_diag[:5]
            summary_rows.append({
                "group": group, "group_name": GROUP_NAMES[group], "method": method,
                "n": len(values), "emotion_accuracy": correct / len(values),
                "top_confusions": "; ".join(f"{g}→{p}:{n}" for n, g, p in top),
            })
            method_stats.append((correct / len(values), method, counts))
            full_matrix_rows, compact_matrix_rows = [], []
            for gold in LABELS:
                full_matrix_row = {"gold_emotion": gold}
                compact_matrix_row = {"gold_emotion": gold}
                for pred in prediction_labels:
                    n = counts[(gold, pred)]
                    full_matrix_row[pred] = n
                    long_rows.append({
                        "group": group, "group_name": GROUP_NAMES[group], "method": method,
                        "gold_emotion": gold, "predicted_emotion": pred, "count": n,
                        "row_rate": n / actual_counts[gold] if actual_counts[gold] else 0,
                    })
                for pred in LABELS:
                    compact_matrix_row[pred] = counts[(gold, pred)]
                compact_matrix_row["Other"] = sum(counts[(gold, pred)] for pred in prediction_labels if pred not in LABELS)
                full_matrix_rows.append(full_matrix_row)
                compact_matrix_rows.append(compact_matrix_row)
            write_csv(full_matrix_dir / f"{group}__{method}.csv", full_matrix_rows, ["gold_emotion"] + prediction_labels)
            write_csv(compact_matrix_dir / f"{group}__{method}.csv", compact_matrix_rows, ["gold_emotion"] + LABELS + ["Other"])
        best_acc, best_method, best_counts = max(method_stats)
        representative += [
            f"## {GROUP_NAMES[group]} — best method: {best_method} ({best_acc:.2%})", "",
            "Rows are the 11 gold labels; columns are the 11 target labels plus `Other` for predictions outside the target set.", "",
            "| Gold \\ Pred | " + " | ".join(LABELS + ["Other"]) + " |",
            "|---|" + "---:|" * (len(LABELS) + 1),
        ]
        for gold in LABELS:
            other = sum(best_counts[(gold, pred)] for pred in prediction_labels if pred not in LABELS)
            representative.append("| " + gold + " | " + " | ".join([*(str(best_counts[(gold, pred)]) for pred in LABELS), str(other)]) + " |")
        representative.append("")
    write_csv(dest / "confusion_matrix_long.csv", long_rows)
    write_csv(dest / "confusion_summary.csv", summary_rows)
    (dest / "prediction_labels.json").write_text(json.dumps(prediction_labels, ensure_ascii=False, indent=2) + "\n")
    (dest / "representative_matrices.md").write_text("\n".join(representative) + "\n")


def describe(values: list[float]):
    return {
        "mean": statistics.fmean(values) if values else math.nan,
        "median": statistics.median(values) if values else math.nan,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else math.nan,
        "max": max(values) if values else math.nan,
    }


def build_similarity_analysis():
    groups = defaultdict(lambda: {"fact_score": [], "emotion_score": []})
    for method, row in usable_rows("04_normal_multilayer_steering"):
        recall = "correct" if bool(row.get("memory_exact_top1")) else "wrong"
        groups[(method, recall)]["fact_score"].append(float(row["fact_score"]))
        groups[(method, recall)]["emotion_score"].append(float(row["emotion_score"]))
        groups[("all", recall)]["fact_score"].append(float(row["fact_score"]))
        groups[("all", recall)]["emotion_score"].append(float(row["emotion_score"]))
    rows = []
    for method in METHODS + ["all"]:
        for recall in ("correct", "wrong"):
            data = groups[(method, recall)]
            row = {"method": method, "recall": recall, "n": len(data["fact_score"])}
            for metric in ("fact_score", "emotion_score"):
                for stat, value in describe(data[metric]).items():
                    row[f"{metric}_{stat}"] = value
            rows.append(row)
    dest = DERIVED / "similarity_analysis"
    write_csv(dest / "correct_vs_wrong_similarity.csv", rows)
    contrasts = []
    for method in METHODS + ["all"]:
        c, w = groups[(method, "correct")], groups[(method, "wrong")]
        contrasts.append({
            "method": method,
            "correct_n": len(c["fact_score"]), "wrong_n": len(w["fact_score"]),
            "fact_score_correct_mean": statistics.fmean(c["fact_score"]),
            "fact_score_wrong_mean": statistics.fmean(w["fact_score"]),
            "fact_score_gap": statistics.fmean(c["fact_score"]) - statistics.fmean(w["fact_score"]),
            "emotion_score_correct_mean": statistics.fmean(c["emotion_score"]),
            "emotion_score_wrong_mean": statistics.fmean(w["emotion_score"]),
            "emotion_score_gap": statistics.fmean(c["emotion_score"]) - statistics.fmean(w["emotion_score"]),
        })
    write_csv(dest / "similarity_contrasts.csv", contrasts)
    md = [
        "# Retrieval Similarity: Correct vs Wrong Exact Recall", "",
        "Scores are taken from Normal + Multilayer Steering retrieval records.", "",
        "| Method | Correct n | Wrong n | Fact score (correct) | Fact score (wrong) | Gap | Emotion score (correct) | Emotion score (wrong) | Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in contrasts:
        md.append(
            f"| {r['method']} | {r['correct_n']} | {r['wrong_n']} | {r['fact_score_correct_mean']:.4f} | "
            f"{r['fact_score_wrong_mean']:.4f} | {r['fact_score_gap']:+.4f} | "
            f"{r['emotion_score_correct_mean']:.4f} | {r['emotion_score_wrong_mean']:.4f} | {r['emotion_score_gap']:+.4f} |"
        )
    (dest / "similarity_analysis.md").write_text("\n".join(md) + "\n")


def load_case_indexes():
    indexes = {}
    for group in GROUPS:
        idx = {}
        baseline = {}
        for row in read_jsonl(SOURCE / group / "per_generation_metrics.jsonl"):
            qid = int(row["query_row_id"])
            if row.get("condition") == "baseline":
                baseline[qid] = row
                continue
            method = norm_method(row.get("retrieval_method") or row.get("condition"))
            if method in METHODS:
                idx[(qid, method)] = row
        indexes[group] = {"method": idx, "baseline": baseline}
    return indexes


def choose_diverse(rows, count, score_fn):
    chosen, seen = [], set()
    for row in sorted(rows, key=score_fn, reverse=True):
        label = row["target_emotion"]
        if label in seen:
            continue
        chosen.append(row)
        seen.add(label)
        if len(chosen) == count:
            break
    return chosen


def safe_block(text):
    return str(text or "").replace("```", "'''")


def build_case_studies():
    indexes = load_case_indexes()
    normal = list(usable_rows("04_normal_multilayer_steering"))
    enriched = []
    for method, row in normal:
        x = dict(row)
        x["_method"] = method
        x["_event"] = bool(row.get("memory_exact_top1"))
        x["_affect"] = (row.get("injected_emotion") or row.get("retrieved_emotion")) == row.get("target_emotion")
        enriched.append(x)
    successes = [x for x in enriched if x["_event"] and x["_affect"] and int(x.get("emotion_accuracy", 0)) == 1]
    success = choose_diverse(successes, 3, lambda x: float(x["fact_consistency_score_1"]) + float(x["target_emotion_score"]))
    event_wrong_affect_right = choose_diverse(
        [x for x in enriched if not x["_event"] and x["_affect"]], 1,
        lambda x: float(x["target_emotion_score"]) + float(x["fact_consistency_score_1"]),
    )
    event_right_affect_wrong = choose_diverse(
        [x for x in enriched if x["_event"] and not x["_affect"]], 1,
        lambda x: -float(x["target_emotion_score"]),
    )
    factual_failure = choose_diverse(
        [x for x in enriched if x["_event"] and x["_affect"]], 1,
        lambda x: -float(x["fact_consistency_score_1"]),
    )
    selected = [("success", x) for x in success]
    selected += [("event_wrong_affect_right", x) for x in event_wrong_affect_right]
    selected += [("event_right_affect_wrong", x) for x in event_right_affect_wrong]
    selected += [("fact_consistency_failure", x) for x in factual_failure]

    rows, md = [], ["# EM-TRACE Success and Failure Cases", ""]
    for i, (case_type, row) in enumerate(selected, 1):
        qid, method = int(row["query_row_id"]), row["_method"]
        fact = indexes["01_fact_prompt_only"]["method"].get((qid, method))
        steering = indexes["04_normal_multilayer_steering"]["method"].get((qid, method))
        npti = indexes["06_normal_npti"]["method"].get((qid, method))
        bare = indexes["06_normal_npti"]["baseline"].get(qid)
        if not all((fact, steering, npti, bare)):
            raise KeyError(f"Missing aligned case record qid={qid}, method={method}")
        record = {
            "case_id": i, "case_type": case_type, "query_row_id": qid, "method": method,
            "gold_emotion": row["target_emotion"], "retrieved_emotion": row.get("retrieved_emotion"),
            "injected_emotion": row.get("injected_emotion"), "memory_exact_top1": int(row["_event"]),
            "affect_correct": int(row["_affect"]), "emotion_accuracy": row["emotion_accuracy"],
            "target_emotion_score": row["target_emotion_score"],
            "fact_consistency": row["fact_consistency_score_1"],
            "original_text": row["reference_original_text"], "evaluation_query": row["evaluation_query"],
            "retrieved_factual_memory": row["retrieved_factual_memory"],
            "bare_generation": bare["generated_text"], "fact_prompt_generation": fact["generated_text"],
            "steering_generation": steering["generated_text"], "npti_generation": npti["generated_text"],
        }
        rows.append(record)
        title = {
            "success": "Success: Event✓ Affect✓",
            "event_wrong_affect_right": "Failure type 1: Event✗ Affect✓",
            "event_right_affect_wrong": "Failure type 2: Event✓ Affect✗",
            "fact_consistency_failure": "Failure type 3: low factual consistency",
        }[case_type]
        md += [
            f"## Case {i} — {title}", "",
            f"- Query row: `{qid}`; retrieval method: `{method}`",
            f"- Gold emotion: `{record['gold_emotion']}`; retrieved: `{record['retrieved_emotion']}`; injected: `{record['injected_emotion']}`",
            f"- Exact event recall: `{bool(record['memory_exact_top1'])}`; affect correct: `{bool(record['affect_correct'])}`",
            f"- Generated emotion correct: `{bool(int(record['emotion_accuracy']))}`; target score: `{float(record['target_emotion_score']):.4f}`; fact consistency: `{float(record['fact_consistency']):.4f}`", "",
            "**Original text**", "", "```text", safe_block(record["original_text"]), "```", "",
            "**Evaluation query**", "", "```text", safe_block(record["evaluation_query"]), "```", "",
            "**Retrieved factual memory**", "", "```text", safe_block(record["retrieved_factual_memory"]), "```", "",
        ]
        for label, key in (("Bare query", "bare_generation"), ("Fact prompt only", "fact_prompt_generation"), ("Normal multilayer steering", "steering_generation"), ("Normal NPTI", "npti_generation")):
            md += [f"**{label}**", "", "```text", safe_block(record[key]), "```", ""]
    dest = DERIVED / "case_studies"
    write_csv(dest / "selected_cases.csv", rows)
    (dest / "case_studies.md").write_text("\n".join(md) + "\n")


def build_rq3_compact():
    path = DERIVED / "RQ3_emotion_realization" / "all_7_groups_micro.csv"
    with path.open() as handle:
        source_rows = list(csv.DictReader(handle))
    by_group = defaultdict(list)
    for row in source_rows:
        by_group[row["group"]].append(row)
    rows = []
    for group in GROUPS:
        values = by_group[group]
        if len(values) != 5:
            raise ValueError(f"{group}: expected five methods, got {len(values)}")
        best = max(values, key=lambda x: float(x["emotion_accuracy"]))
        rows.append({
            "group": group, "condition": GROUP_NAMES[group], "methods": 5,
            "mean_emotion_accuracy": statistics.fmean(float(x["emotion_accuracy"]) for x in values),
            "mean_target_emotion_score": statistics.fmean(float(x["target_emotion_score"]) for x in values),
            "mean_emotion_similarity": statistics.fmean(float(x["emotion_similarity"]) for x in values),
            "mean_fact_consistency": statistics.fmean(float(x["fact_consistency"]) for x in values),
            "best_method": best["method"],
            "best_emotion_accuracy": float(best["emotion_accuracy"]),
            "best_target_emotion_score": float(best["target_emotion_score"]),
            "best_emotion_similarity": float(best["emotion_similarity"]),
            "best_fact_consistency": float(best["fact_consistency"]),
        })
    dest = DERIVED / "RQ3_emotion_realization"
    write_csv(dest / "compact_main_table.csv", rows)
    md = [
        "# RQ3 — Compact Main Results", "",
        "Each row summarizes the five retrieval methods. The mean shows overall condition performance; the best-method columns identify the strongest emotion-accuracy configuration without expanding to the full 35-row table.", "",
        "| Condition | Mean Emo. Acc | Mean Target | Mean Similarity | Mean Fact | Best method | Best Emo. Acc | Best Target | Best Similarity | Best Fact |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['condition']} | {r['mean_emotion_accuracy']:.2%} | {r['mean_target_emotion_score']:.4f} | "
            f"{r['mean_emotion_similarity']:.4f} | {r['mean_fact_consistency']:.4f} | {r['best_method']} | "
            f"{r['best_emotion_accuracy']:.2%} | {r['best_target_emotion_score']:.4f} | "
            f"{r['best_emotion_similarity']:.4f} | {r['best_fact_consistency']:.4f} |"
        )
    (dest / "RQ3_compact_main_table.md").write_text("\n".join(md) + "\n")


def build_results_narrative():
    def csv_rows(path):
        with path.open() as handle:
            return list(csv.DictReader(handle))

    recall = {r["method"]: r for r in csv_rows(DERIVED / "RQ1_binding" / "exact_recall_at_1.csv")}
    binding = {r["method"]: r for r in csv_rows(DERIVED / "RQ1_binding" / "binding_at_1.csv")}
    quadrants = {r["method"]: r for r in csv_rows(DERIVED / "RQ1_binding" / "event_affect_quadrants.csv")}
    transition = json.loads((DERIVED / "RQ2_error_propagation" / "two_stage_transition_summary.json").read_text())
    gains = csv_rows(DERIVED / "RQ2_error_propagation" / "normal_vs_oracle_gain.csv")
    compact = csv_rows(DERIVED / "RQ3_emotion_realization" / "compact_main_table.csv")
    sim = {r["method"]: r for r in csv_rows(DERIVED / "similarity_analysis" / "similarity_contrasts.csv")}
    max_vector = max((r for r in gains if r["family"] == "multilayer"), key=lambda r: float(r["emotion_accuracy_gain"]))
    max_npti = max((r for r in gains if r["family"] == "npti"), key=lambda r: float(r["emotion_accuracy_gain"]))
    best_condition = max(compact, key=lambda r: float(r["best_emotion_accuracy"]))
    md = [
        "# EM-TRACE Results Narrative", "",
        "## RQ1 — Memory recall does not guarantee affect binding", "",
        f"Four single-stage/combined retrieval variants show near-identity between Exact Recall@1 and Binding@1 because their final affect label is directly tied to the retrieved memory. `concat-sum` reaches {float(recall['concat-sum']['exact_recall_at_1']):.2%} Exact Recall@1 and the same Binding@1. In contrast, `two-stage` retains {float(recall['two-stage']['exact_recall_at_1']):.2%} exact event recall but falls to {float(binding['two-stage']['binding_at_1']):.2%} Binding@1, with {quadrants['two-stage']['event_pos_affect_neg']} Event✓ Affect✗ cases. Thus, recovering the correct event is insufficient when a later affect-recovery stage can overwrite the associated emotion.", "",
        f"Retrieval scores support the dual-track view: pooled correct recalls have a large fact-score advantage of {float(sim['all']['fact_score_gap']):+.4f}. By contrast, the pooled emotion-score gap is {float(sim['all']['emotion_score_gap']):+.4f}, so affective similarity alone does not separate exact-event matches and is slightly higher for wrong recalls in the pooled data. The method-level files show the same pattern without hiding retriever-specific behavior.", "",
        "## RQ2 — Multi-stage affect errors propagate more often than they are corrected", "",
        f"In Two-stage retrieval, the prototype stage corrects only {transition['correction_rate_given_stage1_wrong']:.2%} of initially wrong memory-emotion assignments, but corrupts {transition['corruption_rate_given_stage1_correct']:.2%} of initially correct assignments. This asymmetry indicates that the second stage is predominantly an error-propagation/corruption point rather than a reliable repair mechanism.", "",
        f"Oracle affect information yields the largest Multilayer emotion-accuracy gain for `{max_vector['method']}` ({float(max_vector['emotion_accuracy_gain']):+.2%}) and the largest NPTI gain for `{max_npti['method']}` ({float(max_npti['emotion_accuracy_gain']):+.2%}). The normal–oracle gap therefore quantifies the headroom lost at emotion recovery before realization begins.", "",
        "## RQ3 — Better affect control improves realization but cannot repair wrong binding", "",
        f"The strongest observed configuration is **{best_condition['condition']} / {best_condition['best_method']}**, reaching {float(best_condition['best_emotion_accuracy']):.2%} emotion accuracy. Across the compact seven-condition table, oracle-label and oracle-steering settings outperform their normal counterparts on affect realization, while fact consistency changes are smaller and method-dependent. This supports a separation between memory binding quality and downstream control strength: stronger control can realize a supplied emotion more reliably, but it cannot identify or repair an incorrectly bound event–affect pair by itself.", "",
        "## Paper-ready takeaway", "",
        "EM-TRACE exposes three distinct failure sites: event retrieval, event–affect binding, and emotion realization. Exact event recall alone overestimates system success; multi-stage emotion recovery can corrupt correct bindings; and oracle experiments show that improved affect information creates substantial realization gains. The complete evidence package includes per-method quadrants, transition rates, generation quality by quadrant, similarity contrasts, confusion matrices, and aligned qualitative cases.",
    ]
    (DERIVED / "RESULTS_NARRATIVE.md").write_text("\n".join(md) + "\n")


def main():
    build_dataset_statistics()
    build_confusion_matrices()
    build_similarity_analysis()
    build_case_studies()
    build_rq3_compact()
    build_results_narrative()


if __name__ == "__main__":
    main()
