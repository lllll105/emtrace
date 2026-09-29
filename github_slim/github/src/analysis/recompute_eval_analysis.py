#!/usr/bin/env python3
"""Recompute EM-TRACE sample analyses from the seven canonical evaluation files."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


METHODS = (
    "fact_only", "emotion_only", "concat_sum", "similarity_product",
    "two_stage_fact_then_cue_prototype",
)
EVAL_DIRS = {
    "C1_fact_prompt_only": "eval_fact_prompt_only_20260925",
    "C2_retrieved_emotion_label": "eval_memory_retrieved_emotion_label_20260925",
    "C3_oracle_emotion_label": "eval_memory_oracle_emotion_label_20260925",
    "C4_normal_vector": "eval_normal_retrieval_vector_steering_20260923",
    "C5_oracle_vector": "eval_oracle_retrieval_vector_steering_20260923",
    "C6_normal_npti": "eval_normal_retrieval_npti_20260923",
    "C7_oracle_npti": "eval_oracle_retrieval_npti_20260923",
}
PAIRS = (
    ("emotion_label", "C2_retrieved_emotion_label", "C3_oracle_emotion_label"),
    ("vector", "C4_normal_vector", "C5_oracle_vector"),
    ("npti", "C6_normal_npti", "C7_oracle_npti"),
)
GROUPS = (
    "event_correct_affect_correct", "event_correct_affect_wrong",
    "event_wrong_affect_correct", "event_wrong_affect_wrong",
)
FIELDS = (
    "emotion_accuracy", "target_emotion_score", "emotion_similarity_1",
    "emotion_similarity_11class", "fact_consistency_score_1",
)


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def csv_out(path: Path, entries: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)


def method(row: dict) -> str:
    name = row["retrieval_method"]
    return name.replace("two_stage_fact_then_m2_prototype", "two_stage_fact_then_cue_prototype")


def average(entries: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in entries if row.get(field) is not None]
    return mean(values) if values else None


def rank(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        midrank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = midrank
        start = end
    return result


def spearman(entries: list[dict]) -> float | None:
    points = [(float(r["fact_similarity"]), float(r["target_emotion_score"]))
              for r in entries if r.get("fact_similarity") is not None
              and r.get("target_emotion_score") is not None]
    if len(points) < 3:
        return None
    x, y = rank([p[0] for p in points]), rank([p[1] for p in points])
    mx, my = mean(x), mean(y)
    denominator = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / denominator if denominator else None


def read_evaluations(eval_dir: Path) -> dict[str, dict[str, dict[str, dict]]]:
    data = {}
    for condition, directory in EVAL_DIRS.items():
        indexed = {name: {} for name in METHODS}
        path = eval_dir / directory / "per_generation_metrics.jsonl"
        for row in rows(path):
            if row["condition"] == "baseline":
                continue  # The NPTI baseline is shared, not a retrieval condition.
            name, sample_id = method(row), row["memory_id"]
            if name not in indexed or sample_id in indexed[name]:
                raise ValueError(f"invalid or repeated method/sample in {path}: {name}/{sample_id}")
            indexed[name][sample_id] = row
        if any(len(indexed[name]) != 1103 for name in METHODS):
            raise ValueError(f"incomplete condition {condition}: {[(m, len(v)) for m, v in indexed.items()]}")
        data[condition] = indexed

    anchor = data["C1_fact_prompt_only"]
    for condition, methods in data.items():
        for name in METHODS:
            if methods[name].keys() != anchor[name].keys():
                raise ValueError(f"sample IDs differ in {condition}/{name}")
            for sample_id, row in methods[name].items():
                original = anchor[name][sample_id]
                fields = ("target_emotion", "evaluation_query", "retrieved_memory_id", "retrieved_emotion", "retrieved_factual_memory")
                if any(row.get(field) != original.get(field) for field in fields):
                    raise ValueError(f"retrieval or prompt fact differs in {condition}/{name}/{sample_id}")
    return data


def build_records(data: dict) -> tuple[list[dict], list[dict]]:
    retrieval, samples = [], []
    anchor = data["C1_fact_prompt_only"]
    vector = data["C4_normal_vector"]
    npti = data["C6_normal_npti"]
    for name in METHODS:
        for sample_id, row in anchor[name].items():
            exact = int(row["retrieved_memory_id"] == sample_id)
            affect = int(row["retrieved_emotion"] == row["target_emotion"])
            group = GROUPS[(1 - exact) * 2 + (1 - affect)]
            recorded = {
                "sample_id": sample_id, "retrieval_method": name,
                "target_memory_id": sample_id, "target_emotion": row["target_emotion"],
                "retrieved_memory_id": row["retrieved_memory_id"],
                "retrieved_emotion": row["retrieved_emotion"],
                "stage1_memory_emotion": npti[name][sample_id].get("retrieved_emotion_from_memory"),
                "fact_similarity": vector[name][sample_id].get("fact_score"),
                "emotion_similarity": vector[name][sample_id].get("emotion_score"),
                "exact_match": exact, "emotion_match": affect,
                "binding_success": int(bool(exact and affect)), "binding_group": group,
            }
            retrieval.append(recorded)
            for condition in EVAL_DIRS:
                generation = data[condition][name][sample_id]
                samples.append({**recorded, "condition": condition,
                                **{field: generation.get(field) for field in FIELDS}})
    return retrieval, samples


def retrieval_tables(retrieval: list[dict], output: Path) -> list[dict]:
    table = []
    for name in METHODS:
        selected = [row for row in retrieval if row["retrieval_method"] == name]
        counts = Counter(row["binding_group"] for row in selected)
        table.append({"retrieval_method": name, "n": len(selected),
                      "exact_match_rate": average(selected, "exact_match"),
                      "emotion_match_rate": average(selected, "emotion_match"),
                      "binding_success_rate": average(selected, "binding_success"),
                      **{group: counts[group] for group in GROUPS}})
    csv_out(output / "table1_retrieval_quality.csv", table,
            ["retrieval_method", "n", "exact_match_rate", "emotion_match_rate",
             "binding_success_rate", *GROUPS])
    return table


def generation_table(samples: list[dict], output: Path) -> None:
    table = []
    for condition in EVAL_DIRS:
        for name in METHODS:
            source = [r for r in samples if r["condition"] == condition and r["retrieval_method"] == name]
            for group in GROUPS:
                selected = [r for r in source if r["binding_group"] == group]
                table.append({"condition": condition, "retrieval_method": name,
                              "binding_group": group, "n": len(selected),
                              **{field: average(selected, field) for field in FIELDS}})
    columns = ["condition", "retrieval_method", "binding_group", "n", *FIELDS]
    csv_out(output / "table2_generation_by_binding.csv", table, columns)
    # Replace the legacy three-group table so stale counts cannot be mistaken for current results.
    csv_out(output / "table2_generation_by_recall_group.csv", table, columns)


def similarity_table(samples: list[dict], output: Path) -> None:
    table = []
    for condition in EVAL_DIRS:
        for name in METHODS:
            for match in ("all", "correct", "incorrect"):
                selected = [r for r in samples if r["condition"] == condition
                            and r["retrieval_method"] == name and r["fact_similarity"] is not None
                            and (match == "all" or r["emotion_match"] == int(match == "correct"))]
                selected.sort(key=lambda r: (-float(r["fact_similarity"]), r["sample_id"]))
                size = len(selected)
                if not size:
                    continue
                boundaries = (0, (size + 2) // 3, 2 * ((size + 2) // 3), size)
                for index, group in enumerate(("high", "middle", "low")):
                    third = selected[boundaries[index]:boundaries[index + 1]]
                    table.append({"condition": condition, "retrieval_method": name,
                                  "emotion_retrieval": match, "similarity_group": group,
                                  "n": len(third), "mean_fact_similarity": average(third, "fact_similarity"),
                                  "emotion_accuracy": average(third, "emotion_accuracy"),
                                  "emotion_score": average(third, "target_emotion_score"),
                                  "exact_match": average(third, "exact_match"),
                                  "spearman_fact_vs_emotion_score": None})
                table.append({"condition": condition, "retrieval_method": name,
                              "emotion_retrieval": match, "similarity_group": "spearman",
                              "n": size, "mean_fact_similarity": None,
                              "emotion_accuracy": None, "emotion_score": None, "exact_match": None,
                              "spearman_fact_vs_emotion_score": spearman(selected)})
    csv_out(output / "table3_similarity_analysis.csv", table,
            ["condition", "retrieval_method", "emotion_retrieval", "similarity_group",
             "n", "mean_fact_similarity", "emotion_accuracy", "emotion_score",
             "exact_match", "spearman_fact_vs_emotion_score"])


def oracle_tables(data: dict, retrieval: list[dict], output: Path) -> list[dict]:
    lookup = {(r["retrieval_method"], r["sample_id"]): r for r in retrieval}
    pairs = []
    for kind, normal_condition, oracle_condition in PAIRS:
        for name in METHODS:
            for sample_id, normal in data[normal_condition][name].items():
                oracle = data[oracle_condition][name][sample_id]
                retrieval_row = lookup[name, sample_id]
                pairs.append({"contrast": kind, "retrieval_method": name,
                              "sample_id": sample_id, "binding_group": retrieval_row["binding_group"],
                              "emotion_retrieval_correct": retrieval_row["emotion_match"],
                              "normal_accuracy": normal["emotion_accuracy"],
                              "oracle_accuracy": oracle["emotion_accuracy"],
                              "delta_accuracy": oracle["emotion_accuracy"] - normal["emotion_accuracy"],
                              "normal_emotion_score": normal["target_emotion_score"],
                              "oracle_emotion_score": oracle["target_emotion_score"],
                              "delta_emotion_score": oracle["target_emotion_score"] - normal["target_emotion_score"],
                              "normal_fact_consistency": normal["fact_consistency_score_1"],
                              "oracle_fact_consistency": oracle["fact_consistency_score_1"],
                              "delta_fact_consistency": oracle["fact_consistency_score_1"] - normal["fact_consistency_score_1"],
                              "generated_text_identical": int(normal["generated_text"] == oracle["generated_text"])})
    columns = list(pairs[0])
    csv_out(output / "table4_normal_oracle_pairs.csv", pairs, columns)
    summary = []
    for kind, _, _ in PAIRS:
        for name in METHODS:
            for match in ("all", "correct", "incorrect"):
                selected = [p for p in pairs if p["contrast"] == kind and p["retrieval_method"] == name
                            and (match == "all" or p["emotion_retrieval_correct"] == int(match == "correct"))]
                summary.append({"contrast": kind, "retrieval_method": name,
                                "emotion_retrieval": match, "n": len(selected),
                                **{field: average(selected, field) for field in (
                                    "delta_accuracy", "delta_emotion_score", "delta_fact_consistency",
                                    "generated_text_identical")}})
    csv_out(output / "table4_normal_oracle_summary.csv", summary, list(summary[0]))
    return summary


def stage2_table(retrieval: list[dict], output: Path) -> list[dict]:
    selected = [r for r in retrieval if r["retrieval_method"] == METHODS[-1]]
    table = []
    for first in (1, 0):
        for second in (1, 0):
            subset = [r for r in selected if int(r["stage1_memory_emotion"] == r["target_emotion"]) == first
                      and r["emotion_match"] == second]
            table.append({"stage1_emotion_correct": first, "stage2_emotion_correct": second,
                          "n": len(subset), "rate": len(subset) / len(selected),
                          "exact_match_rate": average(subset, "exact_match")})
    csv_out(output / "table5_two_stage_transitions.csv", table, list(table[0]))
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--outputs-dir", type=Path, default=None,
                        help="legacy alias: its eval subdirectory is used")
    parser.add_argument("--output-dir", type=Path, default=Path("use_2rag/plus/results"))
    args = parser.parse_args()
    canonical = (args.eval_dir / "C1_fact_prompt_only" / "per_generation_metrics.jsonl").is_file()
    if canonical:
        EVAL_DIRS.update({key: key for key in EVAL_DIRS})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = read_evaluations(args.outputs_dir / "eval" if args.outputs_dir else args.eval_dir)
    retrieval, samples = build_records(data)
    csv_out(args.output_dir / "sample_level_analysis.csv", samples, list(samples[0]))
    table1 = retrieval_tables(retrieval, args.output_dir)
    generation_table(samples, args.output_dir)
    similarity_table(samples, args.output_dir)
    table4 = oracle_tables(data, retrieval, args.output_dir)
    table5 = stage2_table(retrieval, args.output_dir)
    report = ["# EM-TRACE: sample-level diagnostics", "",
              f"Source: seven per_generation_metrics.jsonl files under {args.eval_dir}.",
              "Each condition has 5 methods × 1,103 samples; the 1,103-row NPTI baseline is excluded.", "",
              "Binding@1 means exact memory ID AND correct final retrieved emotion.",
              "Fact consistency uses NLI: generated text entails "
              + ("the matching factual_memory." if canonical else "the matching original_text."),
              "Similarity analysis ranks 1,103 unique samples separately within each condition and method.",
              "Normal–Oracle pairs preserve the retrieved memory and prompt fact; only the emotion source changes.",
              "",
              "| Retrieval | Exact@1 | Affect@1 | Binding@1 | Event+ Affect+ | Event+ Affect- | Event- Affect+ | Event- Affect- |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table1:
        report.append("| {retrieval_method} | {exact_match_rate:.2%} | {emotion_match_rate:.2%} | {binding_success_rate:.2%} | {event_correct_affect_correct} | {event_correct_affect_wrong} | {event_wrong_affect_correct} | {event_wrong_affect_wrong} |".format(**row))
    report += ["", "Two-stage first-stage emotion → final prototype emotion:", "",
               "| First stage correct | Second stage correct | n |", "|---:|---:|---:|"]
    for row in table5:
        report.append(f'| {row["stage1_emotion_correct"]} | {row["stage2_emotion_correct"]} | {row["n"]} |')
    report += ["", "See table2_generation_by_binding.csv for the seven conditions and four states;",
               "table3_similarity_analysis.csv for per-condition similarity analysis;",
               "table4_normal_oracle_summary.csv for paired label/vector/NPTI contrasts.", ""]
    (args.output_dir / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "retrieval_rows": len(retrieval),
                      "sample_rows": len(samples), "table1": len(table1),
                      "paired_summary_rows": len(table4), "stage2_rows": len(table5)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
