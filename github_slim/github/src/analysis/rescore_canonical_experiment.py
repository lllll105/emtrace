#!/usr/bin/env python3
"""Refresh factual-memory NLI for seven conditions after NPTI prompt alignment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLD_EVAL = ROOT / "outputs/eval"
NEW_RAW = ROOT / "outputs/npti_helpful_prompt_20260927"
OUTPUT = OLD_EVAL / "eval_canonical_helpful_factual_nli_20260927"
EVALUATOR = ROOT / "use_2rag/code/evaluate_generation_emotion_fact_metrics.py"
CONDITIONS = {
    "C1_fact_prompt_only": "eval_fact_prompt_only_20260925",
    "C2_retrieved_emotion_label": "eval_memory_retrieved_emotion_label_20260925",
    "C3_oracle_emotion_label": "eval_memory_oracle_emotion_label_20260925",
    "C4_normal_vector": "eval_normal_retrieval_vector_steering_20260923",
    "C5_oracle_vector": "eval_oracle_retrieval_vector_steering_20260923",
    "C6_normal_npti": "normal",
    "C7_oracle_npti": "oracle",
}
METHODS = ("fact_only", "emotion_only", "concat_sum", "similarity_product",
           "two_stage_fact_then_cue_prototype")


def count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def complete(directory: Path, expected: int) -> bool:
    result = directory / "per_generation_metrics.jsonl"
    summary = directory / "summary.csv"
    return result.is_file() and summary.is_file() and count(result) == expected


def run_eval(source: Path, target: Path, *, reuse: bool, gpu: int | None,
             baseline: bool = False, raw_directory: bool = False) -> None:
    expected = 1103 if baseline else 5515
    if complete(target, expected):
        print(json.dumps({"status": "reused", "output": str(target), "rows": expected}), flush=True)
        return
    if (target / "per_generation_metrics.jsonl").exists():
        raise RuntimeError(f"partial evaluation exists; inspect before retrying: {target}")
    target.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-u", str(EVALUATOR),
            "--input-dir" if raw_directory else "--input-file", str(source),
            "--output-dir", str(target), "--fact-reference", "factual_memory",
            "--device", "cuda:0" if gpu is not None else "cpu",
            "--condition-field", "condition" if baseline else "retrieval_method"]
    if reuse:
        args.append("--reuse-emotion-metrics")
    if baseline:
        args += ["--only-condition", "bare_query"]
    else:
        args += ["--skip-condition", "baseline"]
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(json.dumps({"status": "evaluating", "source": str(source), "output": str(target),
                      "reference": "factual_memory"}), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)
    if not complete(target, expected):
        raise RuntimeError(f"expected {expected} evaluated rows: {target}")


def wait_for_generation(source: str) -> Path:
    folder = NEW_RAW / source / "shard_00_of_01"
    done = NEW_RAW / f"{source}.done"
    log = NEW_RAW / "logs" / f"{source}.log"
    while not done.exists():
        if log.exists() and "failed " in log.read_text(encoding="utf-8")[-400:]:
            raise RuntimeError(f"NPTI generation failed: {log}")
        print(json.dumps({"status": "waiting_for_generation", "source": source}), flush=True)
        time.sleep(60)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config["prompt_style"] != "helpful":
        raise ValueError(f"expected helpful prompt: {folder}")
    for method in ("fact_only", "emotion_only", "concat_sum", "similarity_product",
                   "two_stage_fact_then_m2_prototype"):
        path = folder / method / "retrieved_fact_prompt_npti_generations.jsonl"
        if count(path) != 1103:
            raise RuntimeError(f"incomplete {method} generation: {path}")
    return folder


def summarize() -> None:
    if not complete(OUTPUT / "C0_bare_query", 1103):
        raise RuntimeError("canonical bare-query baseline is incomplete")
    columns = ("emotion_accuracy", "emotion_score_1", "emotion_similarity_1",
               "emotion_similarity_11class", "fact_consistency_score_1")
    headings = ("Emotion Accuracy", "Target Emotion Score", "Emotion Similarity",
                "11-class Similarity", "Factual-memory NLI")
    report = ["# Canonical prompt and factual-memory evaluation", "",
              "Generation prompt ends with 'Generate a helpful response.' for every retrieval condition.",
              "The original five non-NPTI conditions reuse their saved generations; NPTI Normal/Oracle were regenerated.",
              "NLI premise is generated_text and hypothesis is the matching test sample's factual_memory.",
              "Emotion scores continue to use the original-text ModernBERT reference.",
              "All 35 method-condition cells contain 1,103 unique test samples.", "",
              "| Condition | Method | n | " + " | ".join(headings) + " |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        path = OUTPUT / condition / "summary.csv"
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        micro = {row["condition"].replace("two_stage_fact_then_m2_prototype",
                                           "two_stage_fact_then_cue_prototype"): row
                 for row in rows if row["scope"] == "micro" and row["emotion"] == "all"}
        if set(micro) != set(METHODS):
            raise ValueError(f"missing method in {path}: {set(METHODS) - set(micro)}")
        for method in METHODS:
            row = micro[method]
            if int(row["n"]) != 1103:
                raise ValueError(f"unexpected n in {path}: {method}")
            metrics = [f"{float(row[columns[0]]):.2%}"] + [f"{float(row[c]):.4f}" for c in columns[1:]]
            report.append(f'| {condition} | {method} | 1103 | ' + " | ".join(metrics) + " |")
    with (OUTPUT / "C0_bare_query" / "summary.csv").open(encoding="utf-8") as stream:
        baseline = next(row for row in csv.DictReader(stream)
                        if row["scope"] == "micro" and row["condition"] == "bare_query")
    report.extend(["", "## Bare query", "",
                   f'Accuracy {float(baseline["emotion_accuracy"]):.2%}; '
                   f'target score {float(baseline["emotion_score_1"]):.4f}; '
                   f'factual-memory NLI {float(baseline["fact_consistency_score_1"]):.4f}.', "",
                   "The earlier original-text NLI results remain in outputs/eval/all_results.md."])
    (OUTPUT / "all_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "report": str(OUTPUT / "all_results.md")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-existing", action="store_true",
                        help="Rescore saved generations now without waiting for the new NPTI runs.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for condition, directory in list(CONDITIONS.items())[:5]:
        source = OLD_EVAL / directory / "per_generation_metrics.jsonl"
        run_eval(source, OUTPUT / condition, reuse=True, gpu=None)
    baseline = OLD_EVAL / "eval_two_stage_fact_prompt_m2_20260922" / "per_generation_metrics.jsonl"
    run_eval(baseline, OUTPUT / "C0_bare_query", reuse=True, gpu=None, baseline=True)
    if args.only_existing:
        return
    for condition, source in list(CONDITIONS.items())[5:]:
        folder = wait_for_generation(source)
        run_eval(folder, OUTPUT / condition, reuse=False,
                 gpu=0 if source == "normal" else 6, raw_directory=True)
    summarize()


if __name__ == "__main__":
    main()
