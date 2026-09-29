#!/usr/bin/env python3
"""P0-2 paired query-level bootstrap CIs and McNemar tests for EM-TRACE.

Uses only completed per-query evaluations.  A bootstrap draw samples the same
query_row_id vector for every compared method, preserving the paired design.
No model, retrieval, prompt, or generation call occurs in this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PLUS = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs/eval"
METHOD_NAMES = {
    "fact_only": "fact-only", "emotion_only": "emotion-only", "concat_sum": "concat-sum",
    "similarity_product": "similarity-product",
    "two_stage_fact_then_cue_prototype": "two-stage", "two_stage_fact_then_m2_prototype": "two-stage",
}
SOURCES = {
    "fact_prompt_only": "eval_fact_prompt_only_20260925",
    "memory_retrieved_emotion_label": "eval_memory_retrieved_emotion_label_20260925",
    "memory_oracle_emotion_label": "eval_memory_oracle_emotion_label_20260925",
    "normal_multilayer": "eval_normal_retrieval_vector_steering_20260923",
    "oracle_multilayer": "eval_oracle_retrieval_vector_steering_20260923",
    "normal_npti": "eval_normal_retrieval_npti_20260923",
    "oracle_npti": "eval_oracle_retrieval_npti_20260923",
}
GEN_METRICS = ("emotion_accuracy", "target_emotion_score", "emotion_similarity_1", "fact_consistency_score_1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise ValueError(f"no rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def canonical_method(row: dict[str, Any]) -> str:
    try: return METHOD_NAMES[row["retrieval_method"]]
    except KeyError as e: raise KeyError(f"unknown retrieval method: {row.get('retrieval_method')}") from e


def unique_by_query(rows: list[dict[str, Any]], description: str) -> dict[int, dict[str, Any]]:
    result = {int(r["query_row_id"]): r for r in rows}
    if len(result) != 1103 or len(result) != len(rows) or set(result) != set(range(1103)):
        raise ValueError(f"{description} must contain exactly one row for each query_row_id 0..1102; got {len(rows)} rows / {len(result)} ids")
    return result


def bootstrap_mean(values: np.ndarray, indices: np.ndarray, chunk: int = 250) -> tuple[float, float, float]:
    estimate = float(values.mean())
    draws = []
    for start in range(0, len(indices), chunk):
        draws.append(values[indices[start:start + chunk]].mean(axis=1))
    dist = np.concatenate(draws)
    low, high = np.quantile(dist, [0.025, 0.975])
    return estimate, float(low), float(high)


def exact_binomial_two_sided(a_wins: int, b_wins: int) -> float:
    """Exact two-sided McNemar/binomial p-value; ties are irrelevant."""
    n = a_wins + b_wins
    if n == 0: return 1.0
    k = min(a_wins, b_wins)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def mcnemar(method_a: np.ndarray, method_b: np.ndarray) -> dict[str, Any]:
    # a_win means A correct while B incorrect; b_win reverses it.
    a_win = int(np.sum((method_a == 1) & (method_b == 0)))
    b_win = int(np.sum((method_a == 0) & (method_b == 1)))
    return {"n_a_correct_b_wrong": a_win, "n_a_wrong_b_correct": b_win, "discordant_n": a_win + b_win, "p_value_exact_two_sided": exact_binomial_two_sided(a_win, b_win)}


def load_generation_sets() -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    result: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for group, directory in SOURCES.items():
        rows = read_jsonl(EVAL / directory / "per_generation_metrics.jsonl")
        # NPTI files carry an additional duplicated bare-query baseline.  P0-2
        # concerns the five intervention rows, so it is excluded explicitly.
        if group.endswith("npti"):
            rows = [r for r in rows if r.get("condition") == "retrieved_fact_prompt_npti"]
        methods: dict[str, list[dict[str, Any]]] = {}
        for r in rows: methods.setdefault(canonical_method(r), []).append(r)
        if set(methods) != set(METHOD_NAMES.values()):
            raise ValueError(f"{group}: unexpected methods {sorted(methods)}")
        result[group] = {m: unique_by_query(rs, f"{group}/{m}") for m, rs in methods.items()}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=PLUS / "results/p0_2_bootstrap")
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260928)
    a = p.parse_args(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.iterations < 1000: raise ValueError("use at least 1,000 bootstrap iterations")
    rng = np.random.default_rng(a.seed)
    # Shared draw matrix is important: every paired comparison samples the
    # exact same query ids together.
    indices = rng.integers(0, 1103, size=(a.iterations, 1103), dtype=np.int32)
    grouped = load_generation_sets()
    methods = ["fact-only", "emotion-only", "concat-sum", "similarity-product", "two-stage"]

    # RQ1: retrieval-only CIs, read from Normal Multilayer. Retrieval records
    # are identical across normal generated conditions and carry final Stage-2
    # emotion for two-stage.  UAR is retained as the paper's legacy ESR metric:
    # event wrong AND affect correct, not standard macro recall.
    retrieval_rows: list[dict[str, Any]] = []
    retrieval_arrays: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        records = grouped["normal_multilayer"][method]
        exact = np.array([records[i]["memory_exact_top1"] for i in range(1103)], dtype=float)
        affect = np.array([records[i]["retrieval_emotion_match"] for i in range(1103)], dtype=float)
        metrics = {
            "exact_recall_at1": exact,
            "affect_recovery_accuracy": affect,
            "binding_at1": exact * affect,
            "uar_legacy_event_wrong_affect_correct": (1.0 - exact) * affect,
        }
        retrieval_arrays[method] = metrics
        for metric, values in metrics.items():
            estimate, low, high = bootstrap_mean(values, indices)
            retrieval_rows.append({"analysis": "retrieval", "metric": metric, "group": "normal_retrieval", "method": method, "n": 1103, "estimate": estimate, "ci_low": low, "ci_high": high, "bootstrap_iterations": a.iterations, "seed": a.seed})
    write_csv(out / "retrieval_metric_bootstrap_ci.csv", retrieval_rows)

    # All 35 completed generation conditions receive point estimates + CIs.
    generation_rows: list[dict[str, Any]] = []
    for group, by_method in grouped.items():
        for method in methods:
            records = by_method[method]
            for metric in GEN_METRICS:
                values = np.array([float(records[i][metric]) for i in range(1103)], dtype=float)
                estimate, low, high = bootstrap_mean(values, indices)
                generation_rows.append({"analysis": "generation", "metric": metric, "group": group, "method": method, "n": 1103, "estimate": estimate, "ci_low": low, "ci_high": high, "bootstrap_iterations": a.iterations, "seed": a.seed})
    write_csv(out / "generation_metric_bootstrap_ci.csv", generation_rows)

    # Key paired contrasts: requested retrieval contrasts plus Normal–Oracle
    # deltas for every five-method Vector and NPTI pairing.
    diff_rows: list[dict[str, Any]] = []
    requested = [
        ("exact_recall_at1", "concat-sum", "fact-only", retrieval_arrays["concat-sum"]["exact_recall_at1"], retrieval_arrays["fact-only"]["exact_recall_at1"]),
        ("binding_at1", "two-stage", "fact-only", retrieval_arrays["two-stage"]["binding_at1"], retrieval_arrays["fact-only"]["binding_at1"]),
    ]
    for metric, a_name, b_name, left, right in requested:
        est, low, high = bootstrap_mean(left - right, indices)
        diff_rows.append({"analysis": "retrieval_key_comparison", "metric": metric, "comparison": f"{a_name}_minus_{b_name}", "n": 1103, "estimate_diff": est, "ci_low": low, "ci_high": high, "bootstrap_iterations": a.iterations, "seed": a.seed})
    for family, normal, oracle in (("multilayer", "normal_multilayer", "oracle_multilayer"), ("npti", "normal_npti", "oracle_npti")):
        for method in methods:
            for metric in GEN_METRICS:
                left = np.array([float(grouped[oracle][method][i][metric]) for i in range(1103)])
                right = np.array([float(grouped[normal][method][i][metric]) for i in range(1103)])
                est, low, high = bootstrap_mean(left - right, indices)
                diff_rows.append({"analysis": "normal_oracle", "metric": metric, "comparison": f"oracle_{family}_minus_normal_{family}:{method}", "n": 1103, "estimate_diff": est, "ci_low": low, "ci_high": high, "bootstrap_iterations": a.iterations, "seed": a.seed})
    write_csv(out / "paired_bootstrap_differences.csv", diff_rows)

    tests: list[dict[str, Any]] = []
    for metric, a_name, b_name, left, right in requested:
        tests.append({"analysis": "retrieval_key_comparison", "metric": metric, "comparison": f"{a_name}_vs_{b_name}", "n": 1103, **mcnemar(left.astype(int), right.astype(int))})
    for family, normal, oracle in (("multilayer", "normal_multilayer", "oracle_multilayer"), ("npti", "normal_npti", "oracle_npti")):
        for method in methods:
            left = np.array([grouped[oracle][method][i]["emotion_accuracy"] for i in range(1103)], dtype=int)
            right = np.array([grouped[normal][method][i]["emotion_accuracy"] for i in range(1103)], dtype=int)
            tests.append({"analysis": "normal_oracle", "metric": "emotion_accuracy", "comparison": f"oracle_{family}_vs_normal_{family}:{method}", "n": 1103, **mcnemar(left, right)})
    write_csv(out / "mcnemar_results.csv", tests)

    readme = {
        "protocol": "10,000 query-level nonparametric bootstrap resamples; a resampled query retains all method/condition observations, preserving paired comparisons. CI is percentile 2.5th–97.5th. McNemar uses an exact two-sided binomial test on discordant pairs.",
        "n_queries": 1103, "bootstrap_iterations": a.iterations, "random_seed": a.seed,
        "retrieval_source": str(EVAL / SOURCES["normal_multilayer"] / "per_generation_metrics.jsonl"),
        "generation_sources": SOURCES,
        "metric_note": "uar_legacy_event_wrong_affect_correct is the previously used ESR/UAR-style rate: event retrieval wrong but final retrieved affect correct. It is not standard macro recall.",
        "excluded_rows": "NPTI bare-query baseline rows excluded; five retrieved_fact_prompt_npti rows per query retained.",
    }
    (out / "README.md").write_text("# P0-2 Paired Bootstrap and McNemar Analysis\n\n" + json.dumps(readme, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(out), "retrieval_ci_rows": len(retrieval_rows), "generation_ci_rows": len(generation_rows), "paired_diff_rows": len(diff_rows), "mcnemar_rows": len(tests), **readme}, ensure_ascii=False))


if __name__ == "__main__":
    main()
