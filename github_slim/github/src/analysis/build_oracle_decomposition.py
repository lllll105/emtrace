#!/usr/bin/env python3
"""Build the 2x2 Normal/Oracle Event/Emotion generation decomposition."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parent
EVAL_ROOT = ROOT.parents[1] / "outputs" / "eval"
SOURCE_ROOT = ROOT / "source_results"
DEST = ROOT / "derived" / "oracle_decomposition"

SEED = 20260929
BOOTSTRAP_REPLICATES = 5000
METHODS = ["fact-only", "emotion-only", "concat-sum", "similarity-product", "two-stage"]
FAMILIES = ["emotion-label", "multilayer-vector", "npti"]
CONDITIONS = ["Normal", "Oracle Emotion", "Oracle Event", "Full Oracle"]
METRICS = {
    "emotion_accuracy": "emotion_accuracy",
    "target_emotion_score": "emotion_score_1",
    "emotion_similarity": "emotion_similarity_1",
    "original_text_entailment": "fact_consistency_score_1",
}
COMPARISONS = [
    ("Oracle Emotion - Normal", "Oracle Emotion", "Normal"),
    ("Oracle Event - Normal", "Oracle Event", "Normal"),
    ("Full Oracle - Oracle Emotion", "Full Oracle", "Oracle Emotion"),
    ("Full Oracle - Oracle Event", "Full Oracle", "Oracle Event"),
    ("Full Oracle - Normal", "Full Oracle", "Normal"),
]
ALIASES = {
    "fact_only": "fact-only", "emotion_only": "emotion-only",
    "concat_sum": "concat-sum", "similarity_product": "similarity-product",
    "two_stage": "two-stage",
    "two_stage_fact_then_cue_prototype": "two-stage",
    "two_stage_fact_then_m2_prototype": "two-stage",
}

FAMILY_SOURCES = {
    "emotion-label": {
        "Normal": "02_memory_retrieved_emotion_label",
        "Oracle Emotion": "03_memory_oracle_emotion_label",
        "new": "eval_p1_1_oracle_event_emotion_label_20260928",
    },
    "multilayer-vector": {
        "Normal": "04_normal_multilayer_steering",
        "Oracle Emotion": "05_oracle_multilayer_steering",
        "new": "eval_p1_1_oracle_event_vector_20260928",
    },
    "npti": {
        "Normal": "06_normal_npti",
        "Oracle Emotion": "07_oracle_npti",
        "new": "eval_p1_1_oracle_event_npti_20260928",
    },
}


def norm_method(value):
    value = (value or "").strip()
    return ALIASES.get(value, value.replace("_", "-"))


def read_jsonl(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_new_sources():
    copied = []
    for index, family in enumerate(FAMILIES, 8):
        eval_name = FAMILY_SOURCES[family]["new"]
        dest_name = f"{index:02d}_oracle_event_full_oracle_{family.replace('-', '_')}"
        src, dst = EVAL_ROOT / eval_name, SOURCE_ROOT / dest_name
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("summary.csv", "config.json", "per_generation_metrics.jsonl"):
            source = src / name
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, dst / name)
        copied.append({"family": family, "eval_name": eval_name, "source_group": dest_name})
    return copied


def load_condition_data(copied):
    new_group = {x["family"]: x["source_group"] for x in copied}
    data = defaultdict(dict)
    provenance = []
    for family in FAMILIES:
        for condition in ("Normal", "Oracle Emotion"):
            group = FAMILY_SOURCES[family][condition]
            buckets = defaultdict(dict)
            for row in read_jsonl(SOURCE_ROOT / group / "per_generation_metrics.jsonl"):
                if row.get("condition") == "baseline":
                    continue
                method = norm_method(row.get("retrieval_method") or row.get("condition"))
                if method in METHODS:
                    qid = int(row["query_row_id"])
                    if qid in buckets[method]:
                        raise ValueError(f"Duplicate qid: {family}/{condition}/{method}/{qid}")
                    buckets[method][qid] = row
            for method in METHODS:
                data[(family, method)][condition] = buckets[method]
            provenance.append({"family": family, "condition": condition, "source_group": group})

        group = new_group[family]
        oracle_event = defaultdict(dict)
        full_oracle = {}
        for row in read_jsonl(SOURCE_ROOT / group / "per_generation_metrics.jsonl"):
            method = norm_method(row.get("retrieval_method") or row.get("condition"))
            qid = int(row["query_row_id"])
            if method == "full-oracle":
                if qid in full_oracle:
                    raise ValueError(f"Duplicate Full Oracle qid: {family}/{qid}")
                full_oracle[qid] = row
            elif method in METHODS:
                oracle_event[method][qid] = row
        for method in METHODS:
            data[(family, method)]["Oracle Event"] = oracle_event[method]
            data[(family, method)]["Full Oracle"] = full_oracle
        provenance += [
            {"family": family, "condition": "Oracle Event", "source_group": group},
            {"family": family, "condition": "Full Oracle", "source_group": group},
        ]

    expected_qids = set(range(1103))
    for (family, method), conditions in data.items():
        for condition in CONDITIONS:
            qids = set(conditions[condition])
            if qids != expected_qids:
                raise ValueError(f"Bad qid coverage {family}/{method}/{condition}: {len(qids)}")
        for qid in expected_qids:
            reference = conditions["Normal"][qid]
            for condition in CONDITIONS[1:]:
                row = conditions[condition][qid]
                if row["target_emotion"] != reference["target_emotion"]:
                    raise ValueError(f"Target mismatch {family}/{method}/{condition}/{qid}")
                if row.get("memory_id") != reference.get("memory_id"):
                    raise ValueError(f"Memory id mismatch {family}/{method}/{condition}/{qid}")
    return data, provenance


def get_event_source(condition, row):
    if condition in ("Oracle Event", "Full Oracle"):
        return "oracle_target_memory"
    return row.get("event_source") or "normal_retrieval"


def get_emotion_source(condition, row):
    if condition in ("Oracle Emotion", "Full Oracle"):
        return "oracle_target_emotion"
    return row.get("emotion_source") or "normal_retrieval"


def get_injected_emotion(family, row):
    if family == "emotion-label":
        return row.get("prompt_emotion") or row.get("retrieved_emotion")
    if family == "npti":
        return row.get("npti_emotion") or row.get("retrieved_emotion")
    return row.get("injected_emotion") or row.get("retrieved_emotion")


def experiment_unit_id(family, method, condition):
    if condition == "Full Oracle":
        return f"{family}/full-oracle/shared"
    return f"{family}/{condition.lower().replace(' ', '-')}/{method}"


def build_standardized_and_summary(data):
    standardized, summaries, unique_summaries = [], [], []
    seen_units = set()
    for family in FAMILIES:
        for method in METHODS:
            conditions = data[(family, method)]
            for condition in CONDITIONS:
                values = [conditions[condition][qid] for qid in range(1103)]
                unit = experiment_unit_id(family, method, condition)
                summary = {
                    "generation_family": family,
                    "retrieval_method": method,
                    "generation_condition": condition,
                    "event_source": "oracle" if condition in ("Oracle Event", "Full Oracle") else "normal",
                    "emotion_source": "oracle" if condition in ("Oracle Emotion", "Full Oracle") else "normal",
                    "experiment_unit_id": unit,
                    "shared_across_retrieval_methods": condition == "Full Oracle",
                    "n": len(values),
                }
                for output_key, source_key in METRICS.items():
                    summary[output_key] = float(np.mean([float(x[source_key]) for x in values]))
                summaries.append(summary)
                if unit not in seen_units:
                    unique_summaries.append(summary.copy())
                    seen_units.add(unit)
                for row in values:
                    standardized.append({
                        "query_row_id": int(row["query_row_id"]),
                        "generation_family": family,
                        "retrieval_method": method,
                        "generation_condition": condition,
                        "event_source": get_event_source(condition, row),
                        "emotion_source": get_emotion_source(condition, row),
                        "experiment_unit_id": unit,
                        "shared_across_retrieval_methods": condition == "Full Oracle",
                        "prompt_memory_id": row.get("prompt_memory_id") or row.get("retrieved_memory_id"),
                        "prompt_factual_memory": row.get("prompt_factual_memory") or row.get("retrieved_factual_memory"),
                        "injected_emotion": get_injected_emotion(family, row),
                        "target_emotion": row["target_emotion"],
                        "generated_text": row["generated_text"],
                        "emotion_prediction": row["modernbert_predicted_label"],
                        "emotion_correct": int(row["emotion_accuracy"]),
                        "target_emotion_score": float(row["emotion_score_1"]),
                        "emotion_similarity": float(row["emotion_similarity_1"]),
                        "original_text_entailment": float(row["fact_consistency_score_1"]),
                        "reference_original_text": row["reference_original_text"],
                    })
    write_csv(DEST / "per_query_four_conditions.csv", standardized)
    write_csv(DEST / "four_conditions_by_method.csv", summaries)
    write_csv(DEST / "unique_experiment_units.csv", unique_summaries)
    return summaries


def bootstrap_ci_matrix(difference_matrix):
    rng = np.random.default_rng(SEED)
    n = difference_matrix.shape[1]
    draws = np.empty((difference_matrix.shape[0], BOOTSTRAP_REPLICATES), dtype=np.float64)
    chunk = 100
    for start in range(0, BOOTSTRAP_REPLICATES, chunk):
        size = min(chunk, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, n, size=(size, n), endpoint=False)
        draws[:, start:start + size] = difference_matrix[:, indices].mean(axis=2)
    return np.quantile(draws, [0.025, 0.975], axis=1)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def build_paired_statistics(data):
    descriptors, difference_rows, matrices = [], [], []
    accuracy_pairs = []
    interaction_descriptors, interaction_rows, interaction_matrices = [], [], []
    for family in FAMILIES:
        for method in METHODS:
            conditions = data[(family, method)]
            for comparison, high, low in COMPARISONS:
                for metric, source_key in METRICS.items():
                    differences = np.asarray([
                        float(conditions[high][qid][source_key]) - float(conditions[low][qid][source_key])
                        for qid in range(1103)
                    ], dtype=np.float64)
                    descriptors.append((family, method, comparison, high, low, metric))
                    matrices.append(differences)
                    for qid, value in enumerate(differences):
                        difference_rows.append({
                            "query_row_id": qid, "generation_family": family,
                            "retrieval_method": method, "comparison": comparison,
                            "metric": metric, "paired_difference": value,
                        })
                high_acc = np.asarray([int(conditions[high][qid]["emotion_accuracy"]) for qid in range(1103)])
                low_acc = np.asarray([int(conditions[low][qid]["emotion_accuracy"]) for qid in range(1103)])
                b = int(np.sum((low_acc == 0) & (high_acc == 1)))
                c = int(np.sum((low_acc == 1) & (high_acc == 0)))
                p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
                accuracy_pairs.append({
                    "generation_family": family, "retrieval_method": method,
                    "comparison": comparison, "n_pairs": 1103,
                    "low_wrong_high_correct": b, "low_correct_high_wrong": c,
                    "mcnemar_exact_p_raw": p,
                })

            for metric, source_key in METRICS.items():
                interaction = np.asarray([
                    (float(conditions["Full Oracle"][qid][source_key]) - float(conditions["Oracle Event"][qid][source_key]))
                    - (float(conditions["Oracle Emotion"][qid][source_key]) - float(conditions["Normal"][qid][source_key]))
                    for qid in range(1103)
                ], dtype=np.float64)
                interaction_descriptors.append((family, method, metric))
                interaction_matrices.append(interaction)
                for qid, value in enumerate(interaction):
                    interaction_rows.append({
                        "query_row_id": qid, "generation_family": family,
                        "retrieval_method": method, "metric": metric,
                        "interaction": value,
                    })

    matrix = np.vstack(matrices)
    ci = bootstrap_ci_matrix(matrix)
    paired = []
    for i, descriptor in enumerate(descriptors):
        family, method, comparison, high, low, metric = descriptor
        paired.append({
            "generation_family": family, "retrieval_method": method,
            "comparison": comparison, "higher_condition": high, "lower_condition": low,
            "metric": metric, "n_pairs": 1103,
            "point_estimate": float(matrix[i].mean()),
            "ci_95_low": float(ci[0, i]), "ci_95_high": float(ci[1, i]),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": SEED,
        })

    interaction_matrix = np.vstack(interaction_matrices)
    interaction_ci = bootstrap_ci_matrix(interaction_matrix)
    interactions = []
    for i, (family, method, metric) in enumerate(interaction_descriptors):
        interactions.append({
            "generation_family": family, "retrieval_method": method, "metric": metric,
            "definition": "(Full Oracle - Oracle Event) - (Oracle Emotion - Normal)",
            "n_pairs": 1103, "point_estimate": float(interaction_matrix[i].mean()),
            "ci_95_low": float(interaction_ci[0, i]), "ci_95_high": float(interaction_ci[1, i]),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": SEED,
        })

    adjusted = holm_adjust([x["mcnemar_exact_p_raw"] for x in accuracy_pairs])
    for row, p_adj in zip(accuracy_pairs, adjusted):
        row["holm_family_size"] = len(accuracy_pairs)
        row["mcnemar_exact_p_holm"] = float(p_adj)
        row["significant_holm_0_05"] = bool(p_adj < 0.05)

    write_csv(DEST / "paired_differences_summary.csv", paired)
    write_csv(DEST / "paired_differences_per_query.csv", difference_rows)
    write_csv(DEST / "interaction_summary.csv", interactions)
    write_csv(DEST / "interaction_per_query.csv", interaction_rows)
    write_csv(DEST / "emotion_accuracy_mcnemar.csv", accuracy_pairs)
    return paired, interactions, accuracy_pairs


def format_value(metric, value):
    return f"{value:.2%}" if metric == "emotion_accuracy" else f"{value:.4f}"


def build_tables(summaries, paired, interactions):
    lookup = {(r["generation_family"], r["retrieval_method"], r["generation_condition"]): r for r in summaries}
    paired_lookup = {(r["generation_family"], r["retrieval_method"], r["comparison"], r["metric"]): r for r in paired}
    interaction_lookup = {(r["generation_family"], r["retrieval_method"], r["metric"]): r for r in interactions}
    lines = ["# Oracle Event and Full Oracle Decomposition", ""]
    for family in FAMILIES:
        lines += [f"## {family}", ""]
        for method in METHODS:
            lines += [f"### {method}", "", "| Condition | Event | Emotion | N | Emotion Acc. | Target Score | Emotion Sim. | Original-text Entail. |", "|---|---|---|---:|---:|---:|---:|---:|"]
            for condition in CONDITIONS:
                r = lookup[(family, method, condition)]
                lines.append(f"| {condition} | {r['event_source']} | {r['emotion_source']} | {r['n']} | {r['emotion_accuracy']:.2%} | {r['target_emotion_score']:.4f} | {r['emotion_similarity']:.4f} | {r['original_text_entailment']:.4f} |")
            lines.append("")
    (DEST / "four_conditions_all_methods.md").write_text("\n".join(lines) + "\n")

    appendix = [
        "# Appendix E — Oracle Decomposition", "",
        f"All paired intervals use {BOOTSTRAP_REPLICATES:,} paired bootstrap resamples with seed `{SEED}`. Emotion accuracy additionally uses exact McNemar tests; 75 new McNemar rows are merged with legacy tests into 77 unique hypotheses for the final Holm correction. Full Oracle is one shared generation per family and is displayed under each retrieval method only as a comparison reference.", "",
    ]
    for family in FAMILIES:
        appendix += [f"## {family}", "", "### Four-condition means", "", "| Method | Condition | N | Emotion Acc. | Target | Similarity | Entailment |", "|---|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            for condition in CONDITIONS:
                r = lookup[(family, method, condition)]
                appendix.append(f"| {method} | {condition} | {r['n']} | {r['emotion_accuracy']:.2%} | {r['target_emotion_score']:.4f} | {r['emotion_similarity']:.4f} | {r['original_text_entailment']:.4f} |")
        appendix += ["", "### Paired differences", "", "| Method | Comparison | Metric | Difference | 95% CI |", "|---|---|---|---:|---:|"]
        for method in METHODS:
            for comparison, _, _ in COMPARISONS:
                for metric in METRICS:
                    r = paired_lookup[(family, method, comparison, metric)]
                    appendix.append(f"| {method} | {comparison} | {metric} | {format_value(metric, r['point_estimate'])} | [{format_value(metric, r['ci_95_low'])}, {format_value(metric, r['ci_95_high'])}] |")
        appendix += ["", "### 2×2 interaction", "", "| Method | Metric | Interaction | 95% CI |", "|---|---|---:|---:|"]
        for method in METHODS:
            for metric in METRICS:
                r = interaction_lookup[(family, method, metric)]
                appendix.append(f"| {method} | {metric} | {format_value(metric, r['point_estimate'])} | [{format_value(metric, r['ci_95_low'])}, {format_value(metric, r['ci_95_high'])}] |")
        appendix.append("")
    (DEST / "Appendix_E_oracle_decomposition.md").write_text("\n".join(appendix) + "\n")


def build_figures(paired):
    metric_titles = {
        "emotion_accuracy": "Emotion accuracy",
        "target_emotion_score": "Target emotion score",
        "emotion_similarity": "Emotion similarity",
        "original_text_entailment": "Original-text entailment",
    }
    short_comparisons = {
        "Oracle Emotion - Normal": "OE - N",
        "Oracle Event - Normal": "OEv - N",
        "Full Oracle - Oracle Emotion": "FO - OE",
        "Full Oracle - Oracle Event": "FO - OEv",
        "Full Oracle - Normal": "FO - N",
    }
    colors = dict(zip(METHODS, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]))
    for family in FAMILIES:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), constrained_layout=True)
        for ax, metric in zip(axes, METRICS):
            subset = [r for r in paired if r["generation_family"] == family and r["metric"] == metric]
            for method_index, method in enumerate(METHODS):
                rows = [r for r in subset if r["retrieval_method"] == method]
                x = np.arange(len(COMPARISONS)) + (method_index - 2) * 0.08
                y = np.asarray([r["point_estimate"] for r in rows])
                low = y - np.asarray([r["ci_95_low"] for r in rows])
                high = np.asarray([r["ci_95_high"] for r in rows]) - y
                ax.errorbar(x, y, yerr=np.vstack([low, high]), fmt="o", ms=4, capsize=2, lw=1, color=colors[method], label=method)
            ax.axhline(0, color="#555555", lw=0.8)
            ax.set_xticks(np.arange(len(COMPARISONS)))
            ax.set_xticklabels([short_comparisons[x[0]] for x in COMPARISONS], rotation=35, ha="right")
            ax.set_title(metric_titles[metric])
            ax.set_ylabel("Paired difference")
            ax.grid(axis="y", color="#dddddd", lw=0.6)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
        fig.suptitle(f"Oracle decomposition: {family} (95% paired bootstrap CI)", y=1.04)
        for extension in ("png", "pdf"):
            fig.savefig(DEST / f"figure6_{family}.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)


def build_paper_update(summaries, paired, interactions, mcnemar):
    lookup = {(r["generation_family"], r["retrieval_method"], r["generation_condition"]): r for r in summaries}
    paired_lookup = {(r["generation_family"], r["retrieval_method"], r["comparison"], r["metric"]): r for r in paired}
    paper_dir = ROOT / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EM-TRACE ARR 中文版草稿更新材料：Oracle Event / Full Oracle", "",
        "> 当前 Linux 工作区未找到原草稿 `EM_TRACE_ARR_中文版草稿.md`，因此本文件提供可直接合并的正式更新段落，没有擅自重建或覆盖原稿。", "",
        "## 4.3 Generation Conditions（替换/补充稿）", "",
        "我们对 Emotion Label、Multilayer Vector Steering 和 NPTI 三种生成机制均实施 2×2 oracle 分解。Normal 使用正常检索事件和正常恢复情绪；Oracle Emotion 保留正常检索的 factual memory，仅将控制情绪替换为测试样本的真实 emotion label；Oracle Event 仅将 factual memory 替换为目标事件，同时保留各 retrieval method 的正常情绪预测；Full Oracle 同时使用目标 factual memory 和真实 emotion label。新增条件沿用各自生成机制对应实验的 prompt 结构、最大生成长度和评测器；能够从生成配置直接核验的设置见实验 manifest。Full Oracle 每种生成机制只生成一次，并在五种 retrieval method 的配对分析中作为同一个共享上限，而非五个独立实验。", "",
        "## 5.2 Event Oracle and Full Oracle Decomposition", "",
        "下表列出每种生成机制中五种 retrieval method 的均值。完整逐方法结果、配对置信区间和交互项见 Appendix E。", "",
    ]
    for family in FAMILIES:
        lines += [f"### {family}", "", "| Method | Condition | Emotion Acc. | Target Score | Emotion Sim. | Original-text Entail. |", "|---|---|---:|---:|---:|---:|"]
        for method in METHODS:
            for condition in CONDITIONS:
                r = lookup[(family, method, condition)]
                lines.append(f"| {method} | {condition} | {r['emotion_accuracy']:.2%} | {r['target_emotion_score']:.4f} | {r['emotion_similarity']:.4f} | {r['original_text_entailment']:.4f} |")
        lines.append("")

    lines += ["## 结果解读", ""]
    for family in FAMILIES:
        event_gains = [paired_lookup[(family, m, "Oracle Event - Normal", "emotion_accuracy")]["point_estimate"] for m in METHODS]
        emotion_gains = [paired_lookup[(family, m, "Oracle Emotion - Normal", "emotion_accuracy")]["point_estimate"] for m in METHODS]
        full_gaps = [paired_lookup[(family, m, "Full Oracle - Normal", "emotion_accuracy")]["point_estimate"] for m in METHODS]
        dominant = "事件检索错误" if np.mean(event_gains) > np.mean(emotion_gains) else "情绪恢复错误"
        lines.append(f"- **{family}**：五种方法的 Oracle Event 平均准确率增益为 {np.mean(event_gains):+.2%}，Oracle Emotion 为 {np.mean(emotion_gains):+.2%}，因此该机制下更大的平均瓶颈来自{dominant}。Full Oracle 相对 Normal 的平均上限空间为 {np.mean(full_gaps):+.2%}。这里的跨方法平均仅作描述，不把五种方法视为独立样本；正式推断均基于同一 method 内的 1,103 个 query 配对。")
    lines += [
        "", "## 图 6", "",
        "三种机制分别生成 `figure6_emotion-label.*`、`figure6_multilayer-vector.*` 和 `figure6_npti.*`。每个图含四个指标分面，展示五种 retrieval method 的五类配对差值及 95% paired bootstrap CI。", "",
        "## Appendix E", "",
        "完整内容见 `derived/oracle_decomposition/Appendix_E_oracle_decomposition.md`，包括 60 行四条件方法表、300 个配对差值、60 个 interaction、75 个新增 exact McNemar 检验，以及与旧检验去重后的 77 个唯一正式假设的统一 Holm 校正。", "",
        "## 实验规模", "",
        "原七条件、35 个条件—方法组合保留用于 realization 主比较。三种生成机制各新增 5 个 Oracle Event 单元和 1 个共享 Full Oracle 单元，共新增 18 个唯一生成实验单元。因此扩展结果集包含 53 个唯一条件—方法实验单元，而不是把 Full Oracle 复制五次后写成 65 个独立单元。概念层面可称 13 种 generation conditions，但必须同时注明其中三个 Full Oracle 条件分别在各机制内跨 retrieval method 共享。", "",
        "## 摘要、Introduction 与 Conclusion", "",
        "应根据上述三种机制分别陈述，不能预先统一断言事件或情绪一定是主要瓶颈。可使用“2×2 oracle 分解显示，不同控制机制的主要误差来源不同；完整事件与情绪信息共同定义了显著但机制依赖的生成上限”作为保守总述。", "",
    ]
    (paper_dir / "Oracle_Event_Full_Oracle_更新材料.md").write_text("\n".join(lines) + "\n")


def build_manifest(copied, summaries, paired, interactions, mcnemar):
    files = []
    for item in copied:
        group_dir = SOURCE_ROOT / item["source_group"]
        item["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(group_dir.iterdir()) if p.is_file()}
    files = sorted(str(p.relative_to(ROOT)) for p in DEST.iterdir() if p.is_file())
    manifest = {
        "generated_on": "2026-09-29",
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "families": FAMILIES,
        "retrieval_methods": METHODS,
        "conditions": CONDITIONS,
        "new_source_groups": copied,
        "display_rows": len(summaries),
        "new_unique_experiment_units": 18,
        "total_unique_experiment_units_including_previous_35": 53,
        "full_oracle_policy": "One shared generation per family; repeated only as a method-level comparison reference.",
        "paired_summary_rows": len(paired),
        "interaction_rows": len(interactions),
        "mcnemar_tests": len(mcnemar),
        "holm_family_size": 77,
        "new_mcnemar_rows": len(mcnemar),
        "legacy_mcnemar_rows": 12,
        "duplicate_legacy_new_hypotheses": 10,
        "holm_unique_hypothesis_count": 77,
        "outputs": files,
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    copied = collect_new_sources()
    data, provenance = load_condition_data(copied)
    summaries = build_standardized_and_summary(data)
    paired, interactions, mcnemar = build_paired_statistics(data)
    write_csv(DEST / "source_provenance.csv", provenance)
    build_tables(summaries, paired, interactions)
    build_figures(paired)
    build_paper_update(summaries, paired, interactions, mcnemar)
    build_manifest(copied, summaries, paired, interactions, mcnemar)


if __name__ == "__main__":
    main()
