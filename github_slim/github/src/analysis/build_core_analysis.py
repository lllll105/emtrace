#!/usr/bin/env python3
"""Build the confirmed EM-TRACE RQ1/RQ2 core analyses."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
EVAL_ROOT = PROJECT_ROOT / "outputs" / "eval"

GROUPS = {
    "01_fact_prompt_only": "eval_fact_prompt_only_20260925",
    "02_memory_retrieved_emotion_label": "eval_memory_retrieved_emotion_label_20260925",
    "03_memory_oracle_emotion_label": "eval_memory_oracle_emotion_label_20260925",
    "04_normal_multilayer_steering": "eval_normal_retrieval_vector_steering_20260923",
    "05_oracle_multilayer_steering": "eval_oracle_retrieval_vector_steering_20260923",
    "06_normal_npti": "eval_normal_retrieval_npti_20260923",
    "07_oracle_npti": "eval_oracle_retrieval_npti_20260923",
}

METHOD_ORDER = ["fact-only", "emotion-only", "concat-sum", "similarity-product", "two-stage"]
METHOD_ALIASES = {
    "fact_only": "fact-only", "emotion_only": "emotion-only",
    "concat_sum": "concat-sum", "similarity_product": "similarity-product",
    "two_stage": "two-stage", "two_stage_fact_then_cue_prototype": "two-stage",
    "two_stage_fact_then_m2_prototype": "two-stage",
}


def norm_method(value):
    value = (value or "").strip()
    return METHOD_ALIASES.get(value, value.replace("_", "-"))


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
        writer.writeheader(); writer.writerows(rows)


def collect_sources():
    source_root = ROOT / "source_results"
    source_root.mkdir(parents=True, exist_ok=True)
    for group, eval_name in GROUPS.items():
        src, dst = EVAL_ROOT / eval_name, source_root / group
        if not src.is_dir(): raise FileNotFoundError(src)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("summary.csv", "config.json", "per_generation_metrics.jsonl"):
            source = src / name
            if not source.is_file(): raise FileNotFoundError(source)
            shutil.copy2(source, dst / name)


def build_rq3_summary():
    rows=[]
    for group in GROUPS:
        if group in ("06_normal_npti", "07_oracle_npti"):
            buckets=defaultdict(list)
            for row in read_jsonl(ROOT/"source_results"/group/"per_generation_metrics.jsonl"):
                if row.get("condition") == "baseline":
                    continue
                method=norm_method(row.get("retrieval_method"))
                if method in METHOD_ORDER: buckets[method].append(row)
            for method in METHOD_ORDER:
                values=buckets[method]; n=len(values)
                rows.append({"group":group,"method":method,"n":n,
                    "emotion_accuracy":sum(float(x["emotion_accuracy"]) for x in values)/n,
                    "target_emotion_score":sum(float(x["emotion_score_1"]) for x in values)/n,
                    "emotion_similarity":sum(float(x["emotion_similarity_1"]) for x in values)/n,
                    "emotion_similarity_11class":sum(float(x["emotion_similarity_11class"]) for x in values)/n,
                    "fact_consistency":sum(float(x["fact_consistency_score_1"]) for x in values)/n})
            continue
        with (ROOT/"source_results"/group/"summary.csv").open() as handle:
            for row in csv.DictReader(handle):
                if row.get("scope") != "micro" or row.get("emotion") != "all": continue
                method=norm_method(row.get("condition"))
                if method == "baseline": continue
                rows.append({"group":group,"method":method,"n":row["n"],
                    "emotion_accuracy":row["emotion_accuracy"],"target_emotion_score":row["emotion_score_1"],
                    "emotion_similarity":row["emotion_similarity_1"],
                    "emotion_similarity_11class":row["emotion_similarity_11class"],
                    "fact_consistency":row["fact_consistency_score_1"]})
    order={m:i for i,m in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda r:(r["group"],order.get(r["method"],99)))
    write_csv(ROOT/"derived/RQ3_emotion_realization/all_7_groups_micro.csv",rows)


def load_summary(group):
    result={}
    if group in ("06_normal_npti", "07_oracle_npti"):
        buckets=defaultdict(list)
        for row in read_jsonl(ROOT/"source_results"/group/"per_generation_metrics.jsonl"):
            if row.get("condition") == "baseline":
                continue
            method=norm_method(row.get("retrieval_method"))
            if method in METHOD_ORDER: buckets[method].append(row)
        for method,values in buckets.items():
            n=len(values)
            result[method]={
                "emotion_accuracy":sum(float(x["emotion_accuracy"]) for x in values)/n,
                "fact_consistency_score_1":sum(float(x["fact_consistency_score_1"]) for x in values)/n,
            }
        return result
    with (ROOT/"source_results"/group/"summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row.get("scope")=="micro" and row.get("emotion")=="all":
                method=norm_method(row.get("condition"))
                if method != "baseline": result[method]=row
    return result


def build_normal_oracle_gain():
    rows=[]
    for family,ng,og in (("multilayer","04_normal_multilayer_steering","05_oracle_multilayer_steering"),("npti","06_normal_npti","07_oracle_npti")):
        normal,oracle=load_summary(ng),load_summary(og)
        for method in METHOD_ORDER:
            a,b=normal[method],oracle[method]
            na,oa=float(a["emotion_accuracy"]),float(b["emotion_accuracy"])
            nf,of=float(a["fact_consistency_score_1"]),float(b["fact_consistency_score_1"])
            rows.append({"family":family,"method":method,"normal_emotion_accuracy":na,
                "oracle_emotion_accuracy":oa,"emotion_accuracy_gain":oa-na,
                "normal_fact_consistency":nf,"oracle_fact_consistency":of,"fact_consistency_change":of-nf})
    write_csv(ROOT/"derived/RQ2_error_propagation/normal_vs_oracle_gain.csv",rows)


def build_binding_analysis():
    path=ROOT/"source_results/04_normal_multilayer_steering/per_generation_metrics.jsonl"
    stats=defaultdict(Counter)
    quality=defaultdict(lambda:defaultdict(float))
    for row in read_jsonl(path):
        method=norm_method(row.get("retrieval_method") or row.get("condition"))
        if method not in METHOD_ORDER: continue
        event_ok=bool(row.get("memory_exact_top1"))
        final_emotion=row.get("injected_emotion") or row.get("retrieved_emotion")
        affect_ok=final_emotion==row.get("target_emotion")
        state=("Event+ Affect+" if event_ok and affect_ok else "Event+ Affect-" if event_ok else "Event- Affect+" if affect_ok else "Event- Affect-")
        s=stats[method]; s["n"]+=1; s["event_correct"]+=event_ok; s["binding_correct"]+=event_ok and affect_ok; s[state]+=1
        q=quality[(method,state)]; q["n"]+=1
        q["emotion_accuracy_sum"]+=float(row.get("emotion_accuracy") or 0)
        q["target_emotion_score_sum"]+=float(row.get("target_emotion_score") or row.get("emotion_score_1") or 0)
        q["fact_consistency_sum"]+=float(row.get("fact_consistency_score_1") or 0)
    recall=[]; quadrants=[]; binding=[]
    for method in METHOD_ORDER:
        s=stats[method]; n=s["n"]
        recall.append({"method":method,"n":n,"exact_recall_count":s["event_correct"],"exact_recall_at_1":s["event_correct"]/n})
        quadrants.append({"method":method,"n":n,"event_pos_affect_pos":s["Event+ Affect+"],"event_pos_affect_neg":s["Event+ Affect-"],"event_neg_affect_pos":s["Event- Affect+"],"event_neg_affect_neg":s["Event- Affect-"]})
        binding.append({"method":method,"n":n,"binding_count":s["binding_correct"],"binding_at_1":s["binding_correct"]/n})
    quality_rows=[]
    for method in METHOD_ORDER:
        for state in ("Event+ Affect+","Event+ Affect-","Event- Affect+","Event- Affect-"):
            q=quality[(method,state)]; n=int(q["n"])
            quality_rows.append({"method":method,"state":state,"n":n,
                "emotion_accuracy":q["emotion_accuracy_sum"]/n if n else "",
                "target_emotion_score":q["target_emotion_score_sum"]/n if n else "",
                "fact_consistency":q["fact_consistency_sum"]/n if n else ""})
    dest=ROOT/"derived/RQ1_binding"
    write_csv(dest/"exact_recall_at_1.csv",recall); write_csv(dest/"event_affect_quadrants.csv",quadrants)
    write_csv(dest/"binding_at_1.csv",binding); write_csv(dest/"quadrant_generation_quality.csv",quality_rows)
    md=["# RQ1 — Event–Affect Binding","","定义：Event✓ = memory_exact_top1；Affect✓ = 最终恢复/注入的 emotion label 等于 target_emotion。","","## Exact Memory Recall@1 and Binding@1","","| Method | Exact Recall@1 | Binding@1 |","|---|---:|---:|"]
    for r,b in zip(recall,binding): md.append(f"| {r['method']} | {r['exact_recall_at_1']:.2%} ({r['exact_recall_count']}/{r['n']}) | {b['binding_at_1']:.2%} ({b['binding_count']}/{b['n']}) |")
    md += ["","## Event–Affect 四象限","","| Method | Event✓ Affect✓ | Event✓ Affect✗ | Event✗ Affect✓ | Event✗ Affect✗ |","|---|---:|---:|---:|---:|"]
    for r in quadrants: md.append(f"| {r['method']} | {r['event_pos_affect_pos']} | {r['event_pos_affect_neg']} | {r['event_neg_affect_pos']} | {r['event_neg_affect_neg']} |")
    (dest/"RQ1_core_results.md").write_text("\n".join(md)+"\n")


def build_two_stage_transition():
    path=PROJECT_ROOT/"outputs/normal_retrieval_npti_20260923/shard_00_of_01/two_stage_fact_then_m2_prototype/retrieval.jsonl"
    if not path.is_file():
        path=PROJECT_ROOT/"outputs/npti_helpful_prompt_20260927/normal/shard_00_of_01/two_stage_fact_then_m2_prototype/retrieval.jsonl"
    counts=Counter(); total=0
    for row in read_jsonl(path):
        target=row.get("target_emotion") or row.get("emotion_label")
        stage1=row.get("retrieved_emotion_from_memory") or row.get("memory_emotion")
        stage2_match=row.get("stage2_emotion_match")
        if stage1 is None or stage2_match is None or target is None:
            raise KeyError(f"Missing two-stage fields in {path}: keys={sorted(row)}")
        s1=stage1==target; s2=bool(stage2_match)
        key=("stage1_correct_stage2_correct" if s1 and s2 else "stage1_correct_stage2_wrong" if s1 else "stage1_wrong_stage2_correct" if s2 else "stage1_wrong_stage2_wrong")
        counts[key]+=1; total+=1
    keys=("stage1_correct_stage2_correct","stage1_correct_stage2_wrong","stage1_wrong_stage2_correct","stage1_wrong_stage2_wrong")
    rows=[{"transition":k,"count":counts[k],"rate_all":counts[k]/total} for k in keys]
    wrong=counts[keys[2]]+counts[keys[3]]; correct=counts[keys[0]]+counts[keys[1]]
    summary={"n":total,"stage1_correct":correct,"stage1_wrong":wrong,"stage2_correct":counts[keys[0]]+counts[keys[2]],
        "correction_rate_given_stage1_wrong":counts[keys[2]]/wrong if wrong else None,
        "corruption_rate_given_stage1_correct":counts[keys[1]]/correct if correct else None}
    dest=ROOT/"derived/RQ2_error_propagation"; write_csv(dest/"two_stage_emotion_transition.csv",rows)
    (dest/"two_stage_transition_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    md=["# RQ2 — Two-stage Emotion Error Transition","",
        "一级正确性由 retrieved_emotion_from_memory 与 target_emotion 比较；二级正确性直接使用原始记录的 stage2_emotion_match。","",
        "| 一级结果 | 二级结果 | 数量 | 占全部样本 |","|---|---|---:|---:|",
        f"| 正确 | 正确 | {counts[keys[0]]} | {counts[keys[0]]/total:.2%} |",f"| 正确 | 错误 | {counts[keys[1]]} | {counts[keys[1]]/total:.2%} |",
        f"| 错误 | 正确（纠正） | {counts[keys[2]]} | {counts[keys[2]]/total:.2%} |",f"| 错误 | 错误 | {counts[keys[3]]} | {counts[keys[3]]/total:.2%} |","",
        f"- Correction rate P(stage2 correct | stage1 wrong): **{summary['correction_rate_given_stage1_wrong']:.2%}**",
        f"- Corruption rate P(stage2 wrong | stage1 correct): **{summary['corruption_rate_given_stage1_correct']:.2%}**"]
    (dest/"RQ2_two_stage_transition.md").write_text("\n".join(md)+"\n")


def build_manifest():
    manifest={"generated_on":"2026-09-29","scope":"legacy seven-group core analysis only","groups":[]}
    for group,eval_name in GROUPS.items():
        directory=ROOT/"source_results"/group
        manifest["groups"].append({"group":group,"source_directory":str(EVAL_ROOT/eval_name),"files":{p.name:p.stat().st_size for p in sorted(directory.iterdir()) if p.is_file()}})
    (ROOT/"derived/legacy_core_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")


def main():
    collect_sources(); build_rq3_summary(); build_normal_oracle_gain(); build_binding_analysis(); build_two_stage_transition(); build_manifest()


if __name__ == "__main__": main()
