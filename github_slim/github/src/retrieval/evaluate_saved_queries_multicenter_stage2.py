#!/usr/bin/env python3
"""Evaluate saved strict query vectors with a multi-prototype Stage-2 bank."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAGE1 = ROOT / "use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt"
QUERY_VECTORS = ROOT / "use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt"
QUERY_META = ROOT / "use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage2-bank", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage1-bank", type=Path, default=STAGE1)
    p.add_argument("--query-vectors", type=Path, default=QUERY_VECTORS)
    p.add_argument("--query-metadata", type=Path, default=QUERY_META)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    a = parse_args(); output = a.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    s1 = torch.load(a.stage1_bank, map_location="cpu", weights_only=False)
    s2 = torch.load(a.stage2_bank, map_location="cpu", weights_only=False)
    query = torch.load(a.query_vectors, map_location="cpu", weights_only=True).float()
    meta = read_jsonl(a.query_metadata); labels = list(s2["manifest"]["labels"])
    if query.shape != (len(meta), 1024) or len(meta) != 1103: raise ValueError("unexpected saved strict-query input")
    fact = query @ s1["fact_keys"].float().T
    stage1_score, stage1_index = fact.max(1)
    cues = s1["emotion_cue_values"].float()[stage1_index]
    prototype_scores = cues @ s2["emotion_prototype_keys"].float().T
    class_indices = s2["prototype_class_indices"].tolist()
    class_scores = torch.full((len(meta), len(labels)), float("-inf"))
    for prototype_index, class_index in enumerate(class_indices): class_scores[:, class_index] = torch.maximum(class_scores[:, class_index], prototype_scores[:, prototype_index])
    ranked_score, ranked_class = class_scores.topk(len(labels), dim=1)
    result=[]
    for i,(m,s1i,s1s,classes,scores) in enumerate(zip(meta,stage1_index.tolist(),stage1_score.tolist(),ranked_class.tolist(),ranked_score.tolist())):
        r=s1["records"][s1i]; ranked=[labels[x] for x in classes]
        result.append({"query_row_id":i,"memory_id":m["memory_id"],"target_emotion":m["target_emotion"],"stage1_top1_memory_id":r["memory_id"],"stage1_top1_retrieved_emotion":r["emotion_label"],"stage1_top1_memory_source":r["memory_source"],"stage1_fact_cosine_similarity":s1s,"stage1_retrieved_emotion_match":int(r["emotion_label"]==m["target_emotion"]),"stage2_predicted_emotion":ranked[0],"stage2_ranked_emotions":ranked,"stage2_ranked_class_scores":scores})
    def scores(group: list[dict[str,Any]]) -> dict[str,float]:
        return {f"stage2_emotion_accuracy_at{k}":sum(x["target_emotion"] in x["stage2_ranked_emotions"][:k] for x in group)/len(group) for k in (1,3,5)}
    summary={"n":len(result),"stage1_retrieved_emotion_match_at1":sum(x["stage1_retrieved_emotion_match"] for x in result)/len(result),**scores(result),"protocol":"saved evaluation_query Qwen3 vectors -> Stage-1 Top-1 -> retrieved cue -> multi-prototype Stage-2 class max cosine","stage2_manifest":s2["manifest"]}
    write_jsonl(output/'per_query_retrieval.jsonl',result)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rows=[]
    for label in labels:
        group=[x for x in result if x['target_emotion']==label]
        rows.append({'emotion':label,'n':len(group),'stage1_retrieved_emotion_match_at1':sum(x['stage1_retrieved_emotion_match'] for x in group)/len(group),**scores(group)})
    with (output/'per_emotion.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    confusion=Counter((x['target_emotion'],x['stage2_predicted_emotion']) for x in result)
    with (output/'confusion_matrix.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['target_emotion',*labels]);w.writeheader()
        for target in labels:w.writerow({'target_emotion':target,**{pred:confusion[target,pred] for pred in labels}})
    print(json.dumps({'status':'complete','output':str(output),**{k:v for k,v in summary.items() if k!='stage2_manifest'}},ensure_ascii=False))


if __name__ == '__main__': main()
