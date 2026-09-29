#!/usr/bin/env python3
"""Recall-only comparison: concat-sum, similarity-product, and two-stage retrieval."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

ROOT=Path(__file__).resolve().parents[2]
STAGE1=ROOT/'use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt'
QV=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt'
QM=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl'
MODERNBERT='./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487'
S2=ROOT/'use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt'
OUT=ROOT/'use_2rag/evaluation/three_retrieval_recall_20260922'
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=OUT);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device',default='cpu');a=p.parse_args()
 s1=torch.load(STAGE1,map_location='cpu',weights_only=False); q=torch.load(QV,map_location='cpu',weights_only=True).float(); meta=[json.loads(x) for x in QM.read_text().splitlines() if x.strip()]
 try: tok=AutoTokenizer.from_pretrained(MODERNBERT,local_files_only=True)
 except ValueError: tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(MODERNBERT)/'tokenizer.json'),unk_token='[UNK]',sep_token='[SEP]',pad_token='[PAD]',cls_token='[CLS]',mask_token='[MASK]')
 model=AutoModel.from_pretrained(MODERNBERT,local_files_only=True).to(a.device).eval()
 cues=[]
 with torch.inference_mode():
  for st in range(0,len(meta),a.batch_size):
   enc=tok([x['evaluation_query'] for x in meta[st:st+a.batch_size]],padding=True,truncation=True,max_length=128,return_tensors='pt')
   enc={k:v.to(a.device) for k,v in enc.items() if k in ('input_ids','attention_mask')}; h=model(**enc,output_hidden_states=True,return_dict=True).hidden_states[22]; m=enc['attention_mask'].to(h.dtype).unsqueeze(-1); cues.append(F.normalize((h*m).sum(1)/m.sum(1).clamp_min(1),p=2,dim=1).cpu())
 c=torch.cat(cues); fact=s1['fact_keys'].float(); cue=s1['emotion_cue_values'].float(); labels=[x['target_emotion'] for x in meta]; ids=[x['memory_id'] for x in meta]; records=s1['records']
 sf=q@fact.T; se=c@cue.T
 s2=torch.load(S2,map_location='cpu',weights_only=False); proto=cue # placeholder
 # Three scored rankings; two-stage ranking comes from saved exact protocol using retrieved cue and M=2 prototypes.
 pscore=c@cue.T
 scores={'concat_sum':sf+se,'similarity_product':sf*se}
 # products can be negative; ranking remains direct descending as specified
 rows=[]
 for method,scoremat in scores.items():
  top=scoremat.topk(5,dim=1).indices
  for i in range(len(meta)):
   rr=[records[j] for j in top[i].tolist()]; rows.append({'query_row_id':i,'memory_id':ids[i],'target_emotion':labels[i],'method':method,'ranked_memory_ids':[x['memory_id'] for x in rr],'ranked_emotions':[x['emotion_label'] for x in rr],'top1_memory_exact':int(rr[0]['memory_id']==ids[i]),'top1_emotion_match':int(rr[0]['emotion_label']==labels[i]),'emotion_hit3':int(labels[i] in [x['emotion_label'] for x in rr[:3]]),'emotion_hit5':int(labels[i] in [x['emotion_label'] for x in rr[:5]])})
 # two-stage metrics are copied from the already validated strict M=2 retrieval output
 twopath=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2_multicenter_geometric_m2/per_query_retrieval.jsonl'; two=[json.loads(x) for x in twopath.read_text().splitlines() if x.strip()]
 for x in two:
  ranked=x['stage2_ranked_emotions']; rows.append({'query_row_id':x['query_row_id'],'memory_id':x['memory_id'],'target_emotion':x['target_emotion'],'method':'two_stage_fact_then_m2_prototype','ranked_memory_ids':[x['stage1_top1_memory_id']],'ranked_emotions':ranked,'top1_memory_exact':int(x['stage1_top1_memory_id']==x['memory_id']),'top1_emotion_match':int(x['stage2_predicted_emotion']==x['target_emotion']),'emotion_hit3':int(x['target_emotion'] in ranked[:3]),'emotion_hit5':int(x['target_emotion'] in ranked[:5])})
 a.output_dir.mkdir(parents=True,exist_ok=True); rows.sort(key=lambda x:(x['query_row_id'],x['method'])); (a.output_dir/'per_query.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows))
 report=[]
 for method in ('concat_sum','similarity_product','two_stage_fact_then_m2_prototype'):
  g=[x for x in rows if x['method']==method]; report.append({'method':method,'n':len(g),'memory_exact_top1':sum(x['top1_memory_exact'] for x in g)/len(g),'emotion_match_at1':sum(x['top1_emotion_match'] for x in g)/len(g),'emotion_hit_at3':sum(x['emotion_hit3'] for x in g)/len(g),'emotion_hit_at5':sum(x['emotion_hit5'] for x in g)/len(g)})
 with (a.output_dir/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
 print(json.dumps({'status':'complete','output':str(a.output_dir),'summary':report},ensure_ascii=False))
if __name__=='__main__':main()
