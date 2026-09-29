#!/usr/bin/env python3
"""Post-hoc memory recall versus generation quality analysis."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
from statistics import mean

METHODS=('fact_only','emotion_only','concat_sum','similarity_product','two_stage_fact_then_cue_prototype')
ALIASES={'two_stage_fact_then_m2_prototype':'two_stage_fact_then_cue_prototype'}
EVAL_DIR_PREFIXES=('eval_fact_prompt_only_','eval_memory_retrieved_emotion_label_','eval_normal_retrieval_vector_steering_','eval_oracle_retrieval_vector_steering_','eval_normal_retrieval_npti_','eval_oracle_retrieval_npti_')
def jl(p):
 return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def key(x): return str(x.get('memory_id',x.get('target_memory_id',x.get('query_row_id'))))
def num(x, names):
 for n in names:
  if x.get(n) is not None:
   try:return float(x[n])
   except:pass
 return None
def method(x,p): return ALIASES.get(str(x.get('retrieval_method') or p.parent.name),str(x.get('retrieval_method') or p.parent.name))
def csvout(p,rows,fields=None):
 p.parent.mkdir(parents=True,exist_ok=True); fields=list(fields or [])
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def avg(rs,k):
 v=[x[k] for x in rs if x.get(k) is not None];return mean(v) if v else None
def spearman(a,b):
 z=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
 if len(z)<3:return None
 def rk(v):
  o=sorted(range(len(v)),key=v.__getitem__);r=[0.]*len(v);i=0
  while i<len(v):
   j=i
   while j+1<len(v) and v[o[j+1]]==v[o[i]]:j+=1
   for q in range(i,j+1):r[o[q]]=(i+j)/2+1
   i=j+1
  return r
 x,y=rk([q[0] for q in z]),rk([q[1] for q in z]);mx,my=mean(x),mean(y);dx=sum((q-mx)**2 for q in x);dy=sum((q-my)**2 for q in y)
 return sum((q-mx)*(r-my) for q,r in zip(x,y))/math.sqrt(dx*dy) if dx and dy else 0.
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--outputs-dir',type=Path,default=Path('outputs'));ap.add_argument('--output-dir',type=Path,default=Path('use_2rag/plus/results'));a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 retrieval=defaultdict(dict); generations=defaultdict(dict)
 for p in a.outputs_dir.rglob('retrieval.jsonl'):
  for x in jl(p):
   m=method(x,p)
   if m not in METHODS:continue
   k=key(x);retrieval[m][k]={'sample_id':k,'target_memory_id':k,'target_emotion':x.get('target_emotion'),'retrieved_memory_id':str(x.get('retrieved_memory_id',x.get('stage1_top1_memory_id',''))),'retrieved_emotion':x.get('retrieved_emotion',x.get('stage2_predicted_emotion',x.get('stage1_top1_retrieved_emotion'))),'stage1_retrieved_emotion':x.get('stage1_top1_retrieved_emotion',x.get('retrieved_emotion_from_memory')),'stage2_predicted_emotion':x.get('stage2_predicted_emotion'),'fact_similarity':num(x,('fact_score','stage1_fact_cosine_similarity','retrieved_similarity')),'emotion_similarity':num(x,('emotion_score','stage2_top1_similarity'))}
 for p in a.outputs_dir.rglob('*.jsonl'):
  if p.name not in ('per_generation_metrics.jsonl','scored_results.jsonl','generations.jsonl','retrieved_fact_prompt_npti_generations.jsonl'):continue
  # Generation quality must come from the complete evaluation outputs; raw
  # generation files would duplicate conditions and have incomplete metrics.
  if 'eval' not in p.parts or not any(part.startswith(EVAL_DIR_PREFIXES) for part in p.parts): continue
  for x in jl(p):
   m=method(x,p)
   if m not in METHODS:continue
   c=str(x.get('condition',p.parent.name));pathstr=str(p).lower();kind='npti' if 'npti' in pathstr else ('vector' if 'vector' in pathstr or 'steering' in pathstr else 'unknown');source='oracle' if 'oracle' in pathstr else ('normal' if 'normal' in pathstr else 'unknown');k=key(x);q={'sample_id':k,'condition':c,'source_type':source,'injection_kind':kind,'emotion_accuracy':num(x,('emotion_accuracy',)),'target_emotion_score':num(x,('target_emotion_score','emotion_score_1')),'fact_consistency_score_1':num(x,('fact_consistency_score_1','nli_entailment_probability')),'source':str(p),'priority':2 if p.name in ('per_generation_metrics.jsonl','scored_results.jsonl') else 1};old=generations[m].get((k,c,source,kind))
   if old is None or q['priority']>=old['priority']:generations[m][(k,c,source,kind)]=q
 samples=[]
 for m in METHODS:
  for k,r in retrieval[m].items():
   r=dict(r);r['retrieval_method']=m;r['exact_match']=int(r['retrieved_memory_id']==r['target_memory_id']);r['emotion_match']=int(r['retrieved_emotion']==r['target_emotion']);r['recall_group']='A_exact_memory' if r['exact_match'] else ('B_same_emotion_different_memory' if r['emotion_match'] else 'C_wrong_emotion')
   gs=[g for (kk,_,_,_),g in generations[m].items() if kk==k]
   samples.extend([{**r,**{z:g.get(z) for z in ('condition','emotion_accuracy','target_emotion_score','fact_consistency_score_1','source')}} for g in gs] or [r])
 csvout(a.output_dir/'sample_level_analysis.csv',samples)
 t1=[]
 for m in METHODS:
  q=[dict(x) for x in retrieval[m].values()];
  for x in q:x['exact_match']=int(x['retrieved_memory_id']==x['target_memory_id']);x['emotion_match']=int(x['retrieved_emotion']==x['target_emotion']);x['recall_group']='A_exact_memory' if x['exact_match'] else ('B_same_emotion_different_memory' if x['emotion_match'] else 'C_wrong_emotion')
  t1.append({'retrieval_method':m,'n':len(q),'exact_match_rate':avg(q,'exact_match'),'emotion_match_rate':avg(q,'emotion_match'),'A_exact_memory':sum(x['recall_group']=='A_exact_memory' for x in q),'B_same_emotion_different_memory':sum(x['recall_group']=='B_same_emotion_different_memory' for x in q),'C_wrong_emotion':sum(x['recall_group']=='C_wrong_emotion' for x in q)})
 csvout(a.output_dir/'table1_retrieval_quality.csv',t1)
 t2=[]
 for m in METHODS:
  for c in sorted({x.get('condition') for x in samples if x['retrieval_method']==m and x.get('condition')}):
   q=[x for x in samples if x['retrieval_method']==m and x.get('condition')==c]
   for g in ('A_exact_memory','B_same_emotion_different_memory','C_wrong_emotion'):
    z=[x for x in q if x['recall_group']==g];t2.append({'retrieval_method':m,'condition':c,'recall_group':g,'n':len(z),'emotion_accuracy':avg(z,'emotion_accuracy'),'emotion_score':avg(z,'target_emotion_score'),'fact_consistency':avg(z,'fact_consistency_score_1')})
 csvout(a.output_dir/'table2_generation_by_recall_group.csv',t2)
 t3=[]
 for m in METHODS:
  q=sorted([x for x in samples if x['retrieval_method']==m and x.get('fact_similarity') is not None],key=lambda x:x['fact_similarity'],reverse=True);n=len(q);cuts=(n+2)//3
  for name,z in (('high',q[:cuts]),('middle',q[cuts:2*cuts]),('low',q[2*cuts:])):t3.append({'retrieval_method':m,'similarity_group':name,'n':len(z),'mean_fact_similarity':avg(z,'fact_similarity'),'emotion_accuracy':avg(z,'emotion_accuracy'),'emotion_score':avg(z,'target_emotion_score'),'exact_match':avg(z,'exact_match')})
  t3.append({'retrieval_method':m,'similarity_group':'spearman','n':n,'spearman_fact_vs_emotion_score':spearman([x.get('fact_similarity') for x in q],[x.get('target_emotion_score') for x in q])})
 csvout(a.output_dir/'table3_similarity_analysis.csv',t3)
 t4=[]
 for m in METHODS:
  n={};o={}
  for (k,c,s,kind),x in generations[m].items():
   if s=='oracle':o[(k,kind)]=x
   elif s=='normal':n[(k,kind)]=x
  for k,kind in set(n)&set(o):
   d=next((x for x in samples if x['retrieval_method']==m and x['sample_id']==k),{});t4.append({'retrieval_method':m,'injection_kind':kind,'sample_id':k,'emotion_retrieval_correct':d.get('emotion_match'),'normal_condition':n[(k,kind)].get('condition'),'oracle_condition':o[(k,kind)].get('condition'),'delta_emotion_score':None if n[(k,kind)].get('target_emotion_score') is None or o[(k,kind)].get('target_emotion_score') is None else o[(k,kind)]['target_emotion_score']-n[(k,kind)]['target_emotion_score'],'normal_accuracy':n[(k,kind)].get('emotion_accuracy'),'oracle_accuracy':o[(k,kind)].get('emotion_accuracy'),'normal_fact_consistency':n[(k,kind)].get('fact_consistency_score_1'),'oracle_fact_consistency':o[(k,kind)].get('fact_consistency_score_1')})
 csvout(a.output_dir/'table4_normal_oracle_pairs.csv',t4,fields=['retrieval_method','sample_id','emotion_retrieval_correct','normal_condition','oracle_condition','delta_emotion_score','normal_accuracy','oracle_accuracy','normal_fact_consistency','oracle_fact_consistency'])
 (a.output_dir/'analysis_report.md').write_text('# Memory Recall Quality -> Generation Quality Analysis\n\nThis is post-hoc analysis; no generation was run. Missing metrics remain empty.\n\n- `table1_retrieval_quality.csv`: retrieval identity/emotion quality and A/B/C counts.\n- `table2_generation_by_recall_group.csv`: generation metrics by recall group.\n- `table3_similarity_analysis.csv`: high/middle/low similarity groups and Spearman correlation.\n- `table4_normal_oracle_pairs.csv`: sample-level Normal-Oracle differences.\n',encoding='utf-8')
 print(json.dumps({'status':'complete','output_dir':str(a.output_dir),'sample_rows':len(samples),'table1':len(t1),'table2':len(t2),'table3':len(t3),'table4':len(t4)},ensure_ascii=False))
if __name__=='__main__':
 from recompute_eval_analysis import main as recompute_main
 recompute_main()
