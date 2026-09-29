#!/usr/bin/env python3
"""Generation eval for concat-sum or similarity-product joint retrieval.

Both methods retrieve one of the 8,030 original-bank rows, put its factual
memory in the prompt, and inject the original-bank value for its emotion label.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/phase3_test_evaluation/retrieval_eval'))
from run_qwen_multilayer_11class_retrieval_eval import MultilayerSteering, append_jsonl, configured_vectors, load_classifier, load_qwen, read_jsonl, token_inputs, write_jsonl # noqa:E402
BANK=ROOT/'use/build/multilayer_11class_trainplus_test_memory/qwen_multilayer_memory_bank_regenerated_classified_6927.pt'
QV=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt'
QM=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl'
MODERNBERT='./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487'
def args():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=('fact_only','emotion_only','concat_sum','similarity_product'),required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--max-new-tokens',type=int,default=96);p.add_argument('--classifier-batch-size',type=int,default=64);return p.parse_args()
def messages(q:str,fact:str|None):
 text=f'User query:\n{q}\n\nGenerate a helpful response.' if fact is None else f'Retrieved factual memory (use it as context when relevant; do not mention this retrieval process):\n{fact}\n\nUser query:\n{q}\n\nGenerate a helpful response.'
 return [{'role':'user','content':text}]
@torch.inference_mode()
def generate(tok,model,controller,q,fact,vectors,n):
 inp=token_inputs(tok,messages(q,fact),next(model.parameters()).device); stop={x for x in (tok.eos_token_id,tok.pad_token_id,tok.bos_token_id) if x is not None}
 with controller.use(vectors): out=model(**inp,use_cache=True)
 cache,next_token=out.past_key_values,torch.argmax(out.logits[:,-1,:],dim=-1,keepdim=True); ids=[]
 while len(ids)<n:
  t=int(next_token.item())
  if t in stop:break
  ids.append(t)
  with controller.use(vectors):out=model(input_ids=next_token.to(next(model.parameters()).device),past_key_values=cache,use_cache=True)
  cache,next_token=out.past_key_values,torch.argmax(out.logits[:,-1,:],dim=-1,keepdim=True)
 return tok.decode(ids,skip_special_tokens=True).strip().strip('`').strip()
def query_cues(meta,device,batch=32):
 try:tok=AutoTokenizer.from_pretrained(MODERNBERT,local_files_only=True)
 except ValueError:tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(MODERNBERT)/'tokenizer.json'),unk_token='[UNK]',sep_token='[SEP]',pad_token='[PAD]',cls_token='[CLS]',mask_token='[MASK]')
 model=AutoModel.from_pretrained(MODERNBERT,local_files_only=True).to(device).eval();out=[]
 with torch.inference_mode():
  for st in range(0,len(meta),batch):
   e=tok([x['evaluation_query'] for x in meta[st:st+batch]],padding=True,truncation=True,max_length=128,return_tensors='pt');e={k:v.to(device) for k,v in e.items() if k in ('input_ids','attention_mask')};h=model(**e,output_hidden_states=True,return_dict=True).hidden_states[22];m=e['attention_mask'].to(h.dtype).unsqueeze(-1);out.append(F.normalize((h*m).sum(1)/m.sum(1).clamp_min(1),p=2,dim=1).cpu())
 del model;torch.cuda.empty_cache();return torch.cat(out)
def score(rows,device,batch):
 tok,model,labels=load_classifier(device)
 for st in range(0,len(rows),batch):
  block=rows[st:st+batch];e=tok([x['generated_text'] for x in block],padding=True,truncation=True,max_length=512,return_tensors='pt');p=torch.sigmoid(model(**{k:v.to(device) for k,v in e.items()}).logits).cpu()
  for x,z in zip(block,p):
   j=labels.index(x['target_emotion']);top=int(z.argmax());x.update(modernbert_predicted_label=labels[top],target_emotion_score=float(z[j]),emotion_accuracy=int(labels[top]==x['target_emotion']))
 del model;torch.cuda.empty_cache();return rows
def main():
 a=args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 device=torch.device('cuda');bank=torch.load(BANK,map_location='cpu',weights_only=False);q=torch.load(QV,map_location='cpu',weights_only=True).float();meta=read_jsonl(QM)
 c=query_cues(meta,device);sf=q@bank['fact_vectors'].float().T;se=c@bank['emotion_cue_vectors'].float().T;scoremat={'fact_only':sf,'emotion_only':se,'concat_sum':sf+se,'similarity_product':sf*se}[a.method]; val,idx=scoremat.max(1)
 rows=[]
 for i,(m,j,s) in enumerate(zip(meta,idx.tolist(),val.tolist())):
  r=bank['records'][j];rows.append({'query_row_id':i,'memory_id':m['memory_id'],'target_emotion':m['target_emotion'],'evaluation_query':m['evaluation_query'],'retrieval_method':a.method,'retrieved_memory_id':r['memory_id'],'retrieved_emotion':r['emotion_label'],'retrieved_memory_source':r['memory_source'],'retrieved_factual_memory':r['factual_memory'],'retrieved_similarity':float(s),'fact_score':float(sf[i,j]),'emotion_score':float(se[i,j]),'memory_exact_top1':int(r['memory_id']==m['memory_id']),'retrieval_emotion_match':int(r['emotion_label']==m['target_emotion'])})
 a.output_dir.mkdir(parents=True,exist_ok=True);write_jsonl(a.output_dir/'retrieval.jsonl',rows);(a.output_dir/'config.json').write_text(json.dumps({'method':a.method,'candidate_bank':str(BANK),'n':len(rows),'prompt_fact':'Top-1 retrieved factual_memory','steering':'retrieved_emotion -> original bank configured_layer_vectors; no second alpha'},ensure_ascii=False,indent=2)+'\n')
 tok,model=load_qwen(device);values=bank['injection_values'];layers={l for x in values['values'].values() for l in x['layers']};controller=MultilayerSteering(model,layers)
 for condition in ('bare_query','retrieved_fact_prompt_plus_retrieved_emotion_steering'):
  path=a.output_dir/f'{condition}_generations.jsonl';done={x['memory_id'] for x in read_jsonl(path)}
  for n,row in enumerate(rows,1):
   if row['memory_id'] not in done:
    use=condition!='bare_query';emotion=row['retrieved_emotion'];item=values['values'][emotion] if use else None;vectors=configured_vectors(values,emotion) if use else None
    append_jsonl(path,{**row,'condition':condition,'prompt_includes_retrieved_fact':use,'injected_emotion':emotion if use else None,'injected_layers':item['layers'] if item else [],'layer_weights':item['layer_weights'] if item else {},'generated_text':generate(tok,model,controller,row['evaluation_query'],row['retrieved_factual_memory'] if use else None,vectors,a.max_new_tokens)})
   if n%10==0 or n==len(rows):print(json.dumps({'method':a.method,'condition':condition,'completed':n,'total':len(rows)},ensure_ascii=False),flush=True)
 controller.close();del model;torch.cuda.empty_cache();allrows=[x for cnd in ('bare_query','retrieved_fact_prompt_plus_retrieved_emotion_steering') for x in read_jsonl(a.output_dir/f'{cnd}_generations.jsonl')]
 scored=read_jsonl(a.output_dir/'scored_results.jsonl');
 if len(scored)!=len(allrows):scored=score(allrows,device,a.classifier_batch_size)
 base={x['memory_id']:x for x in scored if x['condition']=='bare_query'}
 for x in scored:x['emotion_gain_vs_bare_query']=x['target_emotion_score']-base[x['memory_id']]['target_emotion_score']
 write_jsonl(a.output_dir/'scored_results.jsonl',scored)
 report=[]
 for cnd in ('bare_query','retrieved_fact_prompt_plus_retrieved_emotion_steering'):
  g=[x for x in scored if x['condition']==cnd];report.append({'condition':cnd,'n':len(g),'emotion_accuracy':sum(x['emotion_accuracy'] for x in g)/len(g),'mean_target_emotion_score':sum(x['target_emotion_score'] for x in g)/len(g),'mean_emotion_gain_vs_bare_query':sum(x['emotion_gain_vs_bare_query'] for x in g)/len(g),'memory_exact_top1':sum(x['memory_exact_top1'] for x in g)/len(g),'retrieval_emotion_match':sum(x['retrieval_emotion_match'] for x in g)/len(g)})
 with (a.output_dir/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
 print(json.dumps({'status':'complete','method':a.method,'output':str(a.output_dir)},ensure_ascii=False))
if __name__=='__main__':main()
