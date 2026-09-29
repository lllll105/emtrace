#!/usr/bin/env python3
"""Run NPTI with retrieved factual-memory prompt injection.

The baseline is the original bare-query generation. Each retrieval method then
adds its Top-1 factual_memory to the prompt and applies the corresponding NPTI
neuron set. No multilayer steering-value injection is used here.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast

ROOT=Path(__file__).resolve().parents[3]
NPTI_DIR=ROOT/'code/phase3_test_evaluation/npti'
sys.path.insert(0,str(NPTI_DIR))
from run_qwen_multilayer_11class_npti_eval import NPTIController, cue_tokenizer, load_qwen, load_state, read_jsonl, append_jsonl, write_jsonl # noqa:E402
STAGE1=ROOT/'use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt'
STAGE2=ROOT/'use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt'
QV=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt'
QM=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl'
NPTI_SET=ROOT/'use/npti_set/emotion_neuron_sets_11class.json'
MODERNBERT='./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487'
METHODS=('fact_only','emotion_only','concat_sum','similarity_product','two_stage_fact_then_m2_prototype')
LABELS=('admiration','amusement','curiosity','joy','confusion','sadness','disappointment','surprise','excitement','desire','embarrassment')
def parse():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);p.add_argument('--gamma',type=float,default=1.0);p.add_argument('--max-new-tokens',type=int,default=96);p.add_argument('--classifier-batch-size',type=int,default=64);p.add_argument('--emotion-source',choices=('normal','oracle'),default='normal');p.add_argument('--prompt-style',choices=('legacy','helpful'),default='legacy');p.add_argument('--baseline-eval-file',type=Path);p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));return p.parse_args()
def prompt(tok,q,fact=None,device=None,style='legacy'):
 suffix='Generate a helpful response.' if style=='helpful' else 'Generate a response.'
 text=f'User query:\n{q}\n\n{suffix}' if fact is None else f'Retrieved factual memory (use it as context when relevant; do not mention this retrieval process):\n{fact}\n\nUser query:\n{q}\n\n{suffix}'
 x=tok.apply_chat_template([{'role':'user','content':text}],add_generation_prompt=True,return_tensors='pt',return_dict=True)
 if isinstance(x,torch.Tensor):return {'input_ids':x.to(device)}
 return {k:v.to(device) for k,v in x.items() if torch.is_tensor(v)}
@torch.inference_mode()
def generate(tok,model,controller,q,fact,state,n,style='legacy'):
 d=next(model.parameters()).device
 with controller.use(state):out=model(**prompt(tok,q,fact,d,style),use_cache=True)
 cache,next_token=out.past_key_values,out.logits[:,-1].argmax(dim=-1,keepdim=True);ids=[];stop={x for x in (tok.eos_token_id,tok.pad_token_id,tok.bos_token_id) if x is not None}
 while len(ids)<n:
  t=int(next_token.item())
  if t in stop:break
  ids.append(t)
  with controller.use(state):out=model(input_ids=next_token,past_key_values=cache,use_cache=True)
  cache,next_token=out.past_key_values,out.logits[:,-1].argmax(dim=-1,keepdim=True)
 return tok.decode(ids,skip_special_tokens=True).strip()
def encode_cues(meta,device,batch=32):
 try:tok=AutoTokenizer.from_pretrained(MODERNBERT,local_files_only=True)
 except ValueError:tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(MODERNBERT)/'tokenizer.json'),unk_token='[UNK]',sep_token='[SEP]',pad_token='[PAD]',cls_token='[CLS]',mask_token='[MASK]')
 model=AutoModel.from_pretrained(MODERNBERT,local_files_only=True).to(device).eval();out=[]
 with torch.inference_mode():
  for st in range(0,len(meta),batch):
   e=tok([x['evaluation_query'] for x in meta[st:st+batch]],padding=True,truncation=True,max_length=128,return_tensors='pt');e={k:v.to(device) for k,v in e.items() if k in ('input_ids','attention_mask')};h=model(**e,output_hidden_states=True,return_dict=True).hidden_states[22];m=e['attention_mask'].to(h.dtype).unsqueeze(-1);out.append(F.normalize((h*m).sum(1)/m.sum(1).clamp_min(1),p=2,dim=1).cpu())
 del model;torch.cuda.empty_cache();return torch.cat(out)
def retrieval(meta,device,shard,shards):
 s1=torch.load(STAGE1,map_location='cpu',weights_only=False);s2=torch.load(STAGE2,map_location='cpu',weights_only=False);q=torch.load(QV,map_location='cpu',weights_only=True).float();c=encode_cues(meta,device);fact=s1['fact_keys'].float();cue=s1['emotion_cue_values'].float();sf=q@fact.T;se=c@cue.T
 # Candidate-level methods.
 scores={'fact_only':sf,'emotion_only':se,'concat_sum':sf+se,'similarity_product':sf*se}
 # Two-stage class ranking, with the first-stage row retained for factual prompt.
 pscore=cue[sf.argmax(1)]@s2['emotion_prototype_keys'].float().T;cs=torch.full((len(meta),len(LABELS)),float('-inf'))
 for pi,ci in enumerate(s2['prototype_class_indices'].tolist()):cs[:,ci]=torch.maximum(cs[:,ci],pscore[:,pi])
 scores['two_stage_fact_then_m2_prototype']=cs
 out={}
 for method,score in scores.items():
  if method.startswith('two_stage'):
   _,cls=score.max(1);first=sf.argmax(1);pred=[LABELS[int(x)] for x in cls]
  else:
   _,first=score.max(1);pred=[s1['records'][int(x)]['emotion_label'] for x in first]
  rows=[]
  for i,j in enumerate(first.tolist()):
   r=s1['records'][j]; rows.append({'query_row_id':i,'memory_id':meta[i]['memory_id'],'target_emotion':meta[i]['target_emotion'],'evaluation_query':meta[i]['evaluation_query'],'retrieval_method':method,'retrieved_memory_id':r['memory_id'],'retrieved_factual_memory':r['factual_memory'],'retrieved_emotion':pred[i],'retrieved_memory_source':r['memory_source'],'retrieved_emotion_from_memory':r['emotion_label'],'retrieval_emotion_match':int(r['emotion_label']==meta[i]['target_emotion']),'stage2_emotion_match':int(pred[i]==meta[i]['target_emotion'])})
  out[method]=rows
 return {k:v[shard::shards] for k,v in out.items()},s1,s2
def score(rows,device,batch):
 tok=cue_tokenizer();model=AutoModelForSequenceClassification.from_pretrained(MODERNBERT,local_files_only=True).to(device).eval();labels=[str(model.config.id2label.get(i,f'label_{i}')).lower() for i in range(model.config.num_labels)]
 for st in range(0,len(rows),batch):
  b=rows[st:st+batch];e=tok([x['generated_text'] for x in b],padding=True,truncation=True,max_length=512,return_tensors='pt');p=torch.sigmoid(model(**{k:v.to(device) for k,v in e.items()}).logits).cpu()
  for x,z in zip(b,p):
   ti=labels.index(x['target_emotion']);x.update(modernbert_predicted_label=labels[int(z.argmax())],target_emotion_score=float(z[ti]),emotion_accuracy=int(labels[int(z.argmax())]==x['target_emotion']))
 del model;torch.cuda.empty_cache();return rows
def main():
 a=parse();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 device=torch.device('cuda');meta=read_jsonl(QM);sets=json.loads(NPTI_SET.read_text(encoding='utf-8'))
 retrieval_rows,s1,s2=retrieval(meta,device,a.shard_index,a.shard_count);run=a.output_dir/f'shard_{a.shard_index:02d}_of_{a.shard_count:02d}';run.mkdir(parents=True,exist_ok=True)
 retrieval_rows={m:retrieval_rows[m] for m in a.methods}
 config={'npti_set':str(NPTI_SET),'methods':list(a.methods),'candidate_count':8030,'test_queries':1103,'emotion_source':a.emotion_source,'prompt':'Top-1 retrieved factual_memory injected as context','prompt_style':a.prompt_style,'baseline_eval_file':str(a.baseline_eval_file) if a.baseline_eval_file else None,'npti':'retrieved or oracle emotion neuron set, all 28 layers, gamma='+str(a.gamma),'shard_index':a.shard_index,'shard_count':a.shard_count};(run/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
 for m,rows in retrieval_rows.items():write_jsonl(run/m/'retrieval.jsonl',rows)
 tok,model=load_qwen(device);controller=NPTIController(model);sets=json.loads(NPTI_SET.read_text(encoding='utf-8'))
 canonical=next(iter(retrieval_rows.values()));basepath=run/'baseline_generations.jsonl';base={x['memory_id']:x for x in read_jsonl(basepath)}
 if a.baseline_eval_file:
  existing={x['memory_id']:x for x in read_jsonl(a.baseline_eval_file) if x.get('condition')=='bare_query'}
  if set(existing)!={x['memory_id'] for x in meta}:raise ValueError('canonical bare-query baseline is incomplete')
  for x in canonical:
   source=existing[x['memory_id']]
   if source['evaluation_query']!=x['evaluation_query']:raise ValueError('baseline query does not match')
   if x['memory_id'] in base and base[x['memory_id']]['generated_text']!=source['generated_text']:raise ValueError('existing baseline differs from canonical baseline')
   if x['memory_id'] not in base:
    row={**x,'condition':'baseline','generated_text':source['generated_text'],'baseline_source':str(a.baseline_eval_file)}
    append_jsonl(basepath,row);base[x['memory_id']]=row
 for i,x in enumerate(canonical,1):
  if x['memory_id'] not in base:
   generated=generate(tok,model,controller,x['evaluation_query'],None,None,a.max_new_tokens,a.prompt_style)
   row={**x,'condition':'baseline','generated_text':generated};append_jsonl(basepath,row);base[x['memory_id']]=row
  if i%10==0 or i==len(canonical):print(json.dumps({'condition':'baseline','completed':i,'total':len(canonical)}),flush=True)
 # Greedy decoding with identical query, factual prompt and NPTI state is
 # deterministic. Reuse the actual saved text across retrieval methods, while
 # retaining one row per sample and method in each output JSONL.
 cached={}
 for method in a.methods:
  existing_path=run/method/'retrieved_fact_prompt_npti_generations.jsonl'
  for previous in read_jsonl(existing_path):
   key=(previous['memory_id'],previous['retrieved_factual_memory'],previous['npti_emotion'])
   if key in cached and cached[key]!=previous['generated_text']:
    raise ValueError(f'non-deterministic repeated NPTI generation: {key}')
   cached[key]=previous['generated_text']
 for m,rows in retrieval_rows.items():
  path=run/m/'retrieved_fact_prompt_npti_generations.jsonl';done={x['memory_id'] for x in read_jsonl(path)}
  for i,x in enumerate(rows,1):
   if x['memory_id'] not in done:
    emotion=x['target_emotion'] if a.emotion_source=='oracle' else x['retrieved_emotion']
    key=(x['memory_id'],x['retrieved_factual_memory'],emotion)
    if key not in cached:
     state=load_state(sets,emotion,a.gamma)
     cached[key]=generate(tok,model,controller,x['evaluation_query'],x['retrieved_factual_memory'],state,a.max_new_tokens,a.prompt_style)
    append_jsonl(path,{**x,'condition':'retrieved_fact_prompt_npti','prompt_includes_retrieved_fact':True,'npti_emotion':emotion,'emotion_source':a.emotion_source,'generated_text':cached[key]})
   if i%10==0 or i==len(rows):print(json.dumps({'method':m,'completed':i,'total':len(rows)}),flush=True)
 del model;torch.cuda.empty_cache();base_rows=score(list(base.values()),device,a.classifier_batch_size);write_jsonl(run/'baseline_scored.jsonl',base_rows);base={x['memory_id']:x for x in base_rows}
 for m in a.methods:
  p=run/m/'retrieved_fact_prompt_npti_generations.jsonl';rows=score(read_jsonl(p),device,a.classifier_batch_size);write_jsonl(run/m/'scored_results.jsonl',rows);g=[{'condition':'baseline','n':len(base_rows),'emotion_accuracy':sum(x['emotion_accuracy'] for x in base_rows)/len(base_rows),'target_emotion_score':sum(x['target_emotion_score'] for x in base_rows)/len(base_rows)},{'condition':'retrieved_fact_prompt_npti','n':len(rows),'emotion_accuracy':sum(x['emotion_accuracy'] for x in rows)/len(rows),'target_emotion_score':sum(x['target_emotion_score'] for x in rows)/len(rows),'emotion_gain':sum(x['target_emotion_score']-base[x['memory_id']]['target_emotion_score'] for x in rows)/len(rows),'retrieval_emotion_match':sum(x['retrieval_emotion_match'] for x in rows)/len(rows)}]
  fields=[]
  for row in g:
   for field in row:
    if field not in fields: fields.append(field)
  with (run/m/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(g)
 print(json.dumps({'status':'complete','run_dir':str(run),'queries':len(canonical)},ensure_ascii=False))
if __name__=='__main__':main()
