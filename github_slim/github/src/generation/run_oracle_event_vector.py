#!/usr/bin/env python3
"""P1-1 Vector: Oracle Event + Normal Emotion, plus Full Oracle.

All normal-event rows retain each retrieval method's normal predicted emotion,
but replace prompt factual memory with the target test memory's factual_memory.
Full Oracle replaces both prompt factual memory and injected emotion with target.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/phase3_test_evaluation/retrieval_eval'))
from run_qwen_multilayer_11class_retrieval_eval import MultilayerSteering,append_jsonl,configured_vectors,load_qwen,read_jsonl,token_inputs,write_jsonl
S1=ROOT/'use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt';S2=ROOT/'use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt';QV=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt';QM=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl';MB='./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487'
METHODS=('fact_only','emotion_only','concat_sum','similarity_product','two_stage_fact_then_cue_prototype')
def args():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--max-new-tokens',type=int,default=96);return p.parse_args()
def cues(meta,d):
 try:t=AutoTokenizer.from_pretrained(MB,local_files_only=True)
 except ValueError:t=PreTrainedTokenizerFast(tokenizer_file=str(Path(MB)/'tokenizer.json'),unk_token='[UNK]',sep_token='[SEP]',pad_token='[PAD]',cls_token='[CLS]',mask_token='[MASK]')
 m=AutoModel.from_pretrained(MB,local_files_only=True).to(d).eval();o=[]
 with torch.inference_mode():
  for i in range(0,len(meta),32):
   x=t([z['evaluation_query'] for z in meta[i:i+32]],padding=True,truncation=True,max_length=128,return_tensors='pt');x={k:v.to(d) for k,v in x.items() if k in ('input_ids','attention_mask')};h=m(**x,output_hidden_states=True,return_dict=True).hidden_states[22];a=x['attention_mask'].to(h.dtype).unsqueeze(-1);o.append(F.normalize((h*a).sum(1)/a.sum(1).clamp_min(1),p=2,dim=1).cpu())
 del m;torch.cuda.empty_cache();return torch.cat(o)
def prompt(q,f):return [{'role':'user','content':f'Retrieved factual memory (use it as context when relevant; do not mention this retrieval process):\n{f}\n\nUser query:\n{q}\n\nGenerate a helpful response.'}]
@torch.inference_mode()
def gen(tok,m,ctrl,q,f,v,n):
 d=next(m.parameters()).device;x=token_inputs(tok,prompt(q,f),d);stop={z for z in (tok.eos_token_id,tok.pad_token_id,tok.bos_token_id) if z is not None}
 with ctrl.use(v):o=m(**x,use_cache=True)
 cache,nt=o.past_key_values,o.logits[:,-1].argmax(-1,keepdim=True);ids=[]
 while len(ids)<n:
  z=int(nt.item())
  if z in stop:break
  ids.append(z)
  with ctrl.use(v):o=m(input_ids=nt.to(d),past_key_values=cache,use_cache=True)
  cache,nt=o.past_key_values,o.logits[:,-1].argmax(-1,keepdim=True)
 return tok.decode(ids,skip_special_tokens=True).strip().strip('`').strip()
def main():
 a=args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 d=torch.device('cuda');s1=torch.load(S1,map_location='cpu',weights_only=False);s2=torch.load(S2,map_location='cpu',weights_only=False);q=torch.load(QV,map_location='cpu',weights_only=True).float();meta=read_jsonl(QM);c=cues(meta,d);labels=list(s2['manifest']['labels']);sf=q@s1['fact_keys'].float().T;se=c@s1['emotion_cue_values'].float().T
 first={'fact_only':sf.argmax(1),'emotion_only':se.argmax(1),'concat_sum':(sf+se).argmax(1),'similarity_product':(sf*se).argmax(1)};p=s1['emotion_cue_values'].float()[sf.argmax(1)]@s2['emotion_prototype_keys'].float().T;cs=torch.full((len(meta),len(labels)),float('-inf'))
 for pi,ci in enumerate(s2['prototype_class_indices'].tolist()):cs[:,ci]=torch.maximum(cs[:,ci],p[:,pi])
 first['two_stage_fact_then_cue_prototype']=sf.argmax(1);pred={m:[s1['records'][int(j)]['emotion_label'] for j in first[m].tolist()] for m in METHODS};pred['two_stage_fact_then_cue_prototype']=[labels[int(x)] for x in cs.argmax(1).tolist()]
 target={r['memory_id']:r for r in s1['records'] if r['memory_source']=='test_memory_key_only'}
 if len(target)!=1103:raise ValueError('expected 1,103 target test records')
 a.output_dir.mkdir(parents=True,exist_ok=True);tok,model=load_qwen(d);values={'values':s2['configured_injection_values']};layers={l for x in values['values'].values() for l in x['layers']};ctrl=MultilayerSteering(model,layers)
 def run(name,method,oracle_emotion):
  out=a.output_dir/name/(method if method else 'full_oracle');out.mkdir(parents=True,exist_ok=True);path=out/'generations.jsonl';done={x['memory_id'] for x in read_jsonl(path)}
  for i,m in enumerate(meta,1):
   r=s1['records'][int(first[method][i-1])] if method else target[m['memory_id']];normal=pred[method][i-1] if method else m['target_emotion'];emotion=m['target_emotion'] if oracle_emotion else normal;t=target[m['memory_id']];row={**m,'p1_1_condition':name,'retrieval_method':method or 'full_oracle','event_source':'oracle_target_memory','emotion_source':'oracle' if oracle_emotion else 'normal_retrieval','retrieved_memory_id':r['memory_id'],'retrieved_emotion':normal,'injected_emotion':emotion,'prompt_factual_memory':t['factual_memory'],'prompt_memory_id':t['memory_id'],'memory_exact_top1':int(r['memory_id']==m['memory_id']),'retrieval_emotion_match':int(normal==m['target_emotion'])}
   if m['memory_id'] not in done:append_jsonl(path,{**row,'prompt_includes_retrieved_fact':True,'generated_text':gen(tok,model,ctrl,m['evaluation_query'],t['factual_memory'],configured_vectors(values,emotion),a.max_new_tokens)})
   if i%10==0 or i==len(meta):print(json.dumps({'condition':name,'method':method or 'full_oracle','completed':i,'total':len(meta)}),flush=True)
 for method in METHODS:run('oracle_event_normal_emotion',method,False)
 run('full_oracle',None,True)
 ctrl.close();del model;torch.cuda.empty_cache();(a.output_dir/'config.json').write_text(json.dumps({'protocol':'P1-1 Vector','conditions':['oracle_event_normal_emotion','full_oracle'],'normal_event':'target factual_memory + method-specific normal retrieved emotion','full_oracle':'target factual_memory + target emotion','methods':list(METHODS),'queries':1103,'max_new_tokens':a.max_new_tokens},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'complete','output':str(a.output_dir)}))
if __name__=='__main__':main()
