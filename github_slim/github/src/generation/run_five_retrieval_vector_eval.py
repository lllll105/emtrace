#!/usr/bin/env python3
"""Five retrieval methods with retrieved-fact prompting and vector steering."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/phase3_test_evaluation/retrieval_eval'))
from run_qwen_multilayer_11class_retrieval_eval import MultilayerSteering, configured_vectors, load_qwen, read_jsonl, token_inputs, append_jsonl

S1=ROOT/'use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt'
S2=ROOT/'use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt'
QV=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt'
QM=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl'
MODERNBERT='./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487'
METHODS=('fact_only','emotion_only','concat_sum','similarity_product','two_stage_fact_then_cue_prototype')

def parse():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--emotion-source',choices=('normal','oracle'),required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--max-new-tokens',type=int,default=96);return p.parse_args()
def cue_vectors(meta,device):
 try: tok=AutoTokenizer.from_pretrained(MODERNBERT,local_files_only=True)
 except ValueError: tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(MODERNBERT)/'tokenizer.json'),unk_token='[UNK]',sep_token='[SEP]',pad_token='[PAD]',cls_token='[CLS]',mask_token='[MASK]')
 model=AutoModel.from_pretrained(MODERNBERT,local_files_only=True).to(device).eval();out=[]
 with torch.inference_mode():
  for i in range(0,len(meta),32):
   e=tok([x['evaluation_query'] for x in meta[i:i+32]],padding=True,truncation=True,max_length=128,return_tensors='pt');e={k:v.to(device) for k,v in e.items() if k in ('input_ids','attention_mask')};h=model(**e,output_hidden_states=True,return_dict=True).hidden_states[22];m=e['attention_mask'].to(h.dtype).unsqueeze(-1);out.append(F.normalize((h*m).sum(1)/m.sum(1).clamp_min(1),p=2,dim=1).cpu())
 del model;torch.cuda.empty_cache();return torch.cat(out)
def prompt(q,f): return [{'role':'user','content':f'Retrieved factual memory (use it as context when relevant; do not mention this retrieval process):\n{f}\n\nUser query:\n{q}\n\nGenerate a helpful response.'}]
@torch.inference_mode()
def generate(tok,model,ctrl,q,f,vectors,n):
 d=next(model.parameters()).device; inp=token_inputs(tok,prompt(q,f),d); stop={x for x in (tok.eos_token_id,tok.pad_token_id,tok.bos_token_id) if x is not None}
 with ctrl.use(vectors): o=model(**inp,use_cache=True)
 cache,nt=o.past_key_values,o.logits[:,-1].argmax(-1,keepdim=True);ids=[]
 while len(ids)<n:
  if int(nt.item()) in stop: break
  ids.append(int(nt.item()))
  with ctrl.use(vectors): o=model(input_ids=nt.to(d),past_key_values=cache,use_cache=True)
  cache,nt=o.past_key_values,o.logits[:,-1].argmax(-1,keepdim=True)
 return tok.decode(ids,skip_special_tokens=True).strip()
def main():
 a=parse();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
 d=torch.device('cuda');s1=torch.load(S1,map_location='cpu',weights_only=False);s2=torch.load(S2,map_location='cpu',weights_only=False);q=torch.load(QV,map_location='cpu',weights_only=True).float();meta=read_jsonl(QM);c=cue_vectors(meta,d);sf=q@s1['fact_keys'].float().T;se=c@s1['emotion_cue_values'].float().T
 scores={'fact_only':sf,'emotion_only':se,'concat_sum':sf+se,'similarity_product':sf*se};labels=list(s2['manifest']['labels'])
 first=sf.argmax(1);cue=s1['emotion_cue_values'].float()[first];ps=cue@s2['emotion_prototype_keys'].float().T;cs=torch.full((len(meta),len(labels)),-float('inf'))
 for pi,ci in enumerate(s2['prototype_class_indices'].tolist()):cs[:,ci]=torch.maximum(cs[:,ci],ps[:,pi])
 if a.method.startswith('two_stage'): pred=[labels[int(x)] for x in cs.argmax(1)]
 else: _,first=scores[a.method].max(1);pred=[s1['records'][int(x)]['emotion_label'] for x in first]
 values={'values':s2['configured_injection_values']};rows=[]
 for i,j in enumerate(first.tolist()):
  r=s1['records'][j];emotion=meta[i]['target_emotion'] if a.emotion_source=='oracle' else pred[i];rows.append({**meta[i],'retrieval_method':a.method,'emotion_source':a.emotion_source,'retrieved_memory_id':r['memory_id'],'retrieved_factual_memory':r['factual_memory'],'retrieved_emotion':pred[i],'injected_emotion':emotion,'retrieval_emotion_match':int(pred[i]==meta[i]['target_emotion']),'memory_exact_top1':int(r['memory_id']==meta[i]['memory_id']),'fact_score':float(sf[i,j]),'emotion_score':float(se[i,j])})
 a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'config.json').write_text(json.dumps({'method':a.method,'emotion_source':a.emotion_source,'candidate_count':8030,'prompt':'Top-1 retrieved factual_memory','injection':'configured multilayer vectors'},ensure_ascii=False,indent=2)+'\n');write=a.output_dir/'retrieval.jsonl';
 for x in rows: append_jsonl(write,x)
 tok,model=load_qwen(d);layers={l for x in values['values'].values() for l in x['layers']};ctrl=MultilayerSteering(model,layers);vec={x['injected_emotion']:configured_vectors(values,x['injected_emotion']) for x in rows}
 out=a.output_dir/'generations.jsonl';done={x['memory_id'] for x in read_jsonl(out)}
 for i,x in enumerate(rows,1):
  if x['memory_id'] not in done: append_jsonl(out,{**x,'generated_text':generate(tok,model,ctrl,x['evaluation_query'],x['retrieved_factual_memory'],vec[x['injected_emotion']],a.max_new_tokens)})
  if i%10==0 or i==len(rows): print(json.dumps({'completed':i,'total':len(rows)}),flush=True)
 ctrl.close();del model;torch.cuda.empty_cache();print(json.dumps({'status':'complete','output':str(a.output_dir),'rows':len(rows)}))
if __name__=='__main__': main()
