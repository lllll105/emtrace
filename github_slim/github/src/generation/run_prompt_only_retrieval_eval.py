#!/usr/bin/env python3
"""Generate prompt-only retrieval baselines for all five retrieval methods."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/phase3_test_evaluation/retrieval_eval'))
from run_qwen_multilayer_11class_retrieval_eval import load_qwen, read_jsonl, token_inputs, append_jsonl

S1=ROOT/'use_2rag/build/stage1_fact_to_emotion_cue/stage1_fact_to_emotion_cue_bank.pt'
S2=ROOT/'use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt'
QV=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_fact_vectors_qwen3_fp16.pt'
QM=ROOT/'use_2rag/evaluation/test_query_qwen3_stage1_stage2/query_metadata.jsonl'
MODERNBERT='./local/.cache/huggingface/hub/models--cirimus--modernbert-base-go-emotions/snapshots/690341c8744d225dfd7a1fddae23f541b362b487'
METHODS=('fact_only','emotion_only','concat_sum','similarity_product','two_stage_fact_then_cue_prototype')

def parse():
 p=argparse.ArgumentParser();p.add_argument('--condition',choices=('fact_prompt_only','memory_retrieved_emotion_label','memory_oracle_emotion_label'),required=True);p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpu',default='0');p.add_argument('--max-new-tokens',type=int,default=96);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);return p.parse_args()
def cues(meta,d):
 try:tok=AutoTokenizer.from_pretrained(MODERNBERT,local_files_only=True)
 except ValueError:tok=PreTrainedTokenizerFast(tokenizer_file=str(Path(MODERNBERT)/'tokenizer.json'),unk_token='[UNK]',sep_token='[SEP]',pad_token='[PAD]',cls_token='[CLS]',mask_token='[MASK]')
 model=AutoModel.from_pretrained(MODERNBERT,local_files_only=True).to(d).eval();out=[]
 with torch.inference_mode():
  for i in range(0,len(meta),32):
   e=tok([x['evaluation_query'] for x in meta[i:i+32]],padding=True,truncation=True,max_length=128,return_tensors='pt');e={k:v.to(d) for k,v in e.items() if k in ('input_ids','attention_mask')};h=model(**e,output_hidden_states=True,return_dict=True).hidden_states[22];m=e['attention_mask'].to(h.dtype).unsqueeze(-1);out.append(F.normalize((h*m).sum(1)/m.sum(1).clamp_min(1),p=2,dim=1).cpu())
 del model;torch.cuda.empty_cache();return torch.cat(out)
def make_prompt(q,fact,emotion,with_emotion):
 extra=f'\n\nEmotional state:\n{emotion}' if with_emotion else ''
 return [{'role':'user','content':f'Retrieved factual memory (use it as context when relevant; do not mention this retrieval process):\n{fact}{extra}\n\nUser query:\n{q}\n\nGenerate a helpful response.'}]
@torch.inference_mode()
def generate(tok,model,q,fact,emotion,with_emotion,n):
 d=next(model.parameters()).device;out=model(**token_inputs(tok,make_prompt(q,fact,emotion,with_emotion),d),use_cache=True);cache=out.past_key_values;nt=out.logits[:,-1].argmax(-1,keepdim=True);stop={x for x in (tok.eos_token_id,tok.pad_token_id,tok.bos_token_id) if x is not None};ids=[]
 while len(ids)<n:
  if int(nt.item()) in stop:break
  ids.append(int(nt.item()));out=model(input_ids=nt.to(d),past_key_values=cache,use_cache=True);cache,nt=out.past_key_values,out.logits[:,-1].argmax(-1,keepdim=True)
 return tok.decode(ids,skip_special_tokens=True).strip()
def main():
 a=parse();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
 if not 0<=a.shard_index<a.shard_count:raise ValueError('invalid shard selection')
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 d=torch.device('cuda');s1=torch.load(S1,map_location='cpu',weights_only=False);s2=torch.load(S2,map_location='cpu',weights_only=False);q=torch.load(QV,map_location='cpu',weights_only=True).float();meta=read_jsonl(QM);c=cues(meta,d);sf=q@s1['fact_keys'].float().T;se=c@s1['emotion_cue_values'].float().T;labels=list(s2['manifest']['labels']);scores={'fact_only':sf,'emotion_only':se,'concat_sum':sf+se,'similarity_product':sf*se};first_fact=sf.argmax(1);p=(s1['emotion_cue_values'].float()[first_fact]@s2['emotion_prototype_keys'].float().T);cs=torch.full((len(meta),len(labels)),-float('inf'))
 for pi,ci in enumerate(s2['prototype_class_indices'].tolist()):cs[:,ci]=torch.maximum(cs[:,ci],p[:,pi])
 tok,model=load_qwen(d);a.output_dir.mkdir(parents=True,exist_ok=True);indices=list(range(a.shard_index,len(meta),a.shard_count))
 for method in a.methods:
  first=first_fact if method.startswith('two_stage') else scores[method].argmax(1);emotion=[labels[int(x)] for x in cs.argmax(1)] if method.startswith('two_stage') else [s1['records'][int(x)]['emotion_label'] for x in first]
  path=a.output_dir/f'{method}.jsonl';done={x['memory_id'] for x in read_jsonl(path)}
  for completed,i in enumerate(indices,1):
   j=int(first[i])
   r=s1['records'][j];prompt_emotion=meta[i]['target_emotion'] if a.condition=='memory_oracle_emotion_label' else emotion[i];row={**meta[i],'condition':a.condition,'retrieval_method':method,'retrieved_memory_id':r['memory_id'],'retrieved_factual_memory':r['factual_memory'],'retrieved_emotion':emotion[i],'prompt_emotion':prompt_emotion,'emotion_source':'ground_truth' if a.condition=='memory_oracle_emotion_label' else 'retrieved','memory_exact_top1':int(r['memory_id']==meta[i]['memory_id']),'retrieval_emotion_match':int(emotion[i]==meta[i]['target_emotion'])}
   if row['memory_id'] not in done:append_jsonl(path,{**row,'prompt_includes_retrieved_fact':True,'prompt_includes_retrieved_emotion_label':True,'generated_text':generate(tok,model,row['evaluation_query'],row['retrieved_factual_memory'],prompt_emotion,True,a.max_new_tokens)})
   if completed%10==0 or completed==len(indices):print(json.dumps({'condition':a.condition,'method':method,'completed':completed,'total':len(indices),'shard_index':a.shard_index,'shard_count':a.shard_count}),flush=True)
 del model;torch.cuda.empty_cache();(a.output_dir/'config.json').write_text(json.dumps({'condition':a.condition,'methods':a.methods,'candidate_count':8030,'retrieval_query':'evaluation_query encoded by Qwen3 and ModernBERT','prompt_emotion_source':'ground_truth' if a.condition=='memory_oracle_emotion_label' else ('retrieved' if a.condition=='memory_retrieved_emotion_label' else 'none'),'steering':False,'npti':False,'shard_index':a.shard_index,'shard_count':a.shard_count},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'complete','output':str(a.output_dir)}))
if __name__=='__main__':main()
