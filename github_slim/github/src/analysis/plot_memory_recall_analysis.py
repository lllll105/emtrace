#!/usr/bin/env python3
import argparse,csv
from pathlib import Path
from statistics import mean
def read(p):
 with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def f(x):
 try:return float(x)
 except:return None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--analysis-dir',type=Path,default=Path('use_2rag/plus/results'));a=ap.parse_args();import matplotlib.pyplot as plt
 t1=read(a.analysis_dir/'table1_retrieval_quality.csv');t2=read(a.analysis_dir/'table2_generation_by_recall_group.csv');t3=read(a.analysis_dir/'table3_similarity_analysis.csv');t4=read(a.analysis_dir/'table4_normal_oracle_pairs.csv')
 methods=[x['retrieval_method'] for x in t1];fig,ax=plt.subplots(figsize=(11,5));groups=['event_correct_affect_correct','event_correct_affect_wrong','event_wrong_affect_correct','event_wrong_affect_wrong'];w=.2
 for j,g in enumerate(groups):ax.bar([i+(j-1.5)*w for i in range(len(methods))],[int(next(x[g] for x in t1 if x['retrieval_method']==m)) for m in methods],w,label=g)
 ax.set_xticks(range(len(methods)));ax.set_xticklabels(methods,rotation=25,ha='right');ax.set_ylabel('Samples');ax.set_title('Event-Affect Binding: four retrieval states');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(a.analysis_dir/'figure1_recall_groups.png',dpi=180);plt.close(fig)
 rows=read(a.analysis_dir/'sample_level_analysis.csv');fig,ax=plt.subplots(figsize=(8,5));
 for m in methods:
  q=[x for x in rows if x['condition']=='C4_normal_vector' and x['retrieval_method']==m and f(x.get('fact_similarity')) is not None and f(x.get('target_emotion_score')) is not None];ax.scatter([f(x['fact_similarity']) for x in q],[f(x['target_emotion_score']) for x in q],s=5,alpha=.25,label=m)
 ax.set_xlabel('Fact similarity');ax.set_ylabel('Target emotion score');ax.set_title('Normal vector: similarity vs emotion score');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(a.analysis_dir/'figure2_similarity_scatter.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));
 for m in methods:
  q=[x for x in t3 if x['condition']=='C4_normal_vector' and x['emotion_retrieval']=='all' and x['retrieval_method']==m and x['similarity_group'] in ('high','middle','low')];ax.plot(['high','middle','low'],[f(x.get('emotion_score')) for x in q],marker='o',label=m)
 ax.set_ylabel('Mean target emotion score');ax.set_title('Normal vector: fact similarity thirds');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(a.analysis_dir/'figure3_similarity_trend.png',dpi=180);plt.close(fig)
 vals=[]
 for m in methods:
  q=[f(x.get('delta_emotion_score')) for x in t4 if x['contrast']=='vector' and x['retrieval_method']==m and x['emotion_retrieval_correct']=='0' and f(x.get('delta_emotion_score')) is not None];vals.append(mean(q) if q else 0)
 fig,ax=plt.subplots(figsize=(8,5));ax.bar(methods,vals);ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Oracle - Normal target score');ax.set_title('Vector steering: incorrectly retrieved emotion');ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(a.analysis_dir/'figure4_normal_oracle_delta.png',dpi=180);plt.close(fig)
 print('plots written to',a.analysis_dir)
if __name__=='__main__':main()
