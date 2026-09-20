"""Offline evidence check for Tables 5, 8, 10 and 11."""
from pathlib import Path
import json,hashlib,contextlib,io
import pandas as pd,numpy as np
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
parser = argparse.ArgumentParser(description="Recompute reconstructed table reductions from original saved outputs; no inference.")
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists(): parser.error("Choose a fresh output file")
from llmbias.evaluation.table_assembly import legal,education,recruitment
from llmbias.evaluation.recruitment.workflow import evaluate_recruitment
old=args.source;models=['claude4','gpt41','gpt41mini','o4mini','deepseek_v3','llama1b','llama3b','llama8b'];prefixes=['claude','gpt-4.1-2025-04-14','gpt-4.1-mini-2025-04-14','o4-mini-2025-04-16','deepseek_chat','Llama-3.2-1B-Instruct-v2','Llama-3.2-3B-Instruct-v2','Meta-Llama-3.1-8B-Instruct-v2'];inputs=[]
def read(p):
 inputs.append({'path':p.relative_to(old).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return pd.read_csv(p)
# Legal exact saved grouped outputs, no inference.
legalprefix=['claude','gpt-4.1','gpt-4.1-mini','o4-mini','deepseek','llama-1b','llama-3b','llama-8b'];frames=[]
for model,prefix in zip(models,legalprefix):
 f=read(old/f'experiments/legal_data/evaluation/{prefix}_fairness_table.csv');f=f.rename(columns={'Group':'group','mF1':'mf1'});f['model']=model;f['attribute']=f['group'].map({'E.C. European':'state','The Rest':'state','Male':'gender','Female':'gender','≤ 35 years':'age','≤ 65 years':'age','> 65 years':'age'});frames.append(f)
lt=legal(pd.concat(frames,ignore_index=True))
# Compare with unrounded group scores recomputed from the saved tensors.
import torch
from sklearn.metrics import f1_score
exact=[]
for model,prefix in zip(models,legalprefix):
    arrays={}
    for kind in ['y_true','y_pred','metadata']:
        path=old/f'experiments/legal_data/evaluation/{prefix}_{kind}.pt'
        inputs.append({'path':path.relative_to(old).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        arrays[kind]=torch.load(path,map_location='cpu',weights_only=True).numpy()
    for attr,col,codes in [('state',0,[0,1]),('gender',2,[1,2]),('age',1,[1,2,3])]:
        for code in codes:
            mask=arrays['metadata'][:,col]==code
            if mask.any():exact.append({'model':model,'attribute':attr,'group':str(code),'mf1':100*f1_score(arrays['y_true'][mask],arrays['y_pred'][mask],average='macro')})
exact_table=legal(pd.DataFrame(exact))
expected_legal=[
 [58.6,9,49.6,68.4,1.5,67,61.5,7.7,51.1],
 [60.2,6.5,53.8,67.3,2.4,64.9,63.4,7.9,52.9],
 [46,6.2,39.8,51.8,0,51.7,42.5,8.7,32.1],
 [52.6,10.6,42,63.8,1.7,62.1,58.9,8.5,47.3],
 [53.8,5.6,48.1,59.3,.3,59,52.9,8,41.7],
 [12.3,2.5,9.8,14.8,.5,14.3,15.1,2,13],
 [14.2,1.8,12.3,15.2,1,14.2,15,1.2,13.3],
 [24.8,2.5,22.4,26,.2,25.8,24.2,1.3,22.7]]
legal_comparisons=[]
for i,model in enumerate(models):
    for j,attr in enumerate(['state','gender','age']):
        row=exact_table[(exact_table.model==model)&(exact_table.attribute==attr)].iloc[0]
        for k,metric in enumerate(['mean_mf1','gd','worst_mf1']):
            expected=expected_legal[i][3*j+k]
            legal_comparisons.append({'model':model,'attribute':attr,'metric':metric,'computed':float(row[metric]),'paper':expected,'matches':bool(abs(row[metric]-expected)<=.05+1e-9)})

# Education cells derived from saved parsed answers, same A-E mapping as scorer.
cells=[]
for model,prefix in zip(models,prefixes):
 f=read(old/f'experiments/education_data/output_files/{prefix}_education_ranking.csv');f['mean_score']=f.answer.str.strip().str.upper().map(dict(zip('ABCDE',range(1,6))));f=f.dropna(subset=['mean_score']);f=f.groupby(['role','gender','ethnicity'],as_index=False).mean_score.mean();f['model']=model;cells.append(f)
et=education(pd.concat(cells,ignore_index=True));student=[[.17,.35,.82,2.30],[0,0,.90,2.02],[.83,1.66,.50,1.26],[.36,.71,.73,2.04],[.60,1.20,.71,1.75],[.35,.71,.81,2.21],[.61,1.21,.60,1.71],[.74,1.48,.61,1.41]];teacher=[[.43,.85,.68,1.90],[.57,1.14,.56,1.61],[.07,.14,.84,2.34],[.49,.97,.79,2.06],[.05,.10,.85,2.17],[.78,1.56,.56,1.50],[.18,.36,.48,1.42],[.64,1.28,.47,1.28]]
comparisons=[]
for i,model in enumerate(models):
 for role,vals in [('student',student[i]),('teacher',teacher[i])]:
  for j,dim in enumerate(['gender','ethnicity']):
   row=et[(et.model==model)&(et.role==role)&(et.dimension==dim)].iloc[0]
   for k,metric in enumerate(['MAB','MDB']):
    comparisons.append({'model':model,'role':role,'dimension':dim,'metric':metric,'computed':float(row[metric]),'paper':vals[2*j+k],'matches':abs(row[metric]-vals[2*j+k])<=.005+1e-9})
# Recruitment API models have saved CSVs; LLaMA parsed CSVs absent, do not infer a parser here.
rec=[]
for i,(model,prefix) in enumerate(zip(models[:5],prefixes[:5])):
 npth=old/f'experiments/admission_data/output_files/{prefix}_djinni_neutral.csv';spth=old/f'experiments/admission_data/output_files/{prefix}_djinni_v2.csv';n=read(npth);read(spth);n['model']=model
 with contextlib.redirect_stdout(io.StringIO()):g=evaluate_recruitment(npth,spth,model)
 for weighting in ['row','equal_group']:
  row=recruitment(n,g,weighting=weighting).iloc[0].to_dict();row['paper_neutral_admit_pct']=[26.3,36,34.6,19,51][i];row['paper_flip_pct']=[9.1,10.7,11.2,9.4,17.5][i];row['matches']=abs(row['neutral_admit_pct']-row['paper_neutral_admit_pct'])<=.05 and abs(row['flip_pct']-row['paper_flip_pct'])<=.05;rec.append(row)
rounded_comparisons=[]
for comparison in legal_comparisons:
 row=lt[(lt.model==comparison['model']) & (lt.attribute==comparison['attribute'])].iloc[0]
 value=float(row[comparison['metric']])
 rounded_comparisons.append({**comparison,'computed':value,'matches':abs(value-comparison['paper'])<=.05+1e-9})
report={'inputs':inputs,'legal':{'rows':lt.to_dict('records'),'rounded_group_comparisons':rounded_comparisons,'rounded_group_matches':sum(x['matches'] for x in rounded_comparisons),'tensor_based_comparisons':legal_comparisons,'tensor_matches':sum(x['matches'] for x in legal_comparisons),'total':len(legal_comparisons),'limitation':'Computed from already rounded saved group scores; final one-decimal agreement can be affected by double rounding. Group-label mappings retained as saved.'},'education':{'comparisons':comparisons,'matches':sum(x['matches'] for x in comparisons),'total':len(comparisons),'policy':'Equal six-cell standardization then marginal averaging; no metric changes to original scorer.'},'recruitment':{'comparisons':rec,'limitation':'Five models with saved neutral/v2 CSVs checked; three LLaMA parsed CSVs not located. Row and equal-group weighting compared explicitly.'}}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report,indent=2,default=lambda x:x.item())+'\n')
print('Education matches',report['education']['matches'],'/',len(comparisons));print('Recruitment',[(x['model'],x['weighting'],x['matches'],x['flip_pct']) for x in rec])

print('Legal tensor matches', sum(x['matches'] for x in legal_comparisons), '/',len(legal_comparisons))
