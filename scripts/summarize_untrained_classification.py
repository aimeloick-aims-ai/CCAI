"""Compute paired patient-bootstrap contrasts from the saved classification predictions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
LABELS=['ADH','DCIS','FEA','IC','N','PB','UDH']
def f1(y,pred,weights=None):
 return f1_score(y,pred,labels=range(7),average='macro',sample_weight=weights,zero_division=0)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--work-dir',default='results/bracs_source_patient');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--n-boot',type=int,default=2000);ap.add_argument('--bootstrap-seed',type=int,default=42);a=ap.parse_args()
 root=ROOT/a.work_dir;run=root/f'seed{a.seed}';out=run/'untrained_classification'
 manifest=json.loads((out/'manifest.json').read_text()); ckpt=run/'model.pt';digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
 if manifest['checkpoint_sha256']!=digest:raise ValueError('Saved predictions were produced by a different checkpoint.')
 test=pd.read_csv(root/'test_ids.csv');split_hash=hashlib.sha256((root/'test_ids.csv').read_bytes()).hexdigest()
 if manifest['split_hashes']['test']!=split_hash:raise ValueError('Saved predictions use different test IDs.')
 with np.load(out/'test_predictions.npz') as z:
  if not np.array_equal(z['sample_ids'],test.sample_id.to_numpy(dtype=str)):raise ValueError('Prediction order differs from test IDs.')
  y=z['y'];random=z['untrained_predictions'];trained=z['trained_prediction'];majority=z['majority_prediction']
 np.testing.assert_array_equal(y,test.label.map({l:i for i,l in enumerate(LABELS)}).to_numpy())
 groups,group_index=np.unique(test.patient_id,return_inverse=True);n=len(groups)
 rng=np.random.default_rng(a.bootstrap_seed);counts=rng.multinomial(n,np.full(n,1/n),size=a.n_boot);weights=counts[:,group_index]
 rng_scores=np.stack([[f1(y,p,w) for w in weights] for p in random])
 trained_scores=np.array([f1(y,trained,w) for w in weights]);majority_scores=np.array([f1(y,majority,w) for w in weights])
 rows=[]
 controls=[(f'untrained_{i+1}',random[i],rng_scores[i]) for i in range(len(random))]
 controls.append(('mean_of_five_untrained',None,rng_scores.mean(axis=0)))
 controls.append(('train_majority_constant',majority,majority_scores))
 for name,pred,control_scores in controls:
  point_control=float(np.mean([f1(y,p) for p in random])) if name=='mean_of_five_untrained' else f1(y,pred) if pred is not None else float(rng_scores.mean(axis=0).mean())
  point_trained=f1(y,trained);delta=point_trained-point_control
  delta_draw=trained_scores-control_scores
  lo,hi=np.quantile(delta_draw,[.025,.975])
  rows.append({'comparison':f'trained_minus_{name}','trained_macro_f1':point_trained,'control_macro_f1':point_control,'delta_macro_f1':delta,'paired_patient_bootstrap_95_low':lo,'paired_patient_bootstrap_95_high':hi,'n_patients':n,'n_boot':a.n_boot})
 result=pd.DataFrame(rows);result.to_csv(out/'paired_comparisons.csv',index=False)
 print(result.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__':main()
