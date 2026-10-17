"""Evaluate random-initialized GINE classifiers on the frozen patient-disjoint split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_audit_v3 import make_loader
from concept_audit_v3.baselines import make_untrained_model
from train_gine_baseline import GINEBaseline

LABELS=['ADH','DCIS','FEA','IC','N','PB','UDH']

def predict(model, loader, device):
    ys=[];pred=[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(device)
            pred.extend(model(batch.x,batch.edge_index,batch.edge_attr,batch.batch).argmax(1).cpu().tolist())
            ys.extend(batch.y.cpu().tolist())
    return np.asarray(ys),np.asarray(pred)

def patient_bootstrap(y,pred,patients,n_boot,seed):
    unique,inverse=np.unique(patients,return_inverse=True)
    rng=np.random.default_rng(seed)
    rows=[]
    for draw in rng.integers(0,len(unique),size=(n_boot,len(unique))):
        multiplicity=np.bincount(draw,minlength=len(unique))
        weights=multiplicity[inverse]
        rows.append(f1_score(y,pred,labels=range(7),average='macro',sample_weight=weights,zero_division=0))
    return np.quantile(rows,[.025,.975]).tolist()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir',default='results/bracs_source_patient')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--n-models',type=int,default=5)
    ap.add_argument('--n-boot',type=int,default=2000)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    root=ROOT/args.work_dir
    run=root/f'seed{args.seed}'
    checkpoint=run/'model.pt'
    out=run/'untrained_classification'
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'Refusing to overwrite non-empty {out}; archive it or choose another output path.')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    frames={s:pd.read_csv(root/f'{s}_ids.csv') for s in ('train','validation','test')}
    for s,frame in frames.items():
        if s+'_ids' in ckpt and set(frame.sample_id)!=set(ckpt[s+'_ids']):
            raise ValueError(f'{s} sample IDs differ from the selected checkpoint')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for col in ('sample_id','slide_id','patient_id'):
            if set(frames[a][col])&set(frames[b][col]):raise ValueError(f'Shared {col}: {a}/{b}')
    expected_hashes=ckpt.get('split_hashes',{})
    hashes={s:hashlib.sha256((root/f'{s}_ids.csv').read_bytes()).hexdigest() for s in frames}
    if expected_hashes and any(hashes[s]!=expected_hashes[s] for s in frames):raise ValueError('Partition hashes differ from checkpoint provenance')
    scaler=ckpt['scaler']
    train_loader=make_loader(frames['train'],scaler,args.batch_size)
    test_loader=make_loader(frames['test'],scaler,args.batch_size)
    y_true=None
    rows=[];predictions=[]
    for i in range(args.n_models):
        model=make_untrained_model(ckpt['model_config'],int(ckpt['seed'])+i,device,calibration_loader=train_loader)
        y,pred=predict(model,test_loader,device)
        if y_true is None:y_true=y
        if not np.array_equal(y,y_true):raise ValueError('Test label order changed between models')
        patients=frames['test'].patient_id.to_numpy()
        interval=patient_bootstrap(y,pred,patients,args.n_boot,seed=8400+i)
        rows.append({'model':f'untrained_{i+1}','initialization_seed':int(ckpt['seed'])+i,
                     'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,labels=range(7),average='macro',zero_division=0),
                     'patient_bootstrap_95_low':interval[0],'patient_bootstrap_95_high':interval[1]})
        predictions.append(pred)
        print(f"Untrained {i+1}/{args.n_models}: accuracy={rows[-1]['accuracy']:.4f}, macro-F1={rows[-1]['macro_f1']:.4f}",flush=True)
        del model
    train_y=frames['train'].label.map({label:i for i,label in enumerate(LABELS)}).to_numpy()
    counts=np.bincount(train_y,minlength=7)
    majority=int(np.argmax(counts))
    majority_pred=np.full_like(y_true,majority)
    interval=patient_bootstrap(y_true,majority_pred,frames['test'].patient_id.to_numpy(),args.n_boot,seed=8499)
    rows.append({'model':'train_majority_constant','initialization_seed':None,'accuracy':accuracy_score(y_true,majority_pred),
                 'macro_f1':f1_score(y_true,majority_pred,labels=range(7),average='macro',zero_division=0),
                 'patient_bootstrap_95_low':interval[0],'patient_bootstrap_95_high':interval[1]})
    trained=torch.load(checkpoint,map_location=device,weights_only=False)
    model=GINEBaseline(**trained['model_config']).to(device);model.load_state_dict(trained['model_state_dict'])
    yt,trained_pred=predict(model,test_loader,device)
    if not np.array_equal(yt,y_true):raise ValueError('Trained and untrained labels differ')
    interval=patient_bootstrap(y_true,trained_pred,frames['test'].patient_id.to_numpy(),args.n_boot,seed=8420)
    rows.append({'model':'trained_seed42','initialization_seed':args.seed,'accuracy':accuracy_score(y_true,trained_pred),
                 'macro_f1':f1_score(y_true,trained_pred,labels=range(7),average='macro',zero_division=0),
                 'patient_bootstrap_95_low':interval[0],'patient_bootstrap_95_high':interval[1]})
    metrics=pd.DataFrame(rows)
    summary={'status':'completed','checkpoint_sha256':digest,'split_hashes':hashes,'n_test_graphs':len(y_true),
             'n_test_patients':int(frames['test'].patient_id.nunique()),'n_untrained_models':args.n_models,
             'initialization_seeds':[int(ckpt['seed'])+i for i in range(args.n_models)],'n_boot':args.n_boot,
             'bootstrap_unit':'patient','bootstrap_seed_rule':'8400 + initialization index; 8499 constant baseline; 8420 trained model',
             'bn_calibration':'First 30 ordered TRAIN batches, same procedure as untrained decodability controls; weights remain random and frozen.',
             'scope':'Direct argmax classification by random-initialized GINEs; no classifier retraining on frozen random representations.',
             'train_class_counts':dict(zip(LABELS,counts.tolist())),'majority_class':LABELS[majority],
             'metrics':metrics.to_dict(orient='records')}
    metrics.to_csv(out/'classification_metrics.csv',index=False)
    np.savez_compressed(out/'test_predictions.npz',sample_ids=frames['test'].sample_id.to_numpy(dtype=str),y=y_true,
                        untrained_predictions=np.stack(predictions),trained_prediction=trained_pred,majority_prediction=majority_pred)
    (out/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(metrics.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('Saved:',out.relative_to(ROOT))

if __name__=='__main__':main()
