"""Graph-level CAV/TCAV on a frozen GINE, with patient-block randomized controls.

All targets are graph means and all CAVs live in the pooled graph embedding.
TRAIN q20/q80 and fixed ridge alpha define the protocol; validation/test never fit CAVs.
Control-reference p-values are exploratory and require patient-block sign exchangeability.
Reference: https://proceedings.mlr.press/v80/kim18d.html
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from concept_audit_v3.integrity import validate_splits
from concept_decodability_audit.config import NODE_CONCEPTS
from concept_decodability_audit.utils import encode_batch, graph_mean_edge_distance, set_seed
from train_gine_baseline import GINEBaseline, GraphBRACSDataset, LABELS

CONCEPTS = list(NODE_CONCEPTS) + ["spatial_distance"]


def fit_directions(features, targets, patients, n_controls=999, seed=42, alpha=1.):
    """Weighted RidgeClassifier CAV; null labels flipped together within each patient."""
    low, high = np.quantile(targets, [.2,.8])
    if not low < high:
        raise ValueError("TRAIN q20 and q80 coincide; concept cannot be separated")
    keep = (targets <= low) | (targets >= high)
    x = np.asarray(features[keep],float)
    y = np.where(targets[keep] >= high,1.,-1.)
    if min(np.sum(y==1),np.sum(y==-1)) < 2:
        raise ValueError("Too few high/low TRAIN examples")
    groups,inv = np.unique(np.asarray(patients)[keep],return_inverse=True)
    weights = 1 / np.bincount(inv)[inv]
    weights *= len(weights)/weights.sum()
    center = np.average(x,axis=0,weights=weights)
    scale = np.sqrt(np.average((x-center)**2,axis=0,weights=weights))
    scale[scale<1e-8] = 1.
    design = (x-center)/scale
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.,1.],size=(len(groups),n_controls))
    responses = np.column_stack([y,y[:,None]*signs[inv]])
    # X is weighted-centered, so intercept subtraction cancels in X.T W y.
    gram = design.T @ (weights[:,None]*design) + alpha*np.eye(design.shape[1])
    beta = np.linalg.solve(gram,design.T @ (weights[:,None]*responses))
    coefficients = beta/scale[:,None]
    intercept = np.average(responses,axis=0,weights=weights)-center@coefficients
    norms = np.linalg.norm(coefficients,axis=0)
    if np.any(norms < 1e-12):
        raise ValueError("Degenerate observed/control CAV; inspect concept and TRAIN groups")
    directions = coefficients/norms
    return directions,coefficients[:,0],float(intercept[0]),{
        "q20":float(low),"q80":float(high),"train_low":int(np.sum(y==-1)),
        "train_high":int(np.sum(y==1)),"train_patients":len(groups)}


def holm(values):
    values=np.asarray(values,float)
    result=np.full_like(values,np.nan)
    indices=np.flatnonzero(np.isfinite(values))
    order=indices[np.argsort(values[indices])]
    result[order]=np.minimum(1.,np.maximum.accumulate(values[order]*(len(order)-np.arange(len(order)))))
    return result


def patient_interval(values,patients,n_boot=2000,seed=42):
    """Graph-weighted mean, whole-patient bootstrap within the evaluated class subset."""
    groups,inv=np.unique(np.asarray(patients),return_inverse=True)
    if len(groups)<2:
        return float("nan"),float("nan")
    counts=np.bincount(inv)
    sums=np.bincount(inv,weights=values)
    mult=np.random.default_rng(seed).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=n_boot)
    bootstrap=(mult@sums)/(mult@counts)
    return tuple(np.quantile(bootstrap,[.025,.975]).tolist())


def extract(model,frame,scaler,device,batch_size):
    dataset=GraphBRACSDataset(frame,scaler['x_mean'],scaler['x_std'],float(scaler['edge_mean']),float(scaler['edge_std']))
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    embeddings,targets,gradients,labels=[],[],[],[]
    for index,batch in enumerate(loader):
        batch=batch.to(device)
        with torch.no_grad():
            _,z,_=encode_batch(model,batch)
            measurements=[]
            for column in NODE_CONCEPTS.values():
                raw=batch.x[:,column]*float(scaler['x_std'][column])+float(scaler['x_mean'][column])
                measurements.append(global_mean_pool(raw[:,None],batch.batch)[:,0])
            measurements.append(graph_mean_edge_distance(batch,float(scaler['edge_mean']),float(scaler['edge_std'])))
            targets.append(torch.stack(measurements,dim=1).cpu().numpy())
        z=z.detach().requires_grad_(True)
        scores=model.classifier(z)
        gradient=torch.autograd.grad(scores.gather(1,batch.y[:,None]).sum(),z)[0]
        embeddings.append(z.detach().cpu().numpy())
        gradients.append(gradient.cpu().numpy())
        labels.append(batch.y.cpu().numpy())
        if index%25==0:
            print(f"  embedding/gradient batch {index+1}/{len(loader)}",flush=True)
    return {"embeddings":np.concatenate(embeddings),"targets":np.concatenate(targets),
            "gradients":np.concatenate(gradients),"y":np.concatenate(labels),
            "sample_ids":frame.sample_id.to_numpy(dtype=str)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work-dir',default='results/bracs_source_patient')
    ap.add_argument('--seed',type=int,default=42,help='Training checkpoint seed')
    ap.add_argument('--out-dir',default='')
    ap.add_argument('--n-controls',type=int,default=999)
    ap.add_argument('--n-boot',type=int,default=2000)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--max-graphs',type=int,default=0,help='Smoke test only; never a final result')
    args=ap.parse_args()
    if min(args.n_controls,args.n_boot,args.batch_size,args.threads)<1 or args.max_graphs<0:
        ap.error('Run sizes must be positive')
    torch.set_num_threads(args.threads)
    set_seed(42)
    work=Path(args.work_dir)
    checkpoint=work/f'seed{args.seed}/model.pt'
    output=Path(args.out_dir) if args.out_dir else work/f'seed{args.seed}/cav_tcav'
    if args.max_graphs and not args.out_dir:
        ap.error('Smoke tests require a separate --out-dir')
    frames={s:pd.read_csv(work/f'{s}_ids.csv',dtype={'patient_id':str}) for s in ['train','validation','test']}
    leakage=validate_splits(frames)
    if any('patient_id' not in f or f.patient_id.isna().any() for f in frames.values()):
        raise ValueError('Complete patient IDs required')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    if set(ckpt.get('scaler_train_ids',[]))!=set(frames['train'].sample_id):
        raise ValueError('Scaler training provenance missing or mismatched')
    for s,key in [('train','train_ids'),('validation','validation_ids'),('test','test_ids')]:
        if set(frames[s].sample_id)!=set(ckpt[key]):
            raise ValueError(f'Checkpoint {s} IDs do not match')
    identity={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'split_hashes':{s:hashlib.sha256((work/f'{s}_ids.csv').read_bytes()).hexdigest() for s in frames},
              'arguments':vars(args)}
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['identity']!=identity:
            raise ValueError('Existing CAV/TCAV run has different code/inputs/settings. Use a new --out-dir.')
        if old['status']=='completed':
            print(f'Already completed: {output}. Existing results reused; no GNN rerun.')
            return
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output directory without a matching manifest')
    output.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running','identity':identity,'leakage':leakage,'representation':'pooled_graph_embedding',
              'target_unit':'graph means for all five concepts','threshold_source':'TRAIN q20/q80',
              'ridge_alpha':1.,'cav_weighting':'equal total weight per TRAIN patient among extreme graphs',
              'control':'TRAIN patient-block binary-label sign flips; refit Ridge CAVs',
              'inference':'Exploratory control-reference p-values; patient-block sign exchangeability is an assumption, not established biological validity.',
              'multiple_comparisons':'Holm over 35 concept/class cells separately within each split',
              'smoke_test':bool(args.max_graphs),'n_controls':args.n_controls,'min_control_p':1/(args.n_controls+1)}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    model=GINEBaseline(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state_dict']);model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    arrays={}
    for split,frame in frames.items():
        if args.max_graphs:frame=frame.iloc[:args.max_graphs].copy()
        frames[split]=frame
        cache=output/f'{split}_representations.npz'
        if cache.exists():
            with np.load(cache) as data:arrays[split]={key:data[key] for key in data.files}
            np.testing.assert_array_equal(arrays[split]['sample_ids'],frame.sample_id.to_numpy(dtype=str))
            print(f'Reusing cached representations: {split}',flush=True)
        else:
            print(f'Extracting {split} representations and true-class gradients',flush=True)
            arrays[split]=extract(model,frame,ckpt['scaler'],device,args.batch_size)
            np.savez_compressed(cache,**arrays[split])
    cav_rows,tcav_rows,direction_arrays=[],[],{}
    for ci,concept in enumerate(CONCEPTS):
        print(f'Fitting CAV and {args.n_controls} patient-block controls: {concept}',flush=True)
        directions,coef,intercept,fit=fit_directions(arrays['train']['embeddings'],arrays['train']['targets'][:,ci],
            frames['train'].patient_id.to_numpy(),args.n_controls,seed=42+ci)
        direction_arrays[concept]=directions
        for split in ['validation','test']:
            data=arrays[split]
            target=data['targets'][:,ci]
            keep=(target<=fit['q20'])|(target>=fit['q80'])
            binary=(target[keep]>=fit['q80']).astype(int)
            prediction=data['embeddings'][keep]@coef+intercept
            auc=roc_auc_score(binary,prediction) if len(np.unique(binary))==2 else float('nan')
            cav_rows.append({'concept':concept,'split':split,'auc':auc,'n_extreme_graphs':int(keep.sum()),
                             'n_low':int(np.sum(binary==0)),'n_high':int(np.sum(binary==1)),**fit})
            sensitivity=data['gradients']@directions
            positive=sensitivity>0
            direction_arrays[f'{concept}__{split}__sensitivity']=sensitivity
            for k,label in enumerate(LABELS):
                select=data['y']==k
                patients=frames[split].patient_id.to_numpy()[select]
                row={'concept':concept,'split':split,'class':label,'n_graphs':int(select.sum()),
                     'n_patients':len(np.unique(patients))}
                if not select.any():
                    row.update(tcav=np.nan,control_reference_p=np.nan,status='class_absent')
                else:
                    scores=positive[select].mean(axis=0)
                    observed=float(scores[0]);controls=scores[1:]
                    per_graph_difference=positive[select,0].astype(float)-positive[select,1:].mean(axis=1)
                    low,high=patient_interval(positive[select,0].astype(float),patients,args.n_boot)
                    dlo,dhi=patient_interval(per_graph_difference,patients,args.n_boot)
                    p=(1+np.sum(np.abs(controls-.5)>=abs(observed-.5)-1e-12))/(args.n_controls+1)
                    row.update(tcav=observed,tcav_ci95_low=low,tcav_ci95_high=high,
                               control_tcav_mean=float(controls.mean()),tcav_minus_controls=float(per_graph_difference.mean()),
                               delta_ci95_low=dlo,delta_ci95_high=dhi,control_reference_p=float(p),
                               status='exploratory' if row['n_patients']>=2 else 'insufficient_patients_for_ci')
                tcav_rows.append(row)
    cav=pd.DataFrame(cav_rows);tcav=pd.DataFrame(tcav_rows)
    for split in ['validation','test']:
        mask=tcav.split==split
        tcav.loc[mask,'control_reference_p_holm']=holm(tcav.loc[mask,'control_reference_p'])
    cav.to_csv(output/'cav_scores.csv',index=False)
    tcav.to_csv(output/'tcav_scores.csv',index=False)
    np.savez_compressed(output/'directions_and_sensitivities.npz',**direction_arrays)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,split in zip(axes,['validation','test']):
        table=tcav.query('split==@split').pivot(index='concept',columns='class',values='tcav_minus_controls').reindex(index=CONCEPTS,columns=LABELS)
        image=ax.imshow(table,aspect='auto',cmap='coolwarm',vmin=-.5,vmax=.5)
        ax.set(xticks=range(7),xticklabels=LABELS,yticks=range(5),yticklabels=CONCEPTS,title=split.capitalize())
        for i in range(5):
            for j in range(7):
                if np.isfinite(table.iloc[i,j]):ax.text(j,i,f'{table.iloc[i,j]:+.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(image,ax=axes,label='TCAV minus mean randomized control')
    fig.savefig(output/'tcav.png',dpi=150);fig.savefig(output/'tcav.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
    cav.pivot(index='concept',columns='split',values='auc').reindex(CONCEPTS).plot.bar(ax=ax)
    ax.set(ylabel='CAV high/low separation AUC',ylim=(0,1.05),xlabel='Graph-mean target')
    ax.tick_params(axis='x',rotation=15)
    fig.savefig(output/'cav_auc.png',dpi=150);fig.savefig(output/'cav_auc.pdf');plt.close(fig)
    manifest['status']='completed'
    manifest['sample_ids']={s:frame.sample_id.tolist() for s,frame in frames.items()}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(cav.to_string(index=False,float_format=lambda value:f'{value:.5g}'),flush=True)
    print(f'Done: {output}. Control-reference p-values remain exploratory; see manifest assumptions.',flush=True)


if __name__=='__main__':
    main()
