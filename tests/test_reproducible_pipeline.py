"""Check group partitioning and train-only scaler provenance on small fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_reproducible_run import make_partitions
from concept_audit_v3.integrity import validate_splits
from concept_audit_v3.interventions_v3 import TrainSupport, apply_edits, per_graph_edit_diagnostics
from torch_geometric.data import Batch


class PipelineTests(unittest.TestCase):
    def test_patient_groups_are_preserved(self):
        rows, mapping = [], []
        for group in range(35):
            for k, label in enumerate(["ADH", "DCIS", "FEA", "IC", "N", "PB", "UDH"]):
                sid = f"BRACS_{group*10+k}_{label}_1"
                rows.append(dict(sample_id=sid, label=label, graph_path=f"{sid}.pt", split="train"))
                mapping.append(dict(sample_id=sid, patient_id=str(group)))
        frames = make_partitions(pd.DataFrame(rows), patient_mapping=pd.DataFrame(mapping))
        validate_splits(frames)
        self.assertEqual(sum(map(len, frames.values())), len(rows))
        self.assertFalse(set(frames["train"].patient_id) & set(frames["test"].patient_id))

    def test_diagnostics_weight_graphs_not_batches(self):
        batch = Batch.from_data_list([Data(x=torch.tensor([[0.]])), Data(x=torch.tensor([[.9], [.9], [.9]]))])
        support = TrainSupport(np.array([-1.]), np.array([1.]), -1., 1., np.eye(1))
        changed, _, _ = apply_edits(batch, [("node",0,1.)], .5, support)
        actual, clip = per_graph_edit_diagnostics(batch, changed, ("node",0,1.), .5)
        np.testing.assert_allclose(actual, [.5,.1], atol=1e-6)
        np.testing.assert_allclose(clip, [0.,1.])

    def test_restoring_source_assignments(self):
        rows, mapping, assignments = [], [], []
        for i, split in enumerate(["train", "validation", "test"]):
            for k, label in enumerate(["ADH", "DCIS", "FEA", "IC", "N", "PB", "UDH"]):
                sid = f"BRACS_{i*100+k}_{label}_1"
                rows.append(dict(sample_id=sid,label=label,graph_path=f"{sid}.pt",split="train"))
                mapping.append(dict(sample_id=sid,patient_id=f"p{i}"))
                assignments.append(dict(sample_id=sid,partition=split))
        frames = make_partitions(pd.DataFrame(rows),patient_mapping=pd.DataFrame(mapping),
                                 source_partitions=pd.DataFrame(assignments))
        self.assertEqual(set(frames['test'].sample_id),{r['sample_id'] for r in assignments if r['partition']=='test'})
        self.assertTrue(all(v==0 for v in validate_splits(frames).values()))
        bad = dict(frames)
        bad['test'] = frames['test'].assign(patient_id='p0')
        with self.assertRaisesRegex(ValueError,'Slide overlap'):
            validate_splits(bad)

    def test_scaler_and_training_accept_new_split_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "data/graph-bracs").mkdir(parents=True)
            frames = {}
            torch.manual_seed(3)
            for i, split in enumerate(["train", "validation", "test"]):
                rows = []
                for k, label in enumerate(["ADH", "DCIS", "FEA", "IC", "N", "PB", "UDH"]):
                    sid = f"BRACS_{i*100+k}_{label}_1"
                    path = folder / f"{sid}.pt"
                    graph = Data(x=torch.randn(4,96), edge_index=torch.tensor([[0,1,2,3],[1,2,3,0]]),
                                 edge_attr=torch.rand(4,1)+1)
                    torch.save(graph,path)
                    rows.append(dict(sample_id=sid,label=label,graph_path=str(path),split=split))
                frames[split] = pd.DataFrame(rows)
                frames[split].to_csv(folder/f"{split}.csv",index=False)
            pd.concat(frames.values()).to_csv(folder/'data/graph-bracs/metadata.csv',index=False)
            commands = [
                ["scripts/fit_graph_bracs_scaler.py", "--train-csv", "train.csv", "--output", "scaler.npz"],
                ["scripts/train_gine_baseline.py", "--train-csv", "train.csv", "--val-csv", "validation.csv",
                 "--test-csv", "test.csv", "--scaler", "scaler.npz", "--output", "model.pt",
                 "--epochs", "1", "--hidden-dim", "16", "--threads", "2"],
                ["scripts/run_audit_v3.py", "--checkpoint", "model.pt", "--train-csv", "train.csv",
                 "--val-csv", "validation.csv", "--test-csv", "test.csv", "--out-dir", "audit",
                 "--skip", "baselines", "joint", "--n-random-features", "0", "--n-boot", "2", "--threads", "2"]]
            for command in commands:
                process = subprocess.run([sys.executable,str(ROOT/command[0]),*command[1:]],cwd=folder,
                                         capture_output=True,text=True,timeout=180)
                self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            with np.load(folder/'scaler.npz') as scaler:
                self.assertEqual(set(scaler['train_ids']),set(frames['train'].sample_id))
            metrics = json.loads((folder/'model.metrics.json').read_text())
            self.assertEqual(metrics['protocol'],'internal_grouped_exploratory')
            manifest = json.loads((folder/'audit/manifest.json').read_text())
            self.assertEqual(manifest['status'],'completed')
            self.assertEqual(manifest['splits'],['test'])
            self.assertTrue(manifest['training_ids_verified'])
            with np.load(folder/'audit/interventions_test.npz') as arrays:
                self.assertEqual(arrays['cell_size__isolated__actual'].shape,(7,7))
                np.testing.assert_array_equal(arrays['sample_ids'],frames['test'].sample_id)


if __name__ == '__main__':
    unittest.main()
