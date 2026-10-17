"""Exact model-level controls: known edge dependence and zero node dependence.

Unlike independence from dataset labels, zero classifier dependence is guaranteed
by construction. This validates computation, not the biology of real edits.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from concept_audit_v3.interventions_v3 import evaluate_variants, fit_train_support


class EdgeMean(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr):
        out = torch.zeros_like(x)
        counts = torch.zeros_like(x)
        out.index_add_(0, edge_index[1], edge_attr)
        counts.index_add_(0, edge_index[1], torch.ones_like(edge_attr))
        return out / counts.clamp_min(1)


class KnownEdgeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = torch.nn.Linear(96,1,bias=False)
        self.convs = torch.nn.ModuleList([EdgeMean()])
        self.norms = torch.nn.ModuleList([torch.nn.Identity()])
        self.classifier = torch.nn.Linear(2,7,bias=False)
        self.dropout = 0.
        with torch.no_grad():
            self.input_projection.weight.zero_()
            self.classifier.weight.zero_()
            self.classifier.weight[0,0] = 1.


class OracleTests(unittest.TestCase):
    def test_known_functional_and_unused_features(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(10)
        graphs = []
        for k in range(12):
            n = 5 + k
            src = torch.arange(n)
            graphs.append(Data(x=torch.randn(n,96,generator=generator),
                               edge_index=torch.stack([src,(src+1)%n]),
                               edge_attr=torch.rand(n,1,generator=generator)+1.,y=torch.tensor(0)))
        loader = DataLoader(graphs,batch_size=4,shuffle=False)
        support = fit_train_support(loader,torch.device('cpu'))
        model = KnownEdgeClassifier().eval()
        store,meta = evaluate_variants(model,loader,torch.device('cpu'),support,n_random_features=3,
                                       alphas=(-.25,0.,.25))
        for (concept,variant),record in store.items():
            np.testing.assert_allclose(record['dlogit'][1],0.,atol=1e-7)
            if concept != 'spatial_distance' or variant.startswith('randfeat'):
                np.testing.assert_allclose(record['dlogit'],0.,atol=1e-7)
        spatial = store[('spatial_distance','isolated')]
        self.assertTrue((spatial['dlogit'][0] < 0).all())
        self.assertTrue((spatial['dlogit'][2] > 0).all())
        np.testing.assert_allclose(spatial['dlogit'],spatial['actual'],atol=1e-6)


if __name__ == '__main__':
    unittest.main()
