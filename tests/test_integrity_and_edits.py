"""Regression checks for split leakage and the in-place displacement bug."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from concept_audit_v3.integrity import validate_splits
from concept_audit_v3.interventions_v3 import TrainSupport, apply_edits


class IntegrityTests(unittest.TestCase):
    def test_shared_slide_different_roi_rejected(self):
        frames = {"train": pd.DataFrame({"sample_id": ["BRACS_1_N_0"]}),
                  "test": pd.DataFrame({"sample_id": ["BRACS_1_N_1"]})}
        with self.assertRaisesRegex(ValueError, "Slide overlap"):
            validate_splits(frames)

    def test_displacement_and_clipping(self):
        batch = Data(x=torch.tensor([[0.], [0.8]]), edge_attr=torch.tensor([[0.]]))
        support = TrainSupport(np.array([-1.]), np.array([1.]), -1., 1., np.eye(1))
        for kind in ("node", "edge"):
            changed, actual, clip = apply_edits(batch, [(kind, 0, 1.)], 0.5, support)
            self.assertAlmostEqual(actual, 0.35 if kind == "node" else 0.5, places=6)
            self.assertAlmostEqual(clip, 0.5 if kind == "node" else 0.)
        self.assertTrue(torch.equal(batch.x, torch.tensor([[0.], [0.8]])))
        same, actual, clip = apply_edits(batch, [("node", 0, 1.)], 0., support)
        self.assertTrue(torch.equal(same.x, batch.x))
        self.assertEqual((actual, clip), (0., 0.))


if __name__ == "__main__":
    unittest.main()
