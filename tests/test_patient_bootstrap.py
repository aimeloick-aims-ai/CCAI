"""Validate cluster-weighted confusion counts against explicit patient resampling."""
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_completed_report import bootstrap_f1


class PatientBootstrapTests(unittest.TestCase):
    def test_unequal_patient_sizes_and_missing_classes(self):
        y=np.array([0,1,1,2,2,0])
        pred=np.array([0,0,1,2,1,1])
        groups=np.array([0,0,1,2,2,2])
        weights=np.array([[1,1,1],[2,0,1],[0,3,0]])
        actual=bootstrap_f1(y,pred,groups,weights)
        expected=[]
        for counts in weights:
            indices=np.concatenate([np.tile(np.flatnonzero(groups==g),count) for g,count in enumerate(counts)])
            expected.append(f1_score(y[indices],pred[indices],labels=range(7),average='macro',zero_division=0))
        np.testing.assert_allclose(actual,expected,atol=1e-12)


if __name__=='__main__':
    unittest.main()
