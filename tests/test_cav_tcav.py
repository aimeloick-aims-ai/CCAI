"""Numerical checks for graph CAV directions, derivative orientation and controls."""
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_cav_tcav import fit_directions,holm,patient_interval


class CavTests(unittest.TestCase):
    def test_weighted_ridge_and_orientation(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=(120,6))*np.arange(1,7)
        target=x[:,0]+rng.normal(0,.1,120)
        patients=np.repeat(np.arange(30),4)
        directions,coef,intercept,info=fit_directions(x,target,patients,n_controls=19)
        keep=(target<=info['q20'])|(target>=info['q80'])
        labels=np.where(target[keep]>=info['q80'],1.,-1.)
        _,inv=np.unique(patients[keep],return_inverse=True)
        weights=1/np.bincount(inv)[inv];weights*=len(weights)/weights.sum()
        center=np.average(x[keep],axis=0,weights=weights)
        scale=np.sqrt(np.average((x[keep]-center)**2,axis=0,weights=weights))
        fitted=Ridge(alpha=1.).fit((x[keep]-center)/scale,labels,sample_weight=weights)
        np.testing.assert_allclose(x[keep]@coef+intercept,fitted.predict((x[keep]-center)/scale),atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(directions,axis=0),1.,atol=1e-12)
        self.assertGreater(coef[0],0)
        self.assertEqual(directions.shape,(6,20))

    def test_holm_and_patient_interval(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])
        low,high=patient_interval(np.ones(6),np.repeat([0,1,2],2),n_boot=40)
        self.assertEqual((low,high),(1.,1.))

    def test_constant_concept_rejected(self):
        with self.assertRaisesRegex(ValueError,'coincide'):
            fit_directions(np.eye(5),np.ones(5),np.arange(5),n_controls=2)


if __name__=='__main__':
    unittest.main()
