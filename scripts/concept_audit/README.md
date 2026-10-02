---
noteId: "bf08c060bd9311f187e5c3ce3f0ac4e5"
tags: []

---

# Modular Graph-BRACS Concept Audit

This replaces the previous monolithic `concept_audit_phase1.py`.

## Structure

- `config.py` — paths, constants, concept definitions.
- `utils.py` — frozen GINE encoding, node sampling, graph target, erasure.
- `ridge.py` — Ridge statistics, probes, and prediction metrics.
- `metrics.py` — classification confusion/F1 metrics.
- `controls.py` — matched random directions.
- `data.py` — BRACS loaders, scaler, frozen GINE checkpoint.
- `probes.py` — probe fitting and concept decodability.
- `erasure.py` — residual probes and representational utilization.
- `results.py` — compact `.npz` result export.
- `main.py` — orchestration.
- `../concept_audit_phase1.py` — simple entry point.

## Important correction

The Ridge probe needs the full representation covariance:

`sum_xx = X.T @ X`

The previous version created `sum_xx` incorrectly and did not update it. This version uses:

```python
"sum_xx": np.zeros((dim, dim), dtype=np.float64)
```

and, for every batch:

```python
stats["sum_xx"] += X.T @ X
```

This gives the expected shapes:

- node representation: `128`
- node concepts: `4`
- `sum_xx`: `(128, 128)`
- `sum_xy`: `(128, 4)`
- probe coefficients: `(128, 4)`
- concept directions: `(4, 128)`

## Installation

Copy the `scripts` directory from this package into the project root, replacing the old:

`scripts/concept_audit_phase1.py`

while keeping your existing:

`scripts/train_gine_baseline.py`

## Run

From the project root:

```powershell
python scripts\concept_audit_phase1.py
```

The pipeline does not create intermediate probe files. It keeps intermediate objects in memory and writes only:

`results\concept_audit_phase1.npz`

The official test split is never used to fit the GINE or the probes. It is used only for final evaluation after fitting on the training split.
