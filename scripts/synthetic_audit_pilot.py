"""D/C pilot with functional ground truth; not a trained-GNN benchmark."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from concept_audit_phase2 import fit_cav, make_balanced_quantile_set


OUTPUT = Path("results/synthetic_audit_pilot")
SEEDS = range(10)
RHOS = [0., .5, .9, .99]
MIXINGS = {"direct": np.eye(3), "mixed": np.array([[1., .6, .2], [.2, 1., .3], [.1, .2, 1.]])}


def graphs(rng, n, rho):
    u = rng.normal(size=(n, 3))
    c = u.copy()
    c[:, 1] = rho * u[:, 0] + np.sqrt(1 - rho ** 2) * u[:, 1]
    noise = rng.normal(scale=.1, size=(n, 8, 3))
    noise -= noise.mean(axis=1, keepdims=True)
    x = c[:, None, :] + noise
    assert np.allclose(x.mean(axis=1), c)
    return x, c


def response(c, kind):
    z = c[:, 0] + .5 * c[:, 2]
    return z if kind == "linear" else np.tanh(z)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    rows = []
    for seed in SEEDS:
        for rho in RHOS:
            rng = np.random.default_rng(seed)
            train_x, train_c = graphs(rng, 1200, rho)
            test_x, test_c = graphs(rng, 400, rho)
            for mixing, matrix in MIXINGS.items():
                train_h = train_x.mean(axis=1) @ matrix.T
                test_h = test_x.mean(axis=1) @ matrix.T
                # Estimator receives TRAIN activations and concept annotations.
                design = np.column_stack([train_h, np.ones(len(train_h))])
                probe = np.linalg.lstsq(design, train_c, rcond=None)[0][:3]
                for j in range(3):
                    target = np.eye(3)[j]
                    corrected = np.linalg.lstsq(probe.T, target, rcond=None)[0]
                    q20, q80 = np.quantile(train_c[:, j], [.2, .8])
                    prototype = train_h[train_c[:, j] >= q80].mean(0) - train_h[train_c[:, j] <= q20].mean(0)
                    X, y, _, _ = make_balanced_quantile_set(train_h, train_c[:, j], seed)
                    cav = fit_cav(X, y, seed)["direction"]
                    directions = {"prototype": prototype, "logistic_cav": cav, "constrained_probe": corrected,
                                  "oracle": matrix[:, j]}
                    for method, direction in directions.items():
                        calibration = float(direction @ probe[:, j])
                        if abs(calibration) < 1e-10:
                            raise RuntimeError("Unidentifiable target direction")
                        direction = direction / calibration
                        for dose in [-.5, .5]:
                            modified_h = test_h + dose * direction
                            # Known decoder is used ONLY by the evaluator.
                            modified_c = np.linalg.solve(matrix, modified_h.T).T
                            oracle_c = test_c.copy()
                            oracle_c[:, j] += dose
                            drift = modified_c - test_c
                            for kind in ["linear", "tanh"]:
                                base = response(test_c, kind)
                                delta = response(modified_c, kind) - base
                                truth = response(oracle_c, kind) - base
                                ref_c = test_c.copy()
                                ref_c[:, 0] += dose
                                ref = float(np.mean(np.abs(response(ref_c, kind) - base)))
                                effect = float(np.mean(np.abs(delta)))
                                if j == 1:
                                    assert np.array_equal(response(oracle_c, kind), base)
                                rows.append(dict(seed=seed, rho=rho, representation=mixing, response=kind,
                                    concept=f"C{j + 1}", method=method, dose=dose,
                                    mean_abs_effect=effect, oracle_abs_effect=float(np.mean(np.abs(truth))),
                                    error=float(np.mean(np.abs(delta - truth))), reference_C1_effect=ref,
                                    false_positive=bool(j == 1 and effect > .05 * ref),
                                    actual_target_change=float(drift[:, j].mean()),
                                    max_non_target_drift=float(np.max(np.abs(drift[:, np.arange(3) != j])))))
        print(f"seed {seed + 1}/10 complete", flush=True)
    table = pd.DataFrame(rows)
    unused = table[table.concept == "C2"]
    summary = unused.groupby(["rho", "representation", "response", "method"]).agg(
        false_positive_rate=("false_positive", "mean"), mean_abs_effect=("mean_abs_effect", "mean"),
        mean_error=("error", "mean"), max_non_target_drift=("max_non_target_drift", "max")).reset_index()
    high = summary[summary.rho.isin([.9, .99])]
    prototype_pass = bool((high[high.method == "prototype"].false_positive_rate >= .8).all())
    corrected_pass = bool((high[high.method == "constrained_probe"].false_positive_rate <= .05).all())
    corrected = table[table.method == "constrained_probe"]
    error_pass = bool((corrected.error <= .1 * corrected.reference_C1_effect).all())
    control = summary[(summary.rho == 0) & (summary.representation == "direct") & (summary.method == "prototype")]
    control_pass = bool((control.false_positive_rate <= .2).all())
    decision = dict(prototype_high_rho_pass=prototype_pass, correction_pass=corrected_pass,
                    error_pass=error_pass, independent_control_pass=control_pass,
                    pilot_go=prototype_pass and corrected_pass and error_pass and control_pass,
                    paper_go=False, reason="Pilot uses fixed graph pooling and oracle readouts, not trained GNN architectures.")
    OUTPUT.mkdir(parents=True)
    table.to_csv(OUTPUT / "measurements.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
