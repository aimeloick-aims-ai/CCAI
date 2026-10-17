"""Tables with WSI-cluster bootstrap CIs from evaluate_variants / evaluate_joint outputs."""
import numpy as np
import pandas as pd

from .stats import cluster_bootstrap, cluster_bootstrap_f1_delta

CONCEPTS = ["cell_size", "cell_shape", "cell_intensity", "cell_texture", "spatial_distance"]


def intervention_table(store, meta, groups, n_boot=1000, n_tests=None, seed=0):
    """One row per concept x dose: concept effect minus the random-feature null, with CIs.

    effect_vs_randfeat is the PAIRED per-graph difference between the concept shift and the mean
    over K random-feature shifts of the same size. CI level is Bonferroni-adjusted over all cells.
    """
    alphas = meta["alphas"]
    nz = [i for i, a in enumerate(alphas) if a != 0]
    n_tests = n_tests or 2 * len(CONCEPTS) * len(nz)
    ci_alpha = 0.05 / n_tests
    y, base_pred = meta["y"], meta["base_pred"]
    rows = []
    for c in CONCEPTS:
        keys_rf = sorted(k for k in store if k[0] == c and k[1].startswith("randfeat"))
        rf = np.stack([store[k]["dlogit"] for k in keys_rf]) if keys_rf else None  # (K, A, N)
        for i in nz:
            row = dict(concept=c, alpha=float(alphas[i]))
            for variant in ("isolated", "coherent"):
                rec = store[(c, variant)]
                d = rec["dlogit"][i]
                pt, lo, hi = cluster_bootstrap(d, groups, n_boot=n_boot, seed=seed, alpha=0.05)
                row[f"{variant}_dlogit"], row[f"{variant}_dlogit_lo"], row[f"{variant}_dlogit_hi"] = pt, lo, hi
                f1, f1lo, f1hi = cluster_bootstrap_f1_delta(
                    y, base_pred, rec["pred"][i], groups, n_boot=max(200, n_boot // 3), seed=seed
                )
                row[f"{variant}_dF1"], row[f"{variant}_dF1_lo"], row[f"{variant}_dF1_hi"] = f1, f1lo, f1hi
                row[f"{variant}_clip"] = float(np.mean(rec["clip"][i]))
                row[f"{variant}_actual_change"] = float(np.mean(rec["actual"][i]))
            if rf is not None:
                null_mean = rf[:, i, :].mean(axis=0)
                for variant in ("isolated", "coherent"):
                    diff = store[(c, variant)]["dlogit"][i] - null_mean
                    pt, lo, hi = cluster_bootstrap(diff, groups, n_boot=n_boot, seed=seed, alpha=ci_alpha)
                    row[f"{variant}_vs_randfeat"] = pt
                    row[f"{variant}_vs_randfeat_lo_adj"], row[f"{variant}_vs_randfeat_hi_adj"] = lo, hi
                    row[f"{variant}_beyond_null"] = bool(lo > 0 or hi < 0)
                feat_means = rf[:, i, :].mean(axis=1)  # K means
                row["randfeat_dlogit_mean"] = float(feat_means.mean())
                row["randfeat_dlogit_min"], row["randfeat_dlogit_max"] = float(feat_means.min()), float(feat_means.max())
            else:
                for variant in ("isolated", "coherent"):
                    row[f"{variant}_vs_randfeat"] = np.nan
                    row[f"{variant}_vs_randfeat_lo_adj"] = np.nan
                    row[f"{variant}_vs_randfeat_hi_adj"] = np.nan
                    row[f"{variant}_beyond_null"] = np.nan
                row["randfeat_dlogit_mean"] = np.nan
                row["randfeat_dlogit_min"] = np.nan
                row["randfeat_dlogit_max"] = np.nan
            rows.append(row)
            row["comparison_status"] = "exploratory_descriptor_control"
            row["edge_control_matched"] = c != "spatial_distance"
            row["adjusted_ci_expected_tail_replicates"] = n_boot * ci_alpha / 2
    return pd.DataFrame(rows)


def joint_table(joint, combos, alphas, groups, n_boot=1000, seed=0):
    rows = []
    for (a, b, ci, ai), rec in joint.items():
        gap = rec["joint"] - rec["single_a"] - rec["single_b"]
        pt, lo, hi = cluster_bootstrap(gap, groups, n_boot=n_boot, seed=seed)
        rows.append(dict(concept_a=a, concept_b=b, signs=combos[ci], alpha=alphas[ai],
                         joint_dlogit=float(rec["joint"].mean()),
                         sum_single_dlogit=float((rec["single_a"] + rec["single_b"]).mean()),
                         interaction=pt, interaction_lo=lo, interaction_hi=hi,
                         flip_rate=float(rec["flip"].mean())))
    return pd.DataFrame(rows)


def save_npz(path, store, meta, joint=None, combos=None):
    import os

    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite {path}")
    arrays = {"alphas": meta["alphas"], "y": meta["y"], "base_pred": meta["base_pred"],
              "base_true_logit": meta["base_true_logit"]}
    if "sample_ids" in meta:
        arrays["sample_ids"] = np.asarray(meta["sample_ids"], dtype=str)
    for (c, v), rec in store.items():
        for k in ("dlogit", "pred", "actual", "clip"):
            arrays[f"{c}__{v}__{k}"] = rec[k]
    for c, feats in meta["random_features"].items():
        arrays[f"{c}__random_feature_indices"] = np.asarray(feats)
    if joint is not None:
        for (a, b, ci, ai), rec in joint.items():
            for k, v in rec.items():
                arrays[f"joint__{a}__{b}__s{ci}__a{ai}__{k}"] = v
    np.savez_compressed(path, **arrays)


def symmetry_table(store, groups, alphas, n_boot=1000, seed=0):
    """Split each dose effect into algebraic odd and even components.

    For dose a: odd = (d(+a) - d(-a)) / 2 and even = (d(+a) + d(-a)) / 2, per graph.
    Neither component identifies biological usage or departure from the data distribution.
    Nonlinearity and asymmetric clipping can contribute to either component.
    The random-feature range is descriptive, not a calibrated hypothesis test.
    """
    alphas = np.asarray(alphas, dtype=float)
    rows = []
    for c in CONCEPTS:
        rf_keys = sorted(k for k in store if k[0] == c and k[1].startswith("randfeat"))
        for variant in ("isolated", "coherent"):
            d = store[(c, variant)]["dlogit"]
            for i in [i for i, a in enumerate(alphas) if a > 0]:
                j = int(np.where(np.isclose(alphas, -alphas[i]))[0][0])
                odd, even = (d[i] - d[j]) / 2, (d[i] + d[j]) / 2
                row = dict(concept=c, variant=variant, alpha=float(alphas[i]))
                for name, v in (("odd", odd), ("even", even)):
                    pt, lo, hi = cluster_bootstrap(v, groups, n_boot=n_boot, seed=seed)
                    row[name], row[f"{name}_lo"], row[f"{name}_hi"] = pt, lo, hi
                if rf_keys:
                    rf = np.stack([store[k]["dlogit"] for k in rf_keys])
                    rodd, reven = (rf[:, i] - rf[:, j]).mean(1) / 2, (rf[:, i] + rf[:, j]).mean(1) / 2
                    row["randfeat_odd_min"], row["randfeat_odd_max"] = float(rodd.min()), float(rodd.max())
                    row["randfeat_even_min"], row["randfeat_even_max"] = float(reven.min()), float(reven.max())
                    row["odd_beyond_randfeat_range"] = bool(row["odd"] < rodd.min() or row["odd"] > rodd.max())
                    row["even_beyond_randfeat_range"] = bool(row["even"] < reven.min() or row["even"] > reven.max())
                row["null_matched"] = c != "spatial_distance"  # no feature-space null exists for the edge feature
                rows.append(row)
    return pd.DataFrame(rows)


def load_npz_store(path):
    """Rebuild {(concept, variant): {'dlogit', 'pred', ...}} and alphas from save_npz output."""
    z = np.load(path, allow_pickle=True)
    store = {}
    for key in z.files:
        parts = key.split("__")
        if len(parts) == 3 and parts[2] in ("dlogit", "pred", "actual", "clip"):
            store.setdefault((parts[0], parts[1]), {})[parts[2]] = z[key]
    return store, z["alphas"], z["y"], z["base_pred"]
