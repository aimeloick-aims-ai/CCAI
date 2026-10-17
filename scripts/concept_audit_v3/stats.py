"""Statistics helpers: WSI parsing, cluster bootstrap, permutation p-values (numpy only)."""
import re

import numpy as np

_ID = re.compile(r"^BRACS_(\d+)_([A-Za-z]+)_(\d+)$")


def wsi_ids(sample_ids):
    """Slide id for each BRACS sample id (ROIs from one slide are not independent)."""
    out = []
    for s in sample_ids:
        m = _ID.match(str(s))
        if m is None:
            raise ValueError(f"Unparseable sample_id: {s}")
        out.append(m.group(1))
    return np.asarray(out)


def _cluster_index(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    return members


def cluster_bootstrap(values, groups, statistic=np.mean, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI for ``statistic(values)`` resampling whole groups (WSIs) with replacement.

    values: (n_graphs,) or (n_graphs, k). Returns (point, lo, hi) with matching trailing shape.
    ``statistic`` must reduce axis 0.
    """
    values = np.asarray(values, dtype=np.float64)
    members = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    point = statistic(values, axis=0) if _takes_axis(statistic) else statistic(values)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[p] for p in pick])
        boots.append(statistic(values[idx], axis=0) if _takes_axis(statistic) else statistic(values[idx]))
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2], axis=0)
    return point, lo, hi


def _takes_axis(fn):
    return fn in (np.mean, np.median, np.sum, np.std)


def cluster_bootstrap_macro_f1(y, pred, groups, num_classes=7, n_boot=1000, seed=0, alpha=0.05):
    """WSI-cluster bootstrap CI of macro-F1 (graph-level resampling is not valid here)."""
    from sklearn.metrics import f1_score

    y, pred = np.asarray(y), np.asarray(pred)
    members = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    labels = list(range(num_classes))
    point = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[p] for p in pick])
        boots.append(f1_score(y[idx], pred[idx], labels=labels, average="macro", zero_division=0))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def permutation_pvalue(observed, null, two_sided=True):
    """Add-one permutation p-value. With n permutations the smallest attainable value is 1/(n+1)."""
    null = np.asarray(null, dtype=np.float64)
    n = null.shape[-1]
    centre = null.mean(axis=-1)
    if two_sided:
        exceed = (np.abs(null - centre[..., None]) >= np.abs(observed - centre)[..., None]).sum(axis=-1)
    else:
        exceed = (null >= np.asarray(observed)[..., None]).sum(axis=-1)
    return (1.0 + exceed) / (n + 1.0)


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(ranked, 1.0)
    return out


def cluster_bootstrap_f1_delta(y, pred_base, pred_mod, groups, num_classes=7, n_boot=1000, seed=0, alpha=0.05):
    """Paired WSI-cluster bootstrap of macro-F1(modified) - macro-F1(base)."""
    from sklearn.metrics import f1_score

    y, pb, pm = np.asarray(y), np.asarray(pred_base), np.asarray(pred_mod)
    members = _cluster_index(groups)
    rng = np.random.default_rng(seed)
    labels = list(range(num_classes))

    def delta(idx):
        return f1_score(y[idx], pm[idx], labels=labels, average="macro", zero_division=0) - f1_score(
            y[idx], pb[idx], labels=labels, average="macro", zero_division=0
        )

    point = delta(np.arange(len(y)))
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        boots.append(delta(np.concatenate([members[p] for p in pick])))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)
