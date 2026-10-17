"""Input interventions with the controls the v2 protocol lacked.

v2 compares a concept shift (same sign on every cell) with a random-SIGN shift of the same feature.
That null only tests coherence of direction, so any feature the model is sensitive to "beats" it.
v3 adds:
  * random-feature control : the same +alpha SD coherent shift applied to K other, uncorrelated features
  * coherent intervention  : shift the concept AND move correlated features by their conditional-mean
                             slope (area also moves axes/perimeter, ...), clipped to the TRAIN support
  * joint interventions    : pairs of concepts shifted together, with interaction = joint - sum of singles
  * per-graph outputs      : every metric is stored per graph so WSI-cluster bootstrap CIs are possible
Interventions edit input descriptors; they are not causal interventions on tissue.
"""
import itertools

import numpy as np
import torch

from concept_decodability_audit.config import GRAPH_CONCEPT, NODE_CONCEPTS, SEED
from concept_decodability_audit.utils import encode_batch, sample_nodes
from concept_functional_audit.common import INTERVENTION_ALPHAS

CONCEPTS = list(NODE_CONCEPTS) + [GRAPH_CONCEPT]


# ---------------------------------------------------------------------------------------------
# TRAIN support and correlation structure
# ---------------------------------------------------------------------------------------------
class TrainSupport:
    def __init__(self, x_min, x_max, edge_min, edge_max, corr):
        self.x_min, self.x_max = x_min, x_max
        self.edge_min, self.edge_max = float(edge_min), float(edge_max)
        self.corr = corr  # correlation of standardized node features on TRAIN nodes

    def tensors(self, device, dtype=torch.float32):
        return (
            torch.as_tensor(self.x_min, dtype=dtype, device=device),
            torch.as_tensor(self.x_max, dtype=dtype, device=device),
        )


def fit_train_support(train_loader, device, max_nodes=100):
    """Stream TRAIN graphs: per-feature min/max (standardized units) and feature correlation."""
    n = 0
    sx = sxx = None
    xmin = xmax = None
    emin, emax = np.inf, -np.inf
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            x = batch.x.double()
            bmin, bmax = x.min(0).values.cpu().numpy(), x.max(0).values.cpu().numpy()
            xmin = bmin if xmin is None else np.minimum(xmin, bmin)
            xmax = bmax if xmax is None else np.maximum(xmax, bmax)
            e = batch.edge_attr[:, 0]
            emin, emax = min(emin, float(e.min())), max(emax, float(e.max()))
            xs = x[sample_nodes(batch, max_nodes)].cpu().numpy()
            if sx is None:
                sx, sxx = np.zeros(xs.shape[1]), np.zeros((xs.shape[1],) * 2)
            n += len(xs)
            sx += xs.sum(0)
            sxx += xs.T @ xs
    mean = sx / n
    cov = sxx / n - np.outer(mean, mean)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / sd[:, None] / sd[None, :]
    return TrainSupport(xmin, xmax, emin, emax, corr)


# ---------------------------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------------------------
# An edit is (kind, index, coefficient): kind in {"node", "edge"}; the realised shift in
# standardized units is alpha * coefficient. A spec is a list of edits; edits[0] is the target.

def _bounded(original, candidate, lo, hi):
    lo_eff = torch.minimum(original, lo)  # never push a value back toward the support
    hi_eff = torch.maximum(original, hi)
    return torch.maximum(lo_eff, torch.minimum(hi_eff, candidate))


def apply_edits(batch, edits, alpha, support):
    """Return (modified batch, actual_change, clip_rate) for the first (target) edit."""
    out = batch.clone()
    if float(alpha) == 0.0:
        return out, 0.0, 0.0
    xmin, xmax = support.tensors(out.x.device, out.x.dtype)
    target_actual = target_clip = None
    for k, (kind, idx, coef) in enumerate(edits):
        if kind == "node":
            orig = out.x[:, idx].clone()
            cand = orig + float(alpha) * float(coef)
            new = _bounded(orig, cand, xmin[idx], xmax[idx])
            out.x[:, idx] = new
        else:
            orig = out.edge_attr[:, 0].clone()
            cand = orig + float(alpha) * float(coef)
            lo = torch.as_tensor(support.edge_min, dtype=orig.dtype, device=orig.device)
            hi = torch.as_tensor(support.edge_max, dtype=orig.dtype, device=orig.device)
            new = _bounded(orig, cand, lo, hi)
            out.edge_attr[:, 0] = new
        if k == 0:
            target_actual = float((new - orig).mean())
            target_clip = float((new != cand).float().mean())
    return out, target_actual, target_clip


def node_idx(concept):
    return NODE_CONCEPTS[concept]


def isolated_spec(concept, sign=1.0):
    if concept == GRAPH_CONCEPT:
        return [("edge", 0, sign)]
    return [("node", node_idx(concept), sign)]


def coherent_spec(concept, support, min_abs_corr=0.5):
    """Concept + correlated features moved by conditional-mean slope (std units per concept SD)."""
    if concept == GRAPH_CONCEPT:
        return isolated_spec(concept)  # one edge feature: nothing to propagate (geometry not rebuilt)
    idx = node_idx(concept)
    spec = [("node", idx, 1.0)]
    for j, r in enumerate(support.corr[:, idx]):
        if j != idx and abs(r) >= min_abs_corr:
            spec.append(("node", j, float(r)))
    return spec


def random_feature_specs(concept, support, n_features, seed, max_abs_corr=0.5):
    """Other node features, uncorrelated with the target, shifted by the same coherent alpha."""
    forbidden = set(NODE_CONCEPTS.values())
    d = len(support.x_min)
    if concept != GRAPH_CONCEPT:
        r = np.abs(support.corr[:, node_idx(concept)])
        candidates = [j for j in range(d) if j not in forbidden and r[j] < max_abs_corr]
    else:
        candidates = [j for j in range(d) if j not in forbidden]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=min(n_features, len(candidates)), replace=False)
    return [[("node", int(j), 1.0)] for j in chosen], [int(j) for j in chosen]


def per_graph_edit_diagnostics(original, modified, edit, alpha):
    """Equal-graph-weight displacement/clipping; independent of loader batch size."""
    kind, idx, coefficient = edit
    if kind == "node":
        before, after = original.x[:, idx], modified.x[:, idx]
        groups = original.batch
    else:
        before, after = original.edge_attr[:, 0], modified.edge_attr[:, 0]
        groups = original.batch[original.edge_index[0]]
    counts = torch.bincount(groups, minlength=original.num_graphs).clamp_min(1)
    actual = torch.zeros(original.num_graphs, device=before.device)
    clipped = torch.zeros_like(actual)
    actual.index_add_(0, groups, after - before)
    clipped.index_add_(0, groups, (after != before + float(alpha) * coefficient).float())
    return (actual / counts).cpu().numpy(), (clipped / counts).cpu().numpy()


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_variants(model, loader, device, support, n_random_features=12, min_abs_corr=0.5,
                      alphas=INTERVENTION_ALPHAS, seed=SEED, log_every=25):
    """Per-graph results for isolated / coherent / random-feature interventions on every concept."""
    n_graphs = len(loader.dataset)
    A = len(alphas)
    variants = {}
    chosen_features = {}
    for ci, concept in enumerate(CONCEPTS):
        variants[(concept, "isolated")] = isolated_spec(concept)
        variants[(concept, "coherent")] = coherent_spec(concept, support, min_abs_corr)
        specs, feats = random_feature_specs(concept, support, n_random_features, seed + 17 * ci, min_abs_corr)
        chosen_features[concept] = feats
        for k, spec in enumerate(specs):
            variants[(concept, f"randfeat{k}")] = spec

    store = {
        key: dict(
            dlogit=np.zeros((A, n_graphs), np.float32),
            pred=np.zeros((A, n_graphs), np.int8),
            actual=np.zeros((A, n_graphs), np.float32),
            clip=np.zeros((A, n_graphs), np.float32),
        )
        for key in variants
    }
    y_all = np.zeros(n_graphs, np.int8)
    base_pred = np.zeros(n_graphs, np.int8)
    base_true_logit = np.zeros(n_graphs, np.float32)

    offset = 0
    for b, batch in enumerate(loader):
        batch = batch.to(device)
        _, _, base = encode_batch(model, batch)
        n = batch.num_graphs
        sl = slice(offset, offset + n)
        row = torch.arange(n, device=device)
        y_all[sl] = batch.y.cpu().numpy()
        base_pred[sl] = base.argmax(1).cpu().numpy()
        base_true_logit[sl] = base[row, batch.y].cpu().numpy()
        for key, spec in variants.items():
            rec = store[key]
            for i, alpha in enumerate(alphas):
                modified, actual, clip = apply_edits(batch, spec, float(alpha), support)
                if float(alpha) == 0.0:
                    logits = base
                else:
                    _, _, logits = encode_batch(model, modified)
                rec["dlogit"][i, sl] = (logits[row, batch.y] - base[row, batch.y]).cpu().numpy()
                rec["pred"][i, sl] = logits.argmax(1).cpu().numpy()
                actual_graph, clip_graph = per_graph_edit_diagnostics(batch, modified, spec[0], alpha)
                rec["actual"][i, sl] = actual_graph
                rec["clip"][i, sl] = clip_graph
        offset += n
        if b % log_every == 0:
            print(f"  variants batch {b + 1}/{len(loader)}", flush=True)
    assert offset == n_graphs

    meta = dict(y=y_all, base_pred=base_pred, base_true_logit=base_true_logit,
                alphas=np.asarray(alphas), random_features=chosen_features)
    return store, meta


@torch.no_grad()
def evaluate_joint(model, loader, device, support, pairs=None, alphas=(-1.0, 1.0)):
    """Joint shifts of concept pairs (same and opposite sign) and their additivity gap per graph."""
    pairs = pairs or list(itertools.combinations(CONCEPTS, 2))
    n_graphs = len(loader.dataset)
    combos = [(1.0, 1.0), (1.0, -1.0)]
    out = {
        (a, b, ci, ai): dict(joint=np.zeros(n_graphs, np.float32), single_a=np.zeros(n_graphs, np.float32),
                             single_b=np.zeros(n_graphs, np.float32), flip=np.zeros(n_graphs, np.int8))
        for a, b in pairs for ci in range(len(combos)) for ai in range(len(alphas))
    }
    offset = 0
    for batch in loader:
        batch = batch.to(device)
        _, _, base = encode_batch(model, batch)
        n = batch.num_graphs
        sl = slice(offset, offset + n)
        row = torch.arange(n, device=device)

        def dtrue(modified):
            _, _, lg = encode_batch(model, modified)
            return (lg[row, batch.y] - base[row, batch.y]).cpu().numpy(), lg.argmax(1).ne(base.argmax(1)).cpu().numpy()

        for a, b in pairs:
            for ci, (sa, sb) in enumerate(combos):
                for ai, alpha in enumerate(alphas):
                    ea, eb = isolated_spec(a, sa)[0], isolated_spec(b, sb)[0]
                    mj, _, _ = apply_edits(batch, [ea, eb], alpha, support)
                    ma, _, _ = apply_edits(batch, [ea], alpha, support)
                    mb, _, _ = apply_edits(batch, [eb], alpha, support)
                    dj, fj = dtrue(mj)
                    rec = out[(a, b, ci, ai)]
                    rec["joint"][sl] = dj
                    rec["flip"][sl] = fj
                    rec["single_a"][sl] = dtrue(ma)[0]
                    rec["single_b"][sl] = dtrue(mb)[0]
        offset += n
    return out, combos
