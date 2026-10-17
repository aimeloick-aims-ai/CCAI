"""Add the odd/even decomposition to an EXISTING v3 run, without re-running any model.

Reads <run-dir>/interventions_<split>.npz (per-graph arrays saved by run_audit_v3.py) and the TEST ids
from data/graph-bracs/metadata.csv (same order as the loader) to build slide clusters.

    python scripts/postprocess_symmetry_v3.py results/audit_v3_grouped_seed42
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from concept_audit_v3 import report_v3  # noqa: E402
from concept_audit_v3.stats import wsi_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--split", default="official_test", choices=["official_test"])
    ap.add_argument("--metadata", default="data/graph-bracs/metadata.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = run / f"symmetry_{a.split}.csv"
    if out.exists():
        sys.exit(f"Refusing to overwrite {out}")
    store, alphas, y, _ = report_v3.load_npz_store(run / f"interventions_{a.split}.npz")
    ids = pd.read_csv(a.metadata).query("split == 'test'").sample_id
    with np.load(run / f"interventions_{a.split}.npz") as arrays:
        if "sample_ids" not in arrays:
            sys.exit("Historical output lacks ordered sample IDs: cannot verify slide alignment.")
        if not np.array_equal(arrays["sample_ids"], ids.to_numpy(dtype=str)):
            sys.exit("Ordered sample IDs do not match metadata.")
    if len(ids) != len(y):
        sys.exit(f"Order/size mismatch: {len(ids)} test ids vs {len(y)} graphs in the npz (was --max-graphs used?)")
    table = report_v3.symmetry_table(store, wsi_ids(ids), alphas, n_boot=a.n_boot)
    table.to_csv(out, index=False)
    cols = ["concept", "variant", "alpha", "odd", "odd_lo", "odd_hi", "even", "even_lo", "even_hi",
            "odd_beyond_randfeat_range", "even_beyond_randfeat_range"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
