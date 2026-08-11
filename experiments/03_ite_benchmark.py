"""
Experiment 3 -- ITE prediction benchmark across response surfaces (paper
Section 5.3, Table 5). For each surface (Case A, Case B, multi-peaked) and
each of several realizations, fits T/X/R-learners and reports mean sqrt(PEHE)
+/- std, matching the paper's protocol (10 realizations, 5-fold CV tuning
in spirit -- we keep model complexity fixed rather than grid-searching, to
keep runtime sane; see README for the tradeoff note).

Usage:
    python experiments/03_ite_benchmark.py [--data path/to/real.csv] [--n 100000] [--realizations 5]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as data_mod
from src import features as feat_mod
from src import response_surfaces as rs
from src import ite_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--n", type=int, default=100_000, help="Rows to use for the ITE benchmark")
    ap.add_argument("--realizations", type=int, default=5)
    ap.add_argument("--out", type=str, default="outputs/tables/ite_benchmark.csv")
    args = ap.parse_args()

    if args.data:
        df = data_mod.load_real(args.data).sample(n=min(args.n, 10**9), random_state=0)
    else:
        print(f"[DEV MODE] Generating {args.n} synthetic rows.")
        df = data_mod.generate_synthetic(n=args.n)

    # sparse=False: the ITE benchmark always runs on a bounded subsample
    # (see --n below), and response_surfaces.py needs dense arrays for the
    # RBF distance computations in the multi-peaked surface.
    X_full = feat_mod.encode(df, n_projections=5, sparse=False)  # paper uses 5 projections -> dim 32-ish here too
    print(f"Feature matrix: {X_full.shape}")

    surfaces = ["case_a", "case_b", "multi_peaked"]
    models_cls = ite_models.ITE_MODEL_REGISTRY

    rows = []
    for surface_name in surfaces:
        for real_i in range(args.realizations):
            seed = 1000 * real_i + hash(surface_name) % 997
            treatment, y_factual, tau_true, mu0, mu1 = rs.generate_ite_dataset(
                X_full, surface_name, important_col=0, delta=0.01, seed=seed
            )
            idx = np.random.default_rng(seed).permutation(len(X_full))
            split = len(idx) // 2
            train_idx, test_idx = idx[:split], idx[split:]

            X_tr, t_tr, y_tr = X_full[train_idx], treatment[train_idx], y_factual[train_idx]
            X_te, tau_te = X_full[test_idx], tau_true[test_idx]

            for model_name, cls in models_cls.items():
                model = cls(seed=seed).fit(X_tr, t_tr, y_tr)
                tau_pred = model.predict_ite(X_te)
                pehe = ite_models.sqrt_pehe(tau_te, tau_pred)
                rows.append({"surface": surface_name, "realization": real_i,
                             "model": model_name, "sqrt_pehe": pehe})
            print(f"  surface={surface_name:<13} realization={real_i} done")

    result_df = pd.DataFrame(rows)
    summary = result_df.groupby(["surface", "model"])["sqrt_pehe"].agg(["mean", "std"]).reset_index()
    print("\n=== Summary (mean +/- std sqrt(PEHE), lower is better) ===")
    print(summary.to_string(index=False))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result_df.to_csv(args.out, index=False)
    summary.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
