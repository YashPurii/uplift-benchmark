"""
Experiment 2 -- Model separability vs. test-set size (paper Section 5.2,
Figure 4). Their central empirical claim: at the sample sizes of prior
uplift benchmarks (IHDP ~1k, Jobs ~5k, Hillstrom ~50k), confidence
intervals of different uplift models overlap almost completely -- you
literally cannot tell which model is better. Only past ~1M rows do the
models separate. This experiment reproduces that curve, and adds the
project's own GBM-uplift model to see whether it separates from the
2018-era baselines earlier (i.e. whether a stronger model needs less
data to prove itself).

Usage:
    python experiments/02_model_separability.py [--data path/to/real.csv] [--n 500000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as data_mod
from src import features as feat_mod
from src import uplift_models
from src import uplift_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--n", type=int, default=500_000, help="Synthetic sample size if --data omitted")
    ap.add_argument("--test_sizes", type=int, nargs="+", default=[1_000, 5_000, 50_000, 200_000])
    ap.add_argument("--out_fig", type=str, default="outputs/figures/model_separability.png")
    ap.add_argument("--out_table", type=str, default="outputs/tables/model_separability.csv")
    args = ap.parse_args()

    if args.data:
        df = data_mod.load_real(args.data)
    else:
        print(f"[DEV MODE] Generating {args.n} synthetic rows.")
        df = data_mod.generate_synthetic(n=args.n)

    train, test = data_mod.train_test_split_stratified(df, test_size=0.2)
    print(f"Train: {len(train):,} | Test pool: {len(test):,}")

    X_train = feat_mod.encode(train, n_projections=8)
    t_train = train[data_mod.TREATMENT_COL].values
    y_train = train["visit"].values  # paper recommends visit over conversion (too sparse)

    print("Fitting models on training set...")
    models = {
        "two_model": uplift_models.TwoModel().fit(X_train, t_train, y_train),
        "class_transform": uplift_models.ClassVariableTransformation().fit(X_train, t_train, y_train),
        "gbm_uplift (ours)": uplift_models.GradientBoostedUplift().fit(X_train, t_train, y_train),
    }

    X_test_full = feat_mod.encode(test, n_projections=8)
    t_test_full = test[data_mod.TREATMENT_COL].values
    y_test_full = test["visit"].values
    n_test = len(test)

    rows = []
    seen_sizes = set()
    for size in args.test_sizes:
        size = min(size, n_test)
        if size in seen_sizes:
            continue
        seen_sizes.add(size)
        rng = np.random.default_rng(size)
        idx = rng.choice(n_test, size=size, replace=False)
        X_s, t_s, y_s = X_test_full[idx], t_test_full[idx], y_test_full[idx]

        for name, model in models.items():
            scores = model.predict_uplift(X_s)
            res = uplift_metrics.auuc_bootstrap_ci(scores, t_s, y_s, n_boot=100)
            rows.append({"test_size": size, "model": name, **res})
            print(f"  size={size:>7,} | {name:<20} | AUUC={res['auuc']:+.5f} "
                  f"[{res['ci_low']:+.5f}, {res['ci_high']:+.5f}]")

    result_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_table), exist_ok=True)
    result_df.to_csv(args.out_table, index=False)

    # Plot: AUUC vs test size, error bars = bootstrap CI, log-x axis (mirrors Fig 4)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in models:
        sub = result_df[result_df.model == name].sort_values("test_size")
        yerr = [sub.auuc - sub.ci_low, sub.ci_high - sub.auuc]
        ax.errorbar(sub.test_size, sub.auuc, yerr=yerr, marker="o", capsize=4, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Test set size (log scale)")
    ax.set_ylabel("AUUC")
    ax.set_title("Model separability vs. sample size\n(CIs overlap at small N -- paper's core finding)")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150)
    print(f"\nSaved figure to {args.out_fig}, table to {args.out_table}")


if __name__ == "__main__":
    main()
