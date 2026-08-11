"""
Experiment 0 -- EDA: replicate the headline dataset stats (Table 1) and the
PCA-binned uplift pattern (Figure 3) that motivated the multi-peaked
response surface in the first place.

Usage:
    python experiments/00_eda.py [--data path/to/real.csv] [--n 300000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as data_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--out_fig", type=str, default="outputs/figures/eda_pca_uplift.png")
    args = ap.parse_args()

    if args.data:
        df = data_mod.load_real(args.data)
    else:
        print(f"[DEV MODE] Generating {args.n} synthetic rows.")
        df = data_mod.generate_synthetic(n=args.n)

    print("=== Headline stats ===")
    print(f"Rows: {len(df):,}")
    print(f"Treatment ratio: {df[data_mod.TREATMENT_COL].mean():.3f}")
    print(f"Visit rate: {100 * df['visit'].mean():.2f}%")
    print(f"Conversion rate: {100 * df['conversion'].mean():.2f}%")
    for outcome in ["visit", "conversion"]:
        y_t = df.loc[df[data_mod.TREATMENT_COL] == 1, outcome].mean()
        y_c = df.loc[df[data_mod.TREATMENT_COL] == 0, outcome].mean()
        rel_uplift = 100 * (y_t - y_c) / (y_c + 1e-12)
        print(f"Relative avg uplift ({outcome}): {rel_uplift:.1f}%")

    X_cont = StandardScaler().fit_transform(df[data_mod.CONTINUOUS_FEATURES].values)
    pc1 = PCA(n_components=1, random_state=42).fit_transform(X_cont).ravel()

    bins = pd.qcut(pc1, q=40, duplicates="drop")
    tmp = pd.DataFrame({"bin": bins, "t": df[data_mod.TREATMENT_COL].values, "visit": df["visit"].values})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for t_val, label, color in [(1, "Treatment", "tab:red"), (0, "Control", "tab:blue")]:
        sub = tmp[tmp.t == t_val].groupby("bin", observed=True)["visit"].mean() * 100
        axes[0].plot(range(len(sub)), sub.values, label=label, color=color, alpha=0.8)
    axes[0].set_title("Visit % by PCA bin")
    axes[0].set_xlabel("PCA component 1 (binned)")
    axes[0].set_ylabel("Visit %")
    axes[0].legend()

    uplift_by_bin = (tmp[tmp.t == 1].groupby("bin", observed=True)["visit"].mean()
                      - tmp[tmp.t == 0].groupby("bin", observed=True)["visit"].mean()) * 100
    axes[1].bar(range(len(uplift_by_bin)), uplift_by_bin.values, color="tab:green", alpha=0.8)
    axes[1].set_title("Uplift % by PCA bin\n(non-monotonic pattern -> motivates multi-peaked surface)")
    axes[1].set_xlabel("PCA component 1 (binned)")
    axes[1].set_ylabel("Uplift %")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    print(f"\nSaved figure to {args.out_fig}")


if __name__ == "__main__":
    main()
