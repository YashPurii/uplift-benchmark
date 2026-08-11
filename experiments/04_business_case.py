"""
Experiment 4 -- Business case simulation (not in the paper; this project's
own contribution).

Converts the best uplift model's ranking into a targeting-budget decision:
"if we can only afford to target X% of the population, how much incremental
outcome/revenue does a real uplift model capture vs. random targeting?"
This is the framing a marketing or pharma commercial team actually needs --
AUUC alone doesn't answer "should we do this, and how much is it worth."

Usage:
    python experiments/04_business_case.py [--data path/to/real.csv] [--n 300000] [--value_per_outcome 45]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as data_mod
from src import features as feat_mod
from src import uplift_models
from src import business_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--value_per_outcome", type=float, default=45.0,
                     help="$ value assigned to one incremental 'visit' (placeholder -- "
                          "replace with real conversion economics)")
    ap.add_argument("--out_fig", type=str, default="outputs/figures/business_case.png")
    ap.add_argument("--out_table", type=str, default="outputs/tables/business_case.csv")
    args = ap.parse_args()

    if args.data:
        df = data_mod.load_real(args.data)
    else:
        print(f"[DEV MODE] Generating {args.n} synthetic rows.")
        df = data_mod.generate_synthetic(n=args.n)

    train, test = data_mod.train_test_split_stratified(df, test_size=0.3)
    X_train = feat_mod.encode(train, n_projections=8)
    X_test = feat_mod.encode(test, n_projections=8)
    t_train, y_train = train[data_mod.TREATMENT_COL].values, train["visit"].values
    t_test, y_test = test[data_mod.TREATMENT_COL].values, test["visit"].values

    print("Fitting GBM uplift model...")
    model = uplift_models.GradientBoostedUplift().fit(X_train, t_train, y_train)
    scores = model.predict_uplift(X_test)

    budgets = np.round(np.arange(0.05, 1.0, 0.05), 2)
    policy_df = business_sim.budget_constrained_policy_value(
        scores, t_test, y_test, budgets, value_per_outcome=args.value_per_outcome
    )
    os.makedirs(os.path.dirname(args.out_table), exist_ok=True)
    policy_df.to_csv(args.out_table, index=False)
    print(policy_df.to_string(index=False))

    summary = business_sim.summarize_business_case(policy_df, budget_of_interest=0.3)
    print("\n=== Executive summary @ 30% targeting budget ===")
    print(f"  Incremental revenue (targeted vs. no targeting model): "
          f"${summary['incremental_revenue_targeted']:,.0f}")
    print(f"  Incremental revenue (random targeting, same budget):  "
          f"${summary['incremental_revenue_random']:,.0f}")
    print(f"  Lift from using the model vs. random:                 "
          f"{summary['lift_vs_random_pct']:+.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(policy_df.budget_frac, policy_df.incremental_revenue_targeted, marker="o",
            label="Model-targeted policy")
    ax.plot(policy_df.budget_frac, policy_df.incremental_revenue_random, marker="s",
            linestyle="--", label="Random targeting (same budget)")
    ax.set_xlabel("Targeting budget (fraction of population)")
    ax.set_ylabel(f"Incremental revenue (USD, at USD{args.value_per_outcome:.0f}/outcome)")
    ax.set_title("Uplift-targeted vs. random budget allocation")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150)
    print(f"\nSaved figure to {args.out_fig}, table to {args.out_table}")


if __name__ == "__main__":
    main()
