"""
Business simulation layer -- this project's own addition, not in the paper.

The paper stops at AUUC. But the actual decision a marketing (or, by direct
analogy, a pharma commercial) team faces is: "we can only afford to target
X% of the population/HCP panel -- who goes on the list, and what's the
expected incremental return?" This module converts a model's uplift ranking
into that decision directly: given a budget (fraction of the population you
can afford to treat) and a $ value per positive outcome, it estimates
incremental conversions/revenue captured by targeting the model's top-ranked
individuals, benchmarked against random targeting (today's implicit
baseline in a lot of orgs) and against the population-level ATE-only
strategy (treat everyone, common when nobody has an uplift model at all).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def budget_constrained_policy_value(uplift_scores: np.ndarray, treatment: np.ndarray,
                                      outcome: np.ndarray, budget_fracs, value_per_outcome: float = 1.0) -> pd.DataFrame:
    """
    For each budget level (fraction of population you can afford to expose):
      - `targeted` policy: treat the top-ranked-by-uplift budget_frac of
         the population; among the rest, use control-group behavior as
         the counterfactual (standard uplift-curve assumption).
      - `random` policy: treat a random budget_frac of the population.
    Both are estimated from the same held-out RCT sample (T,Y observed),
    so the comparison is apples-to-apples and doesn't require an actual
    live rollout.
    """
    order = np.argsort(-uplift_scores)
    treatment, outcome = treatment[order], outcome[order]
    n = len(treatment)

    rows = []
    for frac in budget_fracs:
        k = max(1, int(frac * n))

        # targeted: top-k by predicted uplift
        t_mask = treatment[:k] == 1
        c_mask = treatment[:k] == 0
        y_t = outcome[:k][t_mask].mean() if t_mask.sum() > 0 else 0.0
        y_c = outcome[:k][c_mask].mean() if c_mask.sum() > 0 else 0.0
        incr_rate_targeted = y_t - y_c
        incr_outcomes_targeted = incr_rate_targeted * k

        # random baseline: overall treatment effect on a same-size random sample
        rand_mask_t = treatment == 1
        rand_mask_c = treatment == 0
        y_t_all = outcome[rand_mask_t].mean()
        y_c_all = outcome[rand_mask_c].mean()
        incr_rate_random = y_t_all - y_c_all
        incr_outcomes_random = incr_rate_random * k

        rows.append({
            "budget_frac": frac,
            "n_targeted": k,
            "incremental_outcomes_targeted": incr_outcomes_targeted,
            "incremental_outcomes_random": incr_outcomes_random,
            "lift_vs_random_pct": 100 * (incr_outcomes_targeted - incr_outcomes_random)
                                     / (abs(incr_outcomes_random) + 1e-9),
            "incremental_revenue_targeted": incr_outcomes_targeted * value_per_outcome,
            "incremental_revenue_random": incr_outcomes_random * value_per_outcome,
        })
    return pd.DataFrame(rows)


def summarize_business_case(policy_df: pd.DataFrame, budget_of_interest: float = 0.3) -> dict:
    """One-line executive summary at a chosen budget level."""
    row = policy_df.iloc[(policy_df["budget_frac"] - budget_of_interest).abs().argsort()[:1]].iloc[0]
    return {
        "budget_frac": float(row["budget_frac"]),
        "incremental_revenue_targeted": float(row["incremental_revenue_targeted"]),
        "incremental_revenue_random": float(row["incremental_revenue_random"]),
        "lift_vs_random_pct": float(row["lift_vs_random_pct"]),
    }
