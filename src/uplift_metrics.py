"""
Uplift evaluation -- the "separate, relative" AUUC variant the paper adopts
(Section 5.2, citing Devriendt et al. 2020 for its robustness to treatment
imbalance), plus a bootstrap confidence interval as a simpler stand-in for
their AUUC test-set bound (Betlei et al. 2020).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def uplift_curve(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray,
                  n_bins: int = 100) -> pd.DataFrame:
    """
    Rank individuals by predicted uplift (descending), then at each top-k%
    cutoff compute the "separate, relative" group uplift:
        uplift(k) = mean(Y | T=1, top-k) - mean(Y | T=0, top-k)
    which is the ranking-quality signal AUUC integrates.
    """
    order = np.argsort(-uplift_scores)
    treatment, outcome = treatment[order], outcome[order]
    n = len(treatment)
    cutoffs = np.linspace(1, n, n_bins).astype(int)

    rows = []
    for k in cutoffs:
        t_mask = treatment[:k] == 1
        c_mask = treatment[:k] == 0
        y_t = outcome[:k][t_mask].mean() if t_mask.sum() > 0 else 0.0
        y_c = outcome[:k][c_mask].mean() if c_mask.sum() > 0 else 0.0
        rows.append({"frac": k / n, "n": k, "uplift": y_t - y_c})
    return pd.DataFrame(rows)


def auuc(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray,
          n_bins: int = 100) -> float:
    """Area under the (relative) uplift curve, trapezoidal integration over
    the population fraction axis."""
    curve = uplift_curve(uplift_scores, treatment, outcome, n_bins)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz_fn(curve["uplift"].values, curve["frac"].values))


def auuc_bootstrap_ci(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray,
                        n_boot: int = 200, alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap CI for AUUC -- serves the same role as the paper's AUUC
    test-set bound (a single-split confidence interval), simpler to implement
    and adequate for model-comparison purposes."""
    rng = np.random.default_rng(seed)
    n = len(treatment)
    point = auuc(uplift_scores, treatment, outcome)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(auuc(uplift_scores[idx], treatment[idx], outcome[idx]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auuc": point, "ci_low": float(lo), "ci_high": float(hi), "std": float(np.std(boots))}
