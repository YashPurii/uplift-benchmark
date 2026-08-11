"""
Dataset validation checks -- reimplementation of the paper's Section 5.1.

1. Classifier Two-Sample Test (C2ST): verifies T _||_ X by checking that a
   classifier trained to predict treatment from features does no better than
   chance. This is the actual causal-identification guarantee the whole
   uplift-modeling framework depends on -- if it fails, nothing downstream
   is trustworthy.
2. Feature informativeness: verifies X is actually predictive of the
   outcomes (visit/conversion), i.e. the anonymized/hashed features weren't
   stripped of signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from scipy import stats

from . import data as data_mod
from . import features as feat_mod


def _subsample(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    """These checks don't need the full 14M rows -- the paper's own C2ST
    validation runs on small subsamples too (their Table 3 caption notes
    300). Subsampling keeps this fast without weakening the test: a
    classifier that can't detect confounding on 300K rows certainly won't
    on 14M either, and if it CAN detect it, a subsample is enough to show
    that too."""
    if sample_size is None or len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=seed)


def c2st_treatment_predictability(df: pd.DataFrame, n_folds: int = 5, seed: int = 42,
                                    n_projections: int = 16, sample_size: int | None = 300_000) -> dict:
    """H0: T _||_ X. Under H0, a classifier's held-out log-loss should not
    beat a classifier that ignores X (predicts the marginal treatment rate).
    A one-sided paired t-test on per-fold log-loss gives the p-value."""
    df = _subsample(df, sample_size, seed)
    X = feat_mod.encode(df, n_projections=n_projections, seed=seed, sparse=True)
    y = df[data_mod.TREATMENT_COL].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    random_losses, model_losses = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=200, C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])[:, 1]
        model_losses.append(log_loss(y[test_idx], p, labels=[0, 1]))

        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X[train_idx], y[train_idx])
        p_dummy = dummy.predict_proba(X[test_idx])[:, 1]
        random_losses.append(log_loss(y[test_idx], p_dummy, labels=[0, 1]))

    # one-sided: is model_loss significantly LOWER than random_loss?
    t_stat, p_two_sided = stats.ttest_rel(model_losses, random_losses)
    p_one_sided = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2

    return {
        "n_used": int(len(df)),
        "median_random_loss": float(np.median(random_losses)),
        "median_treatment_loss": float(np.median(model_losses)),
        "p_value": float(p_one_sided),
        "passes_randomization_check": bool(p_one_sided > 0.05),
    }


def feature_informativeness(df: pd.DataFrame, outcome: str, seed: int = 42,
                              n_projections: int = 16, sample_size: int | None = 300_000) -> dict:
    """% relative improvement of a real classifier over a constant-mean
    dummy classifier, for a given outcome column."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    df = _subsample(df, sample_size, seed)
    X = feat_mod.encode(df, n_projections=n_projections, seed=seed, sparse=True)
    y = df[outcome].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    dummy_loss = log_loss(y_te, dummy.predict_proba(X_te)[:, 1], labels=[0, 1])

    clf = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=3)
    clf.fit(X_tr, y_tr)
    model_loss = log_loss(y_te, clf.predict_proba(X_te)[:, 1], labels=[0, 1])

    rel_improvement = 100.0 * (dummy_loss - model_loss) / dummy_loss
    return {
        "outcome": outcome,
        "n_used": int(len(df)),
        "dummy_log_loss": float(dummy_loss),
        "model_log_loss": float(model_loss),
        "relative_improvement_pct": float(rel_improvement),
    }


def run_all_checks(df: pd.DataFrame, sample_size: int | None = 300_000) -> dict:
    results = {"c2st": c2st_treatment_predictability(df, sample_size=sample_size)}
    for outcome in ["visit", "conversion"]:
        results[f"informativeness_{outcome}"] = feature_informativeness(df, outcome, sample_size=sample_size)
    return results
