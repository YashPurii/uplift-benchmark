"""
ITE meta-learners -- T-learner, X-learner, R-learner (Kunzel et al. 2019 /
Nie & Wager 2021), matching three of the paper's five baselines. Random
Forest is used as the base regressor throughout, as in the paper's own
benchmark (Section 5.3). DR-learner is omitted for scope; TARNet/CFRNet
(the deep baselines) are omitted since a from-scratch TF implementation
adds real engineering weight for two of eight benchmark cells -- the ROI
is low relative to the meta-learners here, which already cover the
paper's strongest performers (X-Learner, DR-Learner) in spirit.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def _rf(seed: int, **kw):
    # n_estimators/max_depth kept modest by design -- this is a portfolio
    # benchmark meant to run in minutes on a laptop, not a production model.
    # See README "Scope decisions" for the full tradeoff note.
    return RandomForestRegressor(n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1, **kw)


class TLearner:
    def __init__(self, seed: int = 42):
        self.m0, self.m1 = _rf(seed), _rf(seed)

    def fit(self, X, treatment, y):
        self.m0.fit(X[treatment == 0], y[treatment == 0])
        self.m1.fit(X[treatment == 1], y[treatment == 1])
        return self

    def predict_ite(self, X):
        return self.m1.predict(X) - self.m0.predict(X)


class XLearner:
    """Kunzel et al. 2019. Stage 1: T-learner. Stage 2: impute individual
    treatment effects for each arm using the other arm's model, then fit
    effect models on those imputed effects. Stage 3: propensity-weighted
    blend of the two effect models."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.m0, self.m1 = _rf(seed), _rf(seed)
        self.tau0, self.tau1 = _rf(seed), _rf(seed)
        self.g = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1)

    def fit(self, X, treatment, y):
        self.m0.fit(X[treatment == 0], y[treatment == 0])
        self.m1.fit(X[treatment == 1], y[treatment == 1])

        d1 = y[treatment == 1] - self.m0.predict(X[treatment == 1])
        d0 = self.m1.predict(X[treatment == 0]) - y[treatment == 0]
        self.tau1.fit(X[treatment == 1], d1)
        self.tau0.fit(X[treatment == 0], d0)

        self.g.fit(X, treatment)
        return self

    def predict_ite(self, X):
        g = self.g.predict_proba(X)[:, 1]
        return g * self.tau0.predict(X) + (1 - g) * self.tau1.predict(X)


class RLearner:
    """Nie & Wager 2021 (quasi-oracle / R-learner). Residual-on-residual
    regression: regress the outcome residual (y - m(x)) on the treatment
    residual (t - e(x)), weighted by the squared treatment residual."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.m = _rf(seed)  # E[Y|X]
        self.e = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=seed, n_jobs=-1)  # P(T=1|X)
        self.tau_model = _rf(seed)

    def fit(self, X, treatment, y):
        self.m.fit(X, y)
        self.e.fit(X, treatment)

        y_res = y - self.m.predict(X)
        t_res = treatment - self.e.predict_proba(X)[:, 1]
        t_res_safe = np.where(np.abs(t_res) < 1e-3, np.sign(t_res + 1e-6) * 1e-3, t_res)

        pseudo_outcome = y_res / t_res_safe
        weights = t_res ** 2
        self.tau_model.fit(X, pseudo_outcome, sample_weight=weights)
        return self

    def predict_ite(self, X):
        return self.tau_model.predict(X)


ITE_MODEL_REGISTRY = {"t_learner": TLearner, "x_learner": XLearner, "r_learner": RLearner}


def sqrt_pehe(tau_true: np.ndarray, tau_pred: np.ndarray) -> float:
    """Precision in Estimation of Heterogeneous Effects, sqrt'd to match the
    paper's reported metric (Table 5)."""
    return float(np.sqrt(np.mean((tau_true - tau_pred) ** 2)))
