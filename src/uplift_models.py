"""
Uplift models.

Baselines reimplement two of the paper's four (Two-Model, Class Variable
Transformation -- the two that don't require a bespoke shared-representation
architecture). MOM and SDR are intentionally left out: MOM needs the doubly-
robust propensity correction to be well-behaved and SDR is a paper-specific
architecture -- including toy versions of either would misrepresent them.

`GradientBoostedUplift` is this project's own addition: the paper's
baselines are all 2018-era linear/logistic models. XGBoost as the base
learner for the two-model design is a straightforward, defensible upgrade
that the field has broadly moved toward since 2021 -- this benchmarks it
against the originals on equal footing, using the paper's own metric.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class TwoModel:
    """Independently model P(Y=1|X,T=1) and P(Y=1|X,T=0); uplift = difference."""

    def __init__(self, C: float = 1.0):
        self.model_t = LogisticRegression(max_iter=300, C=C)
        self.model_c = LogisticRegression(max_iter=300, C=C)

    def fit(self, X, treatment, y):
        self.model_t.fit(X[treatment == 1], y[treatment == 1])
        self.model_c.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X):
        return self.model_t.predict_proba(X)[:, 1] - self.model_c.predict_proba(X)[:, 1]


class ClassVariableTransformation:
    """Jaskowski & Jaroszewicz 2012. Requires a 50/50 treatment split to be
    unbiased; since this dataset is 85/15, we pass sample weights to correct
    for it (this correction is not in the original paper's description but is
    necessary for a valid estimate under treatment imbalance)."""

    def __init__(self, C: float = 1.0):
        self.clf = LogisticRegression(max_iter=300, C=C)

    def fit(self, X, treatment, y):
        z = np.where((treatment == 1) & (y == 1) | (treatment == 0) & (y == 0), 1, 0)
        p_t = treatment.mean()
        w = np.where(treatment == 1, 1 / (2 * p_t), 1 / (2 * (1 - p_t)))
        self.clf.fit(X, z, sample_weight=w)
        return self

    def predict_uplift(self, X):
        # score = P(Z=1|X); uplift = 2*score - 1
        return 2 * self.clf.predict_proba(X)[:, 1] - 1


class GradientBoostedUplift:
    """This project's extension: two-model design with XGBoost base learners
    instead of logistic regression, to test whether a modern non-linear
    learner meaningfully beats the paper's linear baselines on their own
    metric (AUUC)."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 4, learning_rate: float = 0.08):
        params = dict(n_estimators=n_estimators, max_depth=max_depth,
                      learning_rate=learning_rate, eval_metric="logloss",
                      n_jobs=-1, verbosity=0)
        self.model_t = XGBClassifier(**params)
        self.model_c = XGBClassifier(**params)

    def fit(self, X, treatment, y):
        self.model_t.fit(X[treatment == 1], y[treatment == 1])
        self.model_c.fit(X[treatment == 0], y[treatment == 0])
        return self

    def predict_uplift(self, X):
        return self.model_t.predict_proba(X)[:, 1] - self.model_c.predict_proba(X)[:, 1]


MODEL_REGISTRY = {
    "two_model": TwoModel,
    "class_transform": ClassVariableTransformation,
    "gbm_uplift": GradientBoostedUplift,
}
