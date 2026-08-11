"""
Semi-synthetic response surfaces for ITE evaluation -- reimplements Section 4
of the paper (CRITEO-ITE). Ground-truth CATE is known here by construction,
which is what makes sqrt(PEHE) computable (impossible on the real-label data
because of the fundamental problem of causal inference).
"""
from __future__ import annotations
import numpy as np


def _standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)


def case_a(X: np.ndarray, seed: int = 42) -> dict:
    """Hill (2011) Case A: constant treatment effect, two linear surfaces."""
    rng = np.random.default_rng(seed)
    beta = rng.multinomial(1, [1 / 5] * 5, size=X.shape[1]).dot(np.array([0, 1, 2, 3, 4]))
    mu0 = X @ beta
    mu1 = mu0 + 4.0
    return {"mu0": mu0, "mu1": mu1, "tau": mu1 - mu0}


def case_b(X: np.ndarray, seed: int = 42) -> dict:
    """Hill (2011) Case B: exponential control surface, linear treatment
    surface, offset calibrated so ATT ~= 4 (mirrors paper's setup)."""
    rng = np.random.default_rng(seed)
    beta = rng.multinomial(1, [1 / 5] * 5, size=X.shape[1]).dot(np.array([0, 1, 2, 3, 4]))
    W = rng.normal(0, 0.2, X.shape[1])
    mu0 = np.exp((X + W) @ beta * 0.05)  # scaled to keep exp() numerically sane
    mu1_raw = X @ beta
    omega = mu1_raw.mean() - mu0.mean() - 4.0
    mu1 = mu1_raw - omega
    return {"mu0": mu0, "mu1": mu1, "tau": mu1 - mu0}


def multi_peaked(X: np.ndarray, n_anchors: int = 5, seed: int = 42) -> dict:
    """This paper's own novel contribution (Eq. 6): a sum of radial-basis
    bumps centered on randomly chosen anchor points, motivated by the
    non-monotonic uplift-vs-PCA pattern they observed in the real data
    (Figure 3). Weights are fit so ATE ~= 4, matching the other two surfaces
    for a fair three-way comparison."""
    rng = np.random.default_rng(seed)
    Xs = _standardize(X)
    n, d = Xs.shape
    anchor_idx = rng.choice(n, size=n_anchors, replace=False)
    anchors = Xs[anchor_idx]
    sigma = 1.0

    w0 = rng.uniform(0, 1, n_anchors)
    w1 = rng.uniform(0, 1, n_anchors)

    def surface(weights):
        out = np.zeros(n)
        for c in range(n_anchors):
            dist2 = ((Xs - anchors[c]) ** 2).sum(axis=1)
            out += weights[c] * np.exp(-dist2 / (2 * sigma ** 2))
        return out

    mu0_raw, mu1_raw = surface(w0), surface(w1)
    tau_raw = mu1_raw - mu0_raw
    # shift (additively, not multiplicatively -- tau_raw can be near-zero on
    # average even when it's genuinely heterogeneous, so a multiplicative
    # rescale is numerically unstable) so that ATE == 4, preserving shape
    shift = 4.0 - tau_raw.mean()
    mu1 = mu1_raw + shift
    return {"mu0": mu0_raw, "mu1": mu1, "tau": mu1 - mu0_raw}


SURFACES = {"case_a": case_a, "case_b": case_b, "multi_peaked": multi_peaked}


def confounded_treatment_assignment(X: np.ndarray, important_col: int, delta: float = 0.01,
                                      seed: int = 42) -> np.ndarray:
    """p(x) = (1-2*delta)*sigmoid(alpha^T x) + delta, alpha sparse with a 1 on
    the most-important feature. Guarantees strong ignorability (p in
    [delta, 1-delta]) and confines all confounding to X (Eq. 7)."""
    rng = np.random.default_rng(seed)
    Xs = _standardize(X)
    z = Xs[:, important_col]
    p = (1 - 2 * delta) * (1 / (1 + np.exp(-z))) + delta
    treatment = rng.binomial(1, p)
    return treatment


def generate_ite_dataset(X: np.ndarray, surface_name: str, important_col: int = 0,
                           delta: float = 0.01, seed: int = 42):
    """Full pipeline: pick a surface, generate mu0/mu1/tau, confounded T,
    factual Y. Returns (treatment, y_factual, tau_true, mu0, mu1)."""
    rng = np.random.default_rng(seed)
    surf = SURFACES[surface_name](X, seed=seed)
    mu0, mu1, tau = surf["mu0"], surf["mu1"], surf["tau"]
    treatment = confounded_treatment_assignment(X, important_col, delta, seed=seed)
    noise = rng.normal(0, 0.5, len(X))
    y_factual = np.where(treatment == 1, mu1, mu0) + noise
    return treatment, y_factual, tau, mu0, mu1
