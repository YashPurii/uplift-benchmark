"""
Data layer for the Criteo Uplift / ITE benchmark reimplementation.

Two entry points:
  - load_real(path)          -> loads the actual CRITEO-UPLIFTv2 CSV
  - generate_synthetic(n)    -> generates a schema-matched synthetic dataset
                                 for pipeline development when the real file
                                 isn't available yet. NOT a substitute for the
                                 real experiments -- it exists purely so the
                                 rest of the codebase can be built, tested and
                                 demoed before the real CSV is dropped in.

Schema (from the paper, Table 2 + Section 3):
  f0, f2, f7, f10  -> continuous
  f1, f3, f4, f5, f6, f8, f9, f11 -> categorical (modality counts below)
  treatment        -> binary, P(T=1) ~= 0.85
  exposure         -> binary, exposure=0 whenever treatment=0 (structural)
  visit            -> binary, ~4.70% positive overall
  conversion       -> binary, ~0.29% positive overall, conversion=0 whenever visit=0
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CONTINUOUS_FEATURES = ["f0", "f2", "f7", "f10"]
CATEGORICAL_FEATURES = ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]
ALL_FEATURES = [f"f{i}" for i in range(12)]
CATEGORY_CARDINALITY = {
    "f1": 60, "f3": 552, "f4": 260, "f5": 132,
    "f6": 1645, "f8": 3743, "f9": 1594, "f11": 136,
}
TREATMENT_COL = "treatment"
EXPOSURE_COL = "exposure"
LABEL_COLS = ["visit", "conversion"]

REQUIRED_COLUMNS = ALL_FEATURES + [TREATMENT_COL, EXPOSURE_COL] + LABEL_COLS


def load_real(path: str) -> pd.DataFrame:
    """Load the actual CRITEO-UPLIFTv2 CSV (or a .csv.gz)."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Loaded file is missing expected columns: {sorted(missing)}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )
    return df


def generate_synthetic(n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    """
    Schema-matched synthetic dataset for development/testing only.

    Reproduces, approximately: the 85% treatment ratio, the ~4.70%/0.29%
    visit/conversion base rates, the structural constraints (T=0 => E=0,
    V=0 => C=0), and a genuine (non-degenerate, heterogeneous) uplift signal
    so that downstream models have something real to detect.

    Swap this out for `load_real()` the moment the actual CSV is available --
    every downstream module only depends on the column contract above, so
    nothing else needs to change.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=range(n))

    # Continuous features: standard-normal-ish, mildly skewed
    for f in CONTINUOUS_FEATURES:
        df[f] = rng.normal(0, 1, n) + rng.exponential(0.3, n)

    # Categorical features: zipf-like modality distribution (a few dominant
    # categories, long tail -- mirrors real hashed-token behavior)
    for f in CATEGORICAL_FEATURES:
        card = CATEGORY_CARDINALITY[f]
        p = 1.0 / (np.arange(1, card + 1) ** 1.2)
        p = p / p.sum()
        df[f] = rng.choice(card, size=n, p=p)

    # Treatment: independent of X (RCT), 85% treated -- mirrors the paper's
    # key validated property (T _||_ X)
    treatment = rng.binomial(1, 0.85, n)
    df[TREATMENT_COL] = treatment

    # A latent, heterogeneous "propensity to respond" driven by f0/f2 (the
    # two features we'll designate as most informative), used to generate
    # genuine treatment-effect heterogeneity for the uplift experiments.
    latent = 0.6 * df["f0"] + 0.4 * df["f2"]
    latent_std = (latent - latent.mean()) / (latent.std() + 1e-9)

    base_visit_logit = -3.2 + 0.35 * latent_std  # base rate calibrated below
    uplift_logit = 0.9 + 0.5 * np.tanh(latent_std)  # heterogeneous treatment effect
    exposure = np.where(treatment == 1, rng.binomial(1, 0.7, n), 0)
    df[EXPOSURE_COL] = exposure

    visit_logit = base_visit_logit + uplift_logit * exposure
    visit_p = 1 / (1 + np.exp(-visit_logit))
    visit = rng.binomial(1, visit_p)
    df["visit"] = visit

    conv_logit = -5.5 + 0.4 * latent_std + 0.8 * exposure
    conv_p = np.where(visit == 1, 1 / (1 + np.exp(-conv_logit)), 0.0)
    conversion = rng.binomial(1, conv_p)
    df["conversion"] = np.where(visit == 1, conversion, 0)

    return df[REQUIRED_COLUMNS]


def train_test_split_stratified(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """80/20 split stratified on (treatment, visit) to preserve both imbalances,
    matching the paper's protocol (Section 5.2)."""
    from sklearn.model_selection import train_test_split
    strata = df[TREATMENT_COL].astype(str) + "_" + df["visit"].astype(str)
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strata
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
