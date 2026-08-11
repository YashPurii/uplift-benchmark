"""
Feature encoding shared by the UM and ITE experiments.

The paper doesn't one-hot the full high-cardinality categoricals directly
(f6, f8, f9 alone have thousands of modalities -- infeasible at 14M rows).
Instead it projects each categorical column onto a smaller number of random
buckets, then one-hot encodes those. We reimplement that "hashing trick"
here: `n_projections` controls how many random buckets each categorical
column is hashed into before one-hot encoding, mirroring their "100
projections" (UM, Section 5.2) and "5 projections" (ITE, Section 5.3) setups.

IMPORTANT: the real CRITEO-UPLIFTv2 categorical columns are anonymized
hashed tokens -- NOT small sequential integers. A naive `value % n_buckets`
silently breaks on them (float values, or integers spread across a huge
range) and can blow up into thousands of "buckets" instead of n_buckets.
`_hash_categorical` below fixes this by first collapsing each column to
dense factorized codes, then assigning each unique code to one of
n_buckets via a random lookup table -- this is correct regardless of
whether the raw values are small ints, huge hashed ints, or floats.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse as sp
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

from . import data as data_mod


def _hash_categorical(values: np.ndarray, n_buckets: int, seed: int) -> np.ndarray:
    """Map arbitrary categorical values to n_buckets integer buckets.
    Robust to any raw value type/range (unlike `value % n_buckets`, which
    only behaves on small dense integers)."""
    codes, uniques = pd.factorize(pd.Series(values), sort=False)
    rng = np.random.default_rng(seed)
    bucket_map = rng.integers(0, n_buckets, size=max(len(uniques), 1))
    codes_safe = np.where(codes < 0, 0, codes)  # factorize gives -1 for NaN
    return bucket_map[codes_safe]


def encode(df, n_projections: int = 8, seed: int = 42, sparse: bool = True):
    """
    Design matrix: standardized continuous features concatenated with
    one-hot(hash(categorical, n_projections)).

    Returns a scipy sparse CSR matrix by default (`sparse=True`) -- required
    at real-dataset scale, since even a modest n_projections produces a
    wide, mostly-zero one-hot block that would otherwise need hundreds of
    GB dense at 14M rows. Pass sparse=False for small subsamples (e.g. the
    ITE benchmark) where downstream code needs a dense ndarray.
    """
    X_cont = StandardScaler().fit_transform(df[data_mod.CONTINUOUS_FEATURES].values).astype(np.float32)

    hashed_cols = []
    for i, f in enumerate(data_mod.CATEGORICAL_FEATURES):
        hashed_cols.append(_hash_categorical(df[f].values, n_projections, seed=seed + i))
    X_cat_hashed = np.column_stack(hashed_cols)

    fixed_categories = [list(range(n_projections))] * len(data_mod.CATEGORICAL_FEATURES)
    enc = OneHotEncoder(categories=fixed_categories, handle_unknown="ignore", sparse_output=True)
    X_cat = enc.fit_transform(X_cat_hashed)  # always sparse, shape (n, 8*n_projections) exactly

    if sparse:
        return sp.hstack([sp.csr_matrix(X_cont), X_cat], format="csr")
    return np.hstack([X_cont, X_cat.toarray()]).astype(np.float32)
