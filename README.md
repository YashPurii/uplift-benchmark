# Causal Targeting at Scale: Uplift Modeling & ITE Estimation on the Criteo Benchmark

A from-scratch reimplementation of Diemert et al. (2021), *"A Large Scale
Benchmark for Individual Treatment Effect Prediction and Uplift Modeling"*
(arXiv:2111.10106), extended with a budget-constrained business simulation
layer and a modern gradient-boosted baseline the original paper doesn't test.

## Why this project

Uplift modeling answers one question: **given a limited budget to intervene
on people (ads, offers, sales calls, HCP details), who should you target to
maximize the *incremental* effect** — not who's most likely to convert
regardless of treatment, but who converts *because* you treated them. That's
a direct generalization of HCP targeting and digital engagement prioritization
work in pharma commercial analytics: same "who gets the limited budget"
decision, same causal-inference pitfalls (you never observe the counterfactual
for any one person).

**Hypotheses pre-registered before building:**
1. At the small sample sizes of legacy causal-inference benchmarks (IHDP
   ~1K, Jobs ~5K), no uplift model should be statistically distinguishable
   from another — differences only emerge at scale. *(Confirmed on synthetic
   dry-run data; paper's own headline finding, Fig. 4.)*
2. A modern gradient-boosted two-model design should be at least competitive
   with, and probably beat, the paper's 2018-era logistic-regression
   baselines on their own AUUC metric.
3. Framing results as AUUC alone understates the point for a business
   audience — converting to a budget-constrained revenue simulation should
   make the case more legible without changing the underlying math.

## What's reproduced vs. original

| Component | Status |
|---|---|
| C2ST randomization check, feature informativeness check (Section 5.1) | Reproduced |
| Two-Model, Class Variable Transformation baselines | Reproduced |
| Modified Outcome Method, Shared Data Representation baselines | **Deliberately omitted** — MOM needs a well-behaved doubly-robust propensity correction and SDR is a bespoke paper-specific architecture; toy versions of either would misrepresent them more than they'd add. |
| Model-separability-vs-sample-size experiment (Fig. 4) | Reproduced |
| Case A / Case B synthetic response surfaces (Hill 2011) | Reproduced |
| Multi-peaked RBF response surface (paper's own novel contribution, Eq. 6) | Reproduced |
| T-Learner, X-Learner, R-Learner | Reproduced |
| DR-Learner, TARNet, CFRNet (deep nets) | **Omitted** — a from-scratch TF implementation is real engineering weight for 2 of 8 benchmark cells; the meta-learners already cover the paper's strongest performers in spirit. Documented, not silently dropped. |
| **Gradient-boosted uplift model (XGBoost two-model)** | **New** — benchmarked head-to-head against the paper's linear baselines on AUUC |
| **Budget-constrained business simulation** | **New** — converts uplift rankings into $ incremental revenue at a given targeting budget vs. random targeting; the decision layer the paper's metrics stop short of |

## Architecture

```
src/
  data.py               schema + real-CSV loader + synthetic generator (dev/test only)
  features.py           continuous standardization + categorical hash/one-hot encoding
  sanity_checks.py       C2ST randomization test, feature informativeness
  uplift_metrics.py      uplift curve, AUUC, bootstrap CI
  uplift_models.py       TwoModel, ClassVariableTransformation, GradientBoostedUplift
  response_surfaces.py   Case A, Case B, multi-peaked RBF, confounded treatment assignment
  ite_models.py          T-/X-/R-Learner, sqrt(PEHE)
  business_sim.py        budget-constrained targeting policy simulator

experiments/
  00_eda.py                   headline stats + PCA-binned uplift plot (Fig. 3 analog)
  01_sanity_checks.py         Section 5.1 reproduction
  02_model_separability.py    Section 5.2 / Fig. 4 reproduction + GBM comparison
  03_ite_benchmark.py         Section 5.3 / Table 5 reproduction
  04_business_case.py         original — targeting budget simulation

run_all.py               orchestrates all five steps in sequence
```

Every module only depends on the column contract in `src/data.py`
(`f0..f11`, `treatment`, `exposure`, `visit`, `conversion`) — swapping the
synthetic generator for the real CSV requires no other code changes.

## Notes / known-fixed issues

- **v1.1 fix (categorical encoding):** the initial `src/features.py` hashed
  categorical columns via `value % n_buckets`, which silently breaks on the
  real dataset -- the anonymized `f1..f11` columns are hashed tokens, not
  small sequential integers (and can load as float64 if the CSV has any
  missing values). On the real 14M-row file this produced 8,000+ spurious
  one-hot columns instead of the intended ~128, causing an out-of-memory
  crash. Fixed by factorizing each column to dense codes first, then
  randomly assigning codes to buckets (`_hash_categorical` in
  `features.py`) -- correct regardless of the raw values' type or range.
  Encoding also now returns a sparse matrix by default, which is the real
  fix for scaling to 14M rows (a dense one-hot block at that scale needs
  hundreds of GB even with a *correct* bucket count).
- `01_sanity_checks.py` now subsamples to 300K rows by default (`--sample`,
  0 = use everything) -- matches the paper's own C2ST protocol, which
  validates on small subsamples rather than the full file, and keeps
  runtime sane.

## Running it

```bash
pip install -r requirements.txt

# Dev mode: runs immediately on schema-matched synthetic data, no download needed
python run_all.py

# Real data: download CRITEO-UPLIFTv2 from https://ailab.criteo.com/criteo-uplift-prediction-dataset/
# and place the CSV at data/criteo-uplift-v2.csv, then:
python run_all.py --data data/criteo-uplift-v2.csv
```

Each experiment can also be run individually with its own `--data`/`--n`
flags — see each file's docstring. Outputs (figures + tables) land in
`outputs/figures/` and `outputs/tables/`.

**Note on synthetic dev data:** `src/data.generate_synthetic()` exists
*only* to make the pipeline runnable and testable before the real ~14M-row
file is downloaded. It matches the paper's marginal statistics (85%
treatment ratio, ~4.7% visit rate) but its correlation structure is
invented — none of the numbers it produces are findings. Every table/figure
this repo generates in dev mode is a **pipeline smoke test**, not a result.
Real results only exist once you run with `--data`.

## The HCP-targeting parallel (for the write-up)

| Criteo variable | Pharma commercial analogue |
|---|---|
| User (browser cookie) | HCP (healthcare provider) |
| `treatment` (ad-eligible) | Rep call plan / detail-eligible |
| `exposure` (ad actually seen) | Detail actually delivered (compliance gap) |
| `visit` | Website/portal engagement |
| `conversion` | Prescription lift |
| Budget-constrained targeting | Fixed rep panel size / call capacity |

The mechanics are identical: randomized (or quasi-randomized) treatment,
binary intermediate and terminal outcomes, a hard budget constraint, and the
same fundamental-problem-of-causal-inference limitation that you never see
both potential outcomes for the same person.

## Scope decisions and limitations (for the write-up)

- Hyperparameters are fixed rather than grid-searched per the paper's 5-fold
  CV protocol, to keep realization counts and runtimes tractable outside a
  46-CPU research cluster (the paper's own compute budget, per their
  Appendix B.1). This is a defensible tradeoff for a portfolio benchmark,
  not for a production model.
- The categorical hashing trick (random-bucket + one-hot) trades some
  signal for tractability at high cardinality (`f6`, `f8`, `f9` have
  1,600–3,700 modalities). Increase `n_projections` in `src/features.py`
  if compute allows.
- AUUC confidence intervals use a bootstrap rather than the paper's AUUC
  test-set bound (Betlei et al. 2020) — simpler, asymptotically similar
  purpose, easier to defend in an interview than citing a bound you didn't
  implement.

## Suggested resume framing (XYZ format)

- *Reimplemented a peer-reviewed causal inference benchmark (14M-row Criteo
  uplift dataset) from an academic paper, building uplift modeling and ITE
  estimation pipelines (Two-Model, X-/R-Learner, custom XGBoost uplift
  model) validated via classifier two-sample tests, resulting in a
  reproducible open-source benchmark extending the original methodology
  with a budget-constrained revenue simulation layer.*
- *Extended a 2021 causal ML benchmark paper with a gradient-boosted uplift
  model and a targeting-budget business simulation, translating AUUC model
  comparisons into incremental-revenue-at-a-given-budget terms — the
  decision framing marketing/commercial teams actually need.*

Adjust once you have real numbers from the actual dataset — don't quote the
synthetic dry-run figures anywhere resume-facing.
