"""
Experiment 1 -- Dataset validation (paper Section 5.1, Tables 3-4).

Usage:
    python experiments/01_sanity_checks.py [--data path/to/real.csv] [--n 200000] [--sample 300000]
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import data as data_mod
from src import sanity_checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="Path to real CRITEO-UPLIFTv2 CSV")
    ap.add_argument("--n", type=int, default=200_000, help="Synthetic sample size if --data omitted")
    ap.add_argument("--sample", type=int, default=300_000,
                     help="Rows to subsample for the checks themselves (0 = use all rows, slow at 14M)")
    ap.add_argument("--out", type=str, default="outputs/tables/sanity_checks.json")
    args = ap.parse_args()

    if args.data:
        print(f"Loading real dataset from {args.data} ...")
        df = data_mod.load_real(args.data)
    else:
        print(f"[DEV MODE] No --data given -- generating {args.n} synthetic rows for pipeline testing.")
        df = data_mod.generate_synthetic(n=args.n)

    sample_size = None if args.sample == 0 else args.sample
    print(f"Loaded {len(df):,} rows. Running C2ST + informativeness checks "
          f"(subsampled to {sample_size or 'all'} rows)...")
    results = sanity_checks.run_all_checks(df, sample_size=sample_size)

    print(json.dumps(results, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")

    c2st = results["c2st"]
    verdict = "PASSES" if c2st["passes_randomization_check"] else "FAILS"
    print(f"\nRandomization check: {verdict} (p={c2st['p_value']:.4f}) "
          f"-- {'safe to treat T as independent of X' if c2st['passes_randomization_check'] else 'DO NOT proceed -- hidden confounding likely'}")


if __name__ == "__main__":
    main()
