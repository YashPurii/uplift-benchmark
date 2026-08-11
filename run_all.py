"""
Runs the full pipeline end-to-end: EDA -> sanity checks -> model
separability -> ITE benchmark -> business case.

Defaults to synthetic data (safe to run immediately, no dataset needed).
Pass --data path/to/criteo-uplift-v2.csv to run on the real dataset.

    python run_all.py                          # dev mode, synthetic data
    python run_all.py --data data/criteo.csv    # real data
"""
import argparse
import subprocess
import sys

STEPS = [
    ("experiments/00_eda.py", {"--n": "150000"}),
    ("experiments/01_sanity_checks.py", {"--n": "100000"}),
    ("experiments/02_model_separability.py", {"--n": "200000"}),
    ("experiments/03_ite_benchmark.py", {"--n": "20000", "--realizations": "3"}),
    ("experiments/04_business_case.py", {"--n": "150000"}),
]
# Note: these defaults are sized to finish in a few minutes on a laptop CPU
# for the dev-mode smoke test. When running with --data on the real ~14M-row
# file, increase --n per step (or edit the calls directly) -- runtime will
# scale accordingly; the ITE benchmark in particular (RandomForest x 3
# models x 3 surfaces x N realizations) is the slowest step.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="Path to real CRITEO-UPLIFTv2 CSV")
    args = ap.parse_args()

    for script, extra_args in STEPS:
        cmd = [sys.executable, script]
        if args.data:
            cmd += ["--data", args.data]
        else:
            for k, v in extra_args.items():
                cmd += [k, v]
        print(f"\n{'=' * 70}\nRunning: {' '.join(cmd)}\n{'=' * 70}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED at {script} -- stopping.")
            sys.exit(1)

    print("\nAll steps complete. See outputs/figures/ and outputs/tables/.")


if __name__ == "__main__":
    main()
