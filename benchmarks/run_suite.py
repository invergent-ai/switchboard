"""Four-domain suite (NO API key needed).

For each task: discover the key, get the detector's verdict, and measure the certified table's
decision accuracy on the held-out test set. Shows that the detector says "lookup" only where it can
deliver ~100%, and "judgment" (declines) where it can't.

    python benchmarks/run_suite.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchboard.detector import discover_key, recommend  # noqa: E402
from switchboard.table import learn_table, lookup  # noqa: E402
from switchboard.tasks import TASKS  # noqa: E402


def main():
    print(f"{'task':<13}{'verdict':<14}{'purity':>7}{'coverage':>10}{'table_acc':>11}")
    print("-" * 56)
    for name, task in TASKS.items():
        s = task.make_splits(seed=0)
        train, test = s["train"], s["test"]
        disc = discover_key(train)
        rec = recommend(train, test, disc)
        table = learn_table(train, disc["key"])
        hits = covered = 0
        for ex in test:
            v = lookup(ex["input"], table, disc["key"])
            if v is not None:
                covered += 1
                hits += v[0] == ex["target"]["decision"]
        acc = hits / covered if covered else float("nan")
        verdict = "lookup table" if rec["clean_key"] else "judgment"
        print(f"{name:<13}{verdict:<14}{disc['purity']:>7.2f}{rec['test_key_coverage']:>10.2f}{acc:>11.2f}")


if __name__ == "__main__":
    main()
