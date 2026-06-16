"""Real-data probe (NO API key needed).

Six public OpenML datasets spanning genuine rules -> noisy judgment. For each: discover the key,
build the certified table from train, evaluate on a held-out split, and check whether the
*predicted* accuracy (train purity) matches the table's *actual* test accuracy.

    python benchmarks/run_realdata.py
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.datasets import fetch_openml

# sklearn warns that several datasets have multiple active versions; it deterministically returns the
# lowest active version, which is what these numbers are measured on. Silence the noise.
warnings.filterwarnings("ignore", message="Multiple active versions")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchboard.detector import discover_key, recommend  # noqa: E402

DATASETS = ["mushroom", "car", "nursery", "adult", "credit-g", "bank-marketing"]


def load(name: str, cap: int = 6000):
    d = fetch_openml(name, as_frame=True)
    df = d.frame.dropna()
    target = d.target.name if getattr(d.target, "name", None) else df.columns[-1]
    if len(df) > cap:
        df = df.sample(cap, random_state=0)
    exs = []
    for r in df.to_dict("records"):
        inp = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)) for k, v in r.items() if k != target}
        exs.append({"input": inp, "target": {"decision": str(r[target])}})
    return exs


def plurality_table(train, key):
    g = defaultdict(list)
    for ex in train:
        g[tuple(ex["input"].get(f) for f in key)].append(ex["target"]["decision"])
    return {k: Counter(v).most_common(1)[0][0] for k, v in g.items()}


def probe(name: str):
    exs = load(name)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(exs))
    cut = int(0.7 * len(exs))
    train, test = [exs[i] for i in idx[:cut]], [exs[i] for i in idx[cut:]]
    disc = discover_key(train)
    rec = recommend(train, test, disc)
    table = plurality_table(train, disc["key"])
    glob = Counter(ex["target"]["decision"] for ex in train).most_common(1)[0][0]
    cov = cov_correct = 0
    for ex in test:
        k = tuple(ex["input"].get(f) for f in disc["key"])
        if k in table:
            cov += 1
            cov_correct += table[k] == ex["target"]["decision"]
    covered_acc = cov_correct / cov if cov else float("nan")
    maj = sum(ex["target"]["decision"] == glob for ex in test) / len(test)
    verdict = "RULE" if rec["clean_key"] else "judgment"
    print(f"{name:<15}{verdict:<10} predicted={disc['purity']:.2f}  actual={covered_acc:.2f}  "
          f"(majority baseline {maj:.2f})  key={disc['key']}")


def main():
    print(f"{'dataset':<15}{'verdict':<10} predicted (purity) vs. actual table accuracy")
    print("-" * 100)
    for n in DATASETS:
        try:
            probe(n)
        except Exception as e:  # noqa: BLE001
            print(f"{n:<15} ERROR: {str(e)[:70]}")


if __name__ == "__main__":
    main()
