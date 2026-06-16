"""The detector: discover the minimal key that determines the decision, and decide whether the
task is a clean rule (lookup) or genuine judgment (model).

`purity` measures how deterministic the decision is given the key (the predicted ceiling). The greedy
search ranks candidate fields by *held-out* accuracy, so it rejects fields that only look predictive
by memorizing the training rows (e.g. a continuous timestamp).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

# Free-text fields are never part of a routing key.
FREE_TEXT = {"ticket_id", "customer_message", "request_id", "internal_reasoning_brief", "customer_reply"}


def candidate_fields(train: list[dict]) -> list[str]:
    fields: list[str] = []
    for ex in train:
        for k in ex["input"]:
            if k not in FREE_TEXT and k not in fields:
                fields.append(k)
    return fields


def purity(train: list[dict], fields: list[str]) -> tuple[float, int]:
    """Fraction of training rows whose decision == the plurality decision of their key-group."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for ex in train:
        groups[tuple(ex["input"].get(f) for f in fields)].append(ex["target"]["decision"])
    pure = sum(Counter(v).most_common(1)[0][1] for v in groups.values())
    return pure / len(train), len(groups)


def _holdout_acc(train: list[dict], fields: list[str], folds: int = 3) -> float:
    if not fields:
        return 0.0
    accs = []
    for seed in range(folds):
        idx = list(range(len(train)))
        random.Random(seed).shuffle(idx)
        cut = int(0.7 * len(train))
        fit, val = [train[i] for i in idx[:cut]], [train[i] for i in idx[cut:]]
        g: dict[tuple, list[str]] = defaultdict(list)
        for ex in fit:
            g[tuple(ex["input"].get(f) for f in fields)].append(ex["target"]["decision"])
        look = {k: Counter(v).most_common(1)[0][0] for k, v in g.items()}
        fallback = Counter(ex["target"]["decision"] for ex in fit).most_common(1)[0][0]
        accs.append(
            sum(look.get(tuple(ex["input"].get(f) for f in fields), fallback) == ex["target"]["decision"] for ex in val)
            / max(len(val), 1)
        )
    return float(np.mean(accs))


def discover_key(train: list[dict]) -> dict:
    """Greedily grow the minimal determining key, ranked by held-out generalization accuracy."""
    remaining = candidate_fields(train)
    selected: list[str] = []
    best = 0.0
    while remaining:
        gain, field = sorted(((_holdout_acc(train, selected + [f]), f) for f in remaining), reverse=True)[0]
        if gain <= best + 0.01:
            break
        selected.append(field)
        remaining.remove(field)
        best = gain
    p, n = purity(train, selected)
    return {"key": selected, "purity": p, "holdout_acc": best, "n_groups": n, "n_train": len(train)}


def recommend(train: list[dict], test: list[dict], disc: dict) -> dict:
    """Decide the strategy: a clean rule (lookup) needs high purity AND high test-key coverage."""
    key = disc["key"]
    train_keys = {tuple(ex["input"].get(f) for f in key) for ex in train}
    seen = sum(tuple(ex["input"].get(f) for f in key) in train_keys for ex in test)
    coverage = seen / len(test) if test else 0.0
    clean = disc["purity"] >= 0.95 and coverage >= 0.9
    return {
        "test_key_coverage": coverage,
        "clean_key": clean,
        "strategy": "lookup table" if clean else "model / judgment",
    }
