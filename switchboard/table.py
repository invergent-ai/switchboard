"""The certified lookup table + coverage-aware routing.

The table is the plurality decision per key, learned from training labels. At inference, a covered
key returns its verified decision; an uncovered key returns nothing (the caller abstains).
"""

from __future__ import annotations

from collections import Counter, defaultdict

ABSTAIN = "request_more_info"  # the safe "I have no rule for this — escalate" answer


def learn_table(train: list[dict], key: list[str]) -> dict[tuple, tuple[str, list[str]]]:
    groups: dict[tuple, list[tuple[str, tuple]]] = defaultdict(list)
    for ex in train:
        k = tuple(ex["input"].get(f) for f in key)
        groups[k].append((ex["target"]["decision"], tuple(ex["target"].get("policy_citations", []))))
    table: dict[tuple, tuple[str, list[str]]] = {}
    for k, vals in groups.items():
        decision = Counter(d for d, _ in vals).most_common(1)[0][0]
        cites = next((list(c) for d, c in vals if d == decision), [])
        table[k] = (decision, cites)
    return table


def lookup(x: dict, table: dict, key: list[str]):
    """Return (decision, citations) for a covered key, or None to defer/abstain."""
    return table.get(tuple(x.get(f) for f in key))
