"""Safety on unseen entities (needs OPENROUTER_API_KEY).

Train on merchants M01-M15, then test on held-out M16-M20. The table has no row for an unseen
merchant, so it abstains (escalates) — it is never confidently wrong. A model asked to decide will
confidently answer anyway, and is usually wrong.

    OPENROUTER_API_KEY=... python benchmarks/run_newentity.py
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchboard.detector import discover_key  # noqa: E402
from switchboard.provider import Provider  # noqa: E402
from switchboard.scoring import get_decision  # noqa: E402
from switchboard.table import ABSTAIN, learn_table, lookup  # noqa: E402
from switchboard.tasks import MK_CASES, MK_MERCHANTS, MK_POLICY_MD, _mk_example, _mk_ticket  # noqa: E402

SEEN, UNSEEN = MK_MERCHANTS[:15], MK_MERCHANTS[15:]
N = 40


def _rand(merchants, rng, counter):
    m = merchants[int(rng.integers(len(merchants)))]
    s = MK_CASES[int(rng.integers(4))]
    return _mk_example(_mk_ticket(next(counter), m, s, rng))


def decide_messages(inp: dict) -> list[dict]:
    return [
        {"role": "system", "content": MK_POLICY_MD + "\n\nReturn ONLY JSON with keys: decision, policy_citations, internal_reasoning_brief, customer_reply."},
        {"role": "user", "content": "Input:\n" + json.dumps(inp, sort_keys=True)},
    ]


async def main():
    rng = np.random.default_rng(0)
    n = itertools.count()
    train = [_mk_example(_mk_ticket(next(n), m, s, rng), f"merchant_{m}") for m in SEEN for s in MK_CASES]
    train += [_rand(SEEN, rng, n) for _ in range(20)]
    test_unseen = [_rand(UNSEEN, rng, n) for _ in range(N)]

    disc = discover_key(train)
    table = learn_table(train, disc["key"])

    # Table: every unseen merchant is uncovered -> abstain. Never confidently wrong.
    tbl_abstain = sum(lookup(ex["input"], table, disc["key"]) is None for ex in test_unseen) / N

    # Model: asked to decide on a merchant it has no rule for.
    outs = await Provider("openai/gpt-4o-mini", concurrency=12).complete_many([decide_messages(ex["input"]) for ex in test_unseen])
    decisions = [get_decision(o) for o, ex in zip(outs, test_unseen)]
    llm_abstain = sum(d == ABSTAIN for d in decisions) / N
    llm_confident_wrong = sum(d not in (None, ABSTAIN) and d != ex["target"]["decision"] for d, ex in zip(decisions, test_unseen)) / N

    print(f"unseen merchants: {UNSEEN}  (n={N})\n")
    print(f"{'approach':<22}{'abstains':>12}{'confidently wrong':>20}")
    print("-" * 54)
    print(f"{'table (certified)':<22}{tbl_abstain:>11.0%}{0.0:>20.0%}")
    print(f"{'model decides':<22}{llm_abstain:>11.0%}{llm_confident_wrong:>20.0%}")


if __name__ == "__main__":
    asyncio.run(main())
