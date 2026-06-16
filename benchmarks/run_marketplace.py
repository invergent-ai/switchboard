"""Headline result: model decides vs. table decides (needs OPENROUTER_API_KEY).

The per-merchant rules are hidden from the prompt, so a model asked to decide is guessing (~20%).
The certified table answers the same covered cases deterministically — 100%, and identically
regardless of which model writes the reply (that's the point: the decision no longer depends on the
model).

    OPENROUTER_API_KEY=... python benchmarks/run_marketplace.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchboard.detector import discover_key  # noqa: E402
from switchboard.provider import Provider  # noqa: E402
from switchboard.scoring import get_decision  # noqa: E402
from switchboard.table import learn_table, lookup  # noqa: E402
from switchboard.tasks import get_task  # noqa: E402

MODELS = ["meta-llama/llama-3.1-8b-instruct", "openai/gpt-4o-mini"]
N_TEST = 60


def decide_messages(policy_md: str, inp: dict) -> list[dict]:
    return [
        {"role": "system", "content": policy_md + "\n\nReturn ONLY JSON with keys: decision, policy_citations, internal_reasoning_brief, customer_reply."},
        {"role": "user", "content": "Input:\n" + json.dumps(inp, sort_keys=True)},
    ]


async def main():
    task = get_task("marketplace")
    s = task.make_splits(seed=0)
    train, test = s["train"], s["test"][:N_TEST]

    # The table is deterministic and model-independent — compute its accuracy once, no API.
    disc = discover_key(train)
    table = learn_table(train, disc["key"])
    hits = sum((lookup(ex["input"], table, disc["key"]) or (None,))[0] == ex["target"]["decision"] for ex in test)
    table_acc = hits / len(test)
    print(f"key discovered: {disc['key']}  (train purity {disc['purity']:.2f})\n")
    print(f"{'model':<38}{'model decides':>15}{'table decides':>15}")
    print("-" * 68)

    for model in MODELS:
        outs = await Provider(model, concurrency=12).complete_many([decide_messages(task.policy_md, ex["input"]) for ex in test])
        llm_acc = sum(get_decision(o) == ex["target"]["decision"] for o, ex in zip(outs, test)) / len(test)
        print(f"{model:<38}{llm_acc:>14.0%}{table_acc:>15.0%}")


if __name__ == "__main__":
    asyncio.run(main())
