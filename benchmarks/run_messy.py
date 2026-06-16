"""Messy free-text inputs (needs OPENROUTER_API_KEY).

Real inputs aren't tidy field dictionaries — they're emails. We render each case as a free-text
message, then compare two pipelines:
  - end-to-end : the model reads the email and decides directly (rules hidden -> guesses).
  - extract->lookup : the model only pulls out the merchant id and item state; the table decides.

The split of labor is the whole point: the model does language, the table does the rule.

    OPENROUTER_API_KEY=... python benchmarks/run_messy.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchboard.detector import discover_key  # noqa: E402
from switchboard.provider import Provider  # noqa: E402
from switchboard.scoring import get_decision, parse_json  # noqa: E402
from switchboard.table import learn_table  # noqa: E402
from switchboard.tasks import MK_POLICY_MD, get_task  # noqa: E402

MODELS = ["meta-llama/llama-3.1-8b-instruct", "openai/gpt-4o-mini"]
N_TEST = 60


def email(inp: dict) -> str:
    state = inp["item_state"].replace("_", " ")
    return (
        f"Subject: about my order\n\nHi there,\n\nI ordered from store {inp['merchant_id']} a little while back "
        f"and the item turned up {state}. It's been roughly {inp['order_days_ago']} days now. "
        f"What can you do for me here?\n\nThanks,\nA customer"
    )


def e2e_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": MK_POLICY_MD + "\n\nReturn ONLY JSON with keys: decision, policy_citations, internal_reasoning_brief, customer_reply."},
        {"role": "user", "content": text},
    ]


def extract_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": 'Extract two fields from the message. Return ONLY JSON: {"merchant_id": "...", "item_state": one of defective|unused|final_sale|opened}.'},
        {"role": "user", "content": text},
    ]


async def main():
    task = get_task("marketplace")
    s = task.make_splits(seed=0)
    train, test = s["train"], s["test"][:N_TEST]
    disc = discover_key(train)
    table = learn_table(train, disc["key"])
    golds = [ex["target"]["decision"] for ex in test]
    emails = [email(ex["input"]) for ex in test]

    print(f"{'model':<38}{'end-to-end':>13}{'extract->lookup':>18}")
    print("-" * 69)
    for model in MODELS:
        prov = Provider(model, concurrency=12)
        e2e_out, ext_out = await asyncio.gather(
            prov.complete_many([e2e_messages(t) for t in emails]),
            prov.complete_many([extract_messages(t) for t in emails], max_tokens=80),
        )
        e2e_acc = sum(get_decision(o) == g for o, g in zip(e2e_out, golds)) / len(test)

        ext_hits = 0
        for o, g in zip(ext_out, golds):
            p = parse_json(o) or {}
            v = table.get((p.get("merchant_id"), p.get("item_state")))
            ext_hits += v is not None and v[0] == g
        ext_acc = ext_hits / len(test)
        print(f"{model:<38}{e2e_acc:>12.0%}{ext_acc:>18.0%}")


if __name__ == "__main__":
    asyncio.run(main())
