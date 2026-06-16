# Switchboard

**Don't let your LLM make the decision. Pull out the rule, and let the model just read and write around it.**

This is the code and the benchmarks behind the article [*Don't Let Your LLM Make the Decision*](https://x.com/flaviusburca/status/2066744650736529558?s=20).

The idea in three sentences: a lot of the "decisions" we hand to LLMs are really rules — the answer is fully determined by a few fields in the input. So instead of asking the model to apply the rule (where it guesses, and is sometimes confidently wrong), Switchboard learns the rule into a plain lookup table and lets the *table* decide. The model is left with the two jobs it's good at: reading a messy input into clean fields, and writing the reply once the decision is already made.

It also tells you, up front, **which of your tasks are actually rules** (so it can guarantee them) and which are genuine judgment calls (so it steps aside instead of pretending).

---

## What's in here

```
switchboard/            the method — small and self-contained
  detector.py           find the minimal set of fields that determines the answer + a verdict
  table.py              the learned lookup table + "answer or send to a human" routing
  tasks.py              four synthetic decision tasks (the rules are hidden from the model)
  provider.py           a tiny async client for OpenRouter / any OpenAI-compatible API
  scoring.py            parse the model's JSON, pull out its decision
benchmarks/
  run_realdata.py       six public datasets — NO API key needed
  run_suite.py          four domains, the detector's verdict + table accuracy — NO API key needed
  run_marketplace.py    model-decides vs. table-decides, on two models (needs an API key)
  run_newentity.py      safety on never-before-seen entities (needs an API key)
  run_messy.py          free-text email input: decide-directly vs. extract-then-look-up (needs an API key)
ARTICLE.md              the full write-up
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+.

## Run the benchmarks that need no API key

These two are pure local computation on public data — the cleanest things to reproduce first.

```bash
python benchmarks/run_realdata.py    # six OpenML datasets: predicted vs. actual accuracy
python benchmarks/run_suite.py       # four domains: the detector's verdict + the table's accuracy
```

`run_realdata.py` downloads small datasets from OpenML the first time you run it. You should see the
*predicted* accuracy (estimated from the training labels alone) land usually within a couple of points
of the *actual* held-out accuracy — and a clean **100%** only on the datasets that are genuinely rules
(mushroom), with the judgment tasks (loans, income) landing lower and flagged as judgment.

`run_suite.py` should report a "lookup table" verdict and 100% on the marketplace and access-control
tasks, a near-perfect rule on moderation (purity ~0.99), and a "judgment" verdict on triage — where
purity is only ~0.69, so it declines to certify a table at all.

## Run the benchmarks that compare against a live LLM

These call an LLM through [OpenRouter](https://openrouter.ai). **The key is read from an environment
variable only — never put it in a file.**

```bash
export OPENROUTER_API_KEY=sk-or-...        # your own key

python benchmarks/run_marketplace.py       # ~20% (model decides) vs. 100% (table decides), on two models
python benchmarks/run_newentity.py         # table abstains 100% / 0% confidently wrong; the model guesses
python benchmarks/run_messy.py             # email input: decide-directly vs. extract-then-look-up
```

The models used are small and cheap (`meta-llama/llama-3.1-8b-instruct`, `openai/gpt-4o-mini`); each
script runs a few dozen calls. Edit the `MODELS` list at the top of a script to try others.

## What you should see

The numbers from these scripts are the ones in the [article](ARTICLE.md#the-benchmarks-in-one-place). In short:

- **Marketplace** (80 hidden per-merchant rules): the model deciding on its own scores ~20%; the table scores **100%** — and the same 100% on a cheap model as on a capable one. Once the table decides, model size stops mattering for correctness.
- **Four domains**: ~100% where the task is a real rule (commerce, access control), ~96% on moderation, and a *declined table* on triage — the system only certifies what it can.
- **New entities**: on a merchant it has never seen, the table says "send this to a human" 100% of the time and is confidently wrong 0% of the time; the model alone invents an answer and is wrong ~95% of the time.
- **Messy email input**: deciding directly off the email scores 13–17%; extracting the fields first and looking them up scores 90–98%.
- **Real data**: predicted ≈ actual on all six datasets, but the clean 100% only appears where the task is genuinely a rule.

## How it works

1. **Discover the key** (`detector.py`). Greedily find the smallest set of input fields that determines the answer, ranked by *held-out* accuracy — so it rejects fields that only look predictive by memorizing the training rows (like a timestamp).
2. **Decide rule vs. judgment** (`detector.py`). If the key determines the answer cleanly on training data *and* the test cases reuse keys it has seen, it's a rule — build the table. Otherwise it's judgment — step aside.
3. **Build the table** (`table.py`). The verified answer for each key, learned from the labels.
4. **Route each input** (`table.py`). If the input's key is in the table, return the verified answer. If not, abstain — hand it to a human. That's why it is never confidently wrong on something it hasn't seen.

The LLM is only ever used to turn a messy input into the key fields, and to write the final reply — never to make the decision.

## The honest catch

The tasks where this is *perfect* are perfect because they're explicit rulebooks — but if a task is an explicit rulebook, you usually already have the rulebook. And the tasks where you'd want to *learn* the rules from data are messy precisely because they're judgment calls, where there often isn't a clean rule to find. The [article](ARTICLE.md#the-tension-nobody-tells-you-about) walks through that tension honestly. This repo lets you check it yourself.

## License

[MIT](LICENSE).
