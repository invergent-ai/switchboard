# Don't Let Your LLM Make the Decision

*A verified lookup table beat a frozen LLM on the decisions that mattered — and hit 100% even on a tiny, cheap model. Here's how it works, and the honest catch I ran into when I tested it on real data.*

---

There's a quiet failure mode in a lot of "AI" projects.

You point a large language model at a decision that matters — who gets access, what price applies, which policy governs this case, where to route this ticket — and most of the time it's fine. Then every so often it hands you a wrong answer with total confidence. You can't tell why it landed where it did. And when a case it has never seen walks in the door, it doesn't hesitate either; it just makes something up.

For a demo, that's tolerable. For anything business-critical, it's a non-starter. The usual fix — reach for a bigger, more expensive model — helps less than you'd hope. A top-tier model is confidently wrong too. It's just pricier about it.

I spent a while trying to make a frozen, API-only LLM genuinely *reliable* on this kind of task. What finally worked was counterintuitive, and once you see it you can't unsee it:

**The reliable way to use an LLM for a critical decision is to not use it for the decision.**

## Most "decisions" are secretly rules

Here's the observation that started it.

A lot of the decisions we hand to LLMs aren't really judgment calls. They're rules. The answer is fully determined by a few fields in the input — the merchant and the item's condition, the user's role and the resource they're touching, the plan and the usage tier. Someone, somewhere, wrote that rule down, or could have.

If the answer is a rule, then making the model "smarter" is solving the wrong problem. You don't need a better guesser. You need to stop guessing.

So instead of cramming the rules into a prompt and hoping the model applies them, I pull the rule out into a plain lookup table — input maps to answer — and let the *table* make the decision. The LLM is left with the two jobs it's actually good at: reading a messy input (an email, a chat message) into clean fields, and writing a fluent reply once the decision is already made.

The decision itself never passes through the model. That one move changes everything.

## What that buys you

I built a deliberately hard test: an online marketplace where 20 merchants each set their own return policy across four item conditions. That's 80 separate rules, and none of them appear in the model's instructions. For each support ticket the system has to choose one of refund, replace, deny, escalate, or ask for more information. If you don't know that merchant's specific rule, you're guessing — and guessing scores about 25%.

The results:

- The LLM on its own, deciding straight from the input: about **20%** correct.
- The table-backed system: **100%** correct.

The part that surprised me: the table-backed system hits 100% on a tiny, cheap model just as easily as on a big expensive one. Once the table makes the decision, the model's raw ability stops mattering for correctness. You can run the cheapest model you like.

That already flips the usual cost equation. But the result I actually care about is the next one.

## Knowing what it doesn't know

The scariest failure for a critical system isn't being wrong. It's being *confidently* wrong — handing back a definite answer on something it has no business being sure about.

So I tested exactly that. I hid several merchants from the system entirely — they never appeared in its examples — and then sent it tickets for those brand-new merchants.

- The table-backed system looked for a rule, didn't find one, and said: *"I have no rule for this — send it to a human."* It was confidently wrong **0%** of the time.
- The LLM on its own never hesitated. It invented an answer every time, and was confidently wrong about **95%** of the time.

And this is the key part: a bigger model did *not* fix it. The expensive model guessed just as readily, and just as wrongly, on the cases it hadn't seen. The safety comes from the design — the system only answers what it has a verified rule for, and escalates the rest — not from model size. You cannot buy your way out of this with a bigger model.

For a regulated or high-stakes setting, "never confidently wrong on the unknown, with a clean handoff to a human" is often worth more than raw accuracy.

## It sets itself up — and tells you its own accuracy

You don't have to know in advance which of your tasks are rule-shaped. The system works it out from your examples. It finds which fields actually determine the answer and whether they form a clean rule. If they do, it builds the table. If they don't — if the decision genuinely needs judgment — it says so, and falls back to helping the model instead of pretending it has a rule.

Better still, it can **predict its own accuracy before you deploy anything.** From the labeled examples alone, it estimates how much of the task it will answer with a guarantee and how much it will have to defer. You learn what's safely automatable up front, not after a bad incident.

This all sounds great, so I did the thing you're supposed to do and tested it on real data — not my own carefully-built benchmark, but six public datasets. That's where it got interesting, and honest.

## The honest catch

On real data, the system's self-prediction held up well. What it told me its accuracy would be *before* doing anything matched what it actually achieved — usually within a couple of points. It was honest about itself.

But the 100% only showed up on tasks that are *genuinely* rules.

Take "is this mushroom poisonous?" The answer is essentially decided by its smell — a fixed biological rule. The system found that, built the table, and got it 100% right. Clean rule in, perfect answer out.

Now take the real judgment calls — approving a loan, predicting whether someone earns over a threshold, guessing whether a marketing call will land. There is no clean rule. The decision genuinely involves judgment, context, and luck. On those, the table got 75–90%, and on the messiest one it was no better than guessing the most common answer every time.

That's not a bug. It's the truth about those tasks. And the important thing is that the system *told me* — it correctly flagged them as "not a clean rule, I can't certify this" and stepped aside, instead of dressing up a 75% guess as a 100% answer.

## The benchmarks, in one place

For the readers who want the receipts. Every number below is *decision accuracy* — did it pick the
right answer — measured on held-out cases the system never trained on.

**The headline task, and why model size stops mattering.** The marketplace task, with its 80 hidden
per-merchant rules:

| | the LLM decides | the table decides |
|---|---|---|
| a small, cheap model (Llama-3.1-8B) | ~20% | **100%** |
| a capable model (GPT-4o-mini) | ~20% | **100%** |

Once the table makes the call, the gap between a cheap model and an expensive one simply disappears.

**It holds across very different domains.** The same method, four unrelated tasks:

| task (domain) | the system's verdict | accuracy |
|---|---|---|
| per-merchant return policy (commerce) | rule | 100% |
| access control — 144 rules, role × action × resource (security) | rule | 100% |
| content moderation (trust & safety) | rule (mostly) | ~96% |
| incident triage (ops) | judgment — *declines a table* | keeps the human in the loop |

It also scales: 144 access-control rules with a three-part key, discovered automatically, still 100%.

**Safe on brand-new entities.** We hid some entities from training, then tested on them:

| on a never-before-seen entity | table-backed | the LLM alone |
|---|---|---|
| says "send this to a human" (safe) | **100%** | 0% |
| confidently wrong | **0%** | ~95% |

A bigger model doesn't help here — it's confidently wrong just as often.

**It survives messy, free-text input.** When the request is an email and the system has to pull the
fields out before it can look anything up:

| accuracy on email-style input | the LLM decides | extract, then look up |
|---|---|---|
| GPT-4o-mini | 13% | **98%** |
| Llama-3.1-8B | 17% | **90%** |

The error moves from *reasoning* (gone) to *reading the email correctly* — the part models are
already good at.

**And the real, messy one.** Six public datasets, nothing synthetic:

| dataset | what it really is | predicted | actual | vs. "always guess the most common answer" |
|---|---|---|---|---|
| mushroom (poisonous?) | a genuine rule | 100% | 100% | +38 pts |
| nursery admissions | rule-derived | 89% | 90% | +56 pts |
| census income (>$50k?) | judgment | 80% | 79% | +4 pts |
| bank marketing | judgment | 89% | 90% | +2 pts |
| German credit (loans) | judgment | 75% | 69% | +0 |

Two things to read here. First, **predicted ≈ actual on every row** — the system's up-front estimate
of its own accuracy is honest, even on real noisy data. Second, **the 100% only appears where the
task is genuinely a rule.** On real judgment calls it lands at 75–90%, and on the loan data it's no
better than guessing — which it tells you plainly, instead of pretending.

(For contrast: an iterative prompt-optimizer that tries to *teach* the rules to the model, rather than
take the decision away from it, reached **18%** on the marketplace task where the table reached 100%.)

## The tension nobody tells you about

Sit with that, and a deeper pattern shows up. This is the real lesson.

The tasks where this approach is *perfect* — access permissions, pricing, fee schedules, entitlements — are perfect precisely because they're explicit rulebooks. But if a task is an explicit rulebook, **you usually already have the rulebook written down somewhere.** You didn't need a system to learn it from data.

And the tasks where you *do* need to learn the rules from messy history — because nobody wrote them down — are messy *because* they're judgment calls. Which means there usually isn't a clean rule to learn in the first place.

So the place where the magic happens and the place where you actually need it don't overlap as much as the headline suggests.

## So what's it actually good for?

Not "100% on everything." Nothing does that, and anyone who tells you otherwise is selling something.

What it's genuinely good for is narrower and, I'd argue, more useful:

- **100% where a real rule exists** — permissions, prices, routing tables, entitlements — on a model cheap enough to run anywhere, with a full audit trail.
- **An honest "I can't" where a rule doesn't** — risk, judgment, anything genuinely statistical — instead of a confident guess.
- **Never confidently wrong on something it has never seen.**
- **A straight answer, up front, about which of your tasks fall into which bucket** — so you know what you can safely automate and what needs a human before you ship.

For business-critical work, a system that knows its limits beats one that's confidently wrong. That's the whole pitch, and it's an honest one.

I call it Switchboard. The one-line version is the same as where I started:

**The safe way to use an LLM for a critical decision is to not use it for the decision — pull out the rule, and let the model just read and write around it.**
