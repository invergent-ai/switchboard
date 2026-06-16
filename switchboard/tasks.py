"""The four synthetic decision tasks used in the article, fully self-contained.

Each task exposes `make_splits(seed)` returning {"train": [...], "test": [...], ...} where every
example is {"input": {...fields...}, "target": {"decision", "policy_citations", ...}}. The per-entity
rule tables are drawn at random with a fixed seed and WITHHELD from `policy_md` — a system must learn
or look them up, it can't read them off the base instructions.

Tasks: marketplace (per-merchant return policy), authz (RBAC, composite key), moderation
(compositional rules / judgment), triage (lookup + an override).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Task:
    name: str
    policy_md: str
    make_splits: Callable[..., dict]


# ─────────────────────────────────────────────────────────────────────────────
# marketplace — 20 merchants x 4 item states = 80 hidden per-merchant rules
# ─────────────────────────────────────────────────────────────────────────────
MK_MERCHANTS = [f"M{i:02d}" for i in range(1, 21)]
MK_CASES = ["defective", "unused", "final_sale", "opened"]
MK_DECISIONS = ["approve_refund", "deny_refund", "offer_replacement", "request_more_info", "escalate"]
_MK_TABLE = {
    (m, c): MK_DECISIONS[int(r)]
    for (m, c), r in zip(
        ((m, c) for m in MK_MERCHANTS for c in MK_CASES),
        np.random.default_rng(12345).integers(0, len(MK_DECISIONS), size=80),
    )
}
_MK_REPLIES = {
    "approve_refund": "We've approved your refund and it will be processed shortly.",
    "offer_replacement": "We'll arrange a replacement under our defective-item policy.",
    "deny_refund": "This case falls outside our policy, so we're unable to proceed.",
    "escalate": "I'm passing your case to a manager for review.",
    "request_more_info": "Could you share a photo of the issue so we can review it under our policy?",
}
MK_POLICY_MD = (
    "This marketplace hosts 20 independent merchants (M01..M20). EACH merchant sets its OWN return "
    "policy: the decision depends on BOTH merchant_id AND item_state (defective/unused/final_sale/"
    "opened). The full per-merchant table is NOT reproduced here — apply the governing rule and cite "
    "it as '<merchant_id>-<item_state>'. Global overrides:\nG1: orders older than 365 days are denied."
    "\nG2: unknown merchant_id -> request more info.\nDecisions: approve_refund, deny_refund, "
    "offer_replacement, request_more_info, escalate."
)


def _mk_decide(x):
    m, s, days = x.get("merchant_id"), x.get("item_state"), x.get("order_days_ago")
    if m not in MK_MERCHANTS:
        return "request_more_info", ["G2"]
    if isinstance(days, int) and days > 365:
        return "deny_refund", ["G1"]
    return _MK_TABLE[(m, s)], [f"{m}-{s}"]


def _mk_example(inp, *slices):
    d, c = _mk_decide(inp)
    return {
        "input": inp,
        "target": {"decision": d, "policy_citations": c, "internal_reasoning_brief": f"rule {c}", "customer_reply": _MK_REPLIES[d]},
        "metadata": {"slices": sorted(set(slices))},
    }


def _mk_ticket(i, m, s, rng, over=False):
    cat = ["electronics", "apparel", "shoes", "home", "toys"][int(rng.integers(5))]
    days = int(rng.integers(366, 600)) if over else int(rng.integers(0, 365))
    return {"ticket_id": f"K-{i:05d}", "merchant_id": m, "item_state": s, "order_days_ago": days,
            "item_category": cat, "requested_action": "refund",
            "customer_message": f"Order from {m}: my item is {s}; I'd like a refund."}


def make_marketplace_splits(seed=0):
    rng = np.random.default_rng(seed)
    n = itertools.count()
    cells = [_mk_example(_mk_ticket(next(n), m, s, rng), f"merchant_{m}") for m in MK_MERCHANTS for s in MK_CASES]
    rnd = lambda: _mk_example(_mk_ticket(next(n), MK_MERCHANTS[int(rng.integers(20))], MK_CASES[int(rng.integers(4))], rng), "random")  # noqa: E731
    off = lambda: _mk_example(_mk_ticket(next(n), MK_MERCHANTS[int(rng.integers(20))], MK_CASES[int(rng.integers(4))], rng, over=True), "off_domain")  # noqa: E731
    return {"train": cells + [rnd() for _ in range(20)], "validation": [rnd() for _ in range(40)],
            "test": [rnd() for _ in range(60)], "off_domain": [off() for _ in range(20)]}


# ─────────────────────────────────────────────────────────────────────────────
# authz — RBAC, 6 roles x 6 actions x 4 resources = 144 rules, composite key
# ─────────────────────────────────────────────────────────────────────────────
AZ_ROLES = ["viewer", "editor", "admin", "billing_clerk", "support_agent", "auditor"]
AZ_ACTIONS = ["read", "write", "delete", "share", "export", "configure"]
AZ_RESOURCES = ["document", "billing_record", "user_account", "audit_log"]
AZ_DECISIONS = ["allow", "deny", "require_approval", "require_mfa"]
_AZ_TABLE = {
    (a, b, c): AZ_DECISIONS[int(r)]
    for (a, b, c), r in zip(
        ((a, b, c) for a in AZ_ROLES for b in AZ_ACTIONS for c in AZ_RESOURCES),
        np.random.default_rng(7).integers(0, len(AZ_DECISIONS), size=144),
    )
}
_AZ_REPLIES = {"allow": "Access granted.", "deny": "Access denied by policy.",
               "require_approval": "This action requires a manager's approval.", "require_mfa": "Please complete MFA to continue."}
AZ_POLICY_MD = (
    "Per-tenant access-control matrix. The decision depends on principal_role, action, AND "
    "resource_type together. Apply the matching rule and cite it 'role:action:resource_type'. The "
    "full matrix is not reproduced here. Decisions: allow, deny, require_approval, require_mfa."
)


def _az_example(inp, *slices):
    key = (inp.get("principal_role"), inp.get("action"), inp.get("resource_type"))
    d = _AZ_TABLE.get(key, "deny")
    cite = [":".join(key)] if key in _AZ_TABLE else []
    return {"input": inp, "target": {"decision": d, "policy_citations": cite, "internal_reasoning_brief": f"rule {cite}", "customer_reply": _AZ_REPLIES[d]},
            "metadata": {"slices": sorted(set(slices))}}


def _az_ticket(i, a, b, c, rng):
    return {"ticket_id": f"Q-{i:05d}", "principal_role": a, "action": b, "resource_type": c,
            "environment": ["production", "staging", "development"][int(rng.integers(3))],
            "request_hour": int(rng.integers(0, 24)), "requested_action": b,
            "customer_message": f"{a} requests to {b} a {c}."}


def make_authz_splits(seed=0):
    rng = np.random.default_rng(seed)
    n = itertools.count()
    cells = [_az_example(_az_ticket(next(n), a, b, c, rng), f"role_{a}") for a in AZ_ROLES for b in AZ_ACTIONS for c in AZ_RESOURCES]
    rnd = lambda: _az_example(_az_ticket(next(n), AZ_ROLES[int(rng.integers(6))], AZ_ACTIONS[int(rng.integers(6))], AZ_RESOURCES[int(rng.integers(4))], rng), "random")  # noqa: E731
    return {"train": cells + [rnd() for _ in range(40)], "validation": [rnd() for _ in range(60)], "test": [rnd() for _ in range(60)]}


# ─────────────────────────────────────────────────────────────────────────────
# moderation — compositional rules (judgment), no per-entity lookup key
# ─────────────────────────────────────────────────────────────────────────────
MOD_RULES = [
    ("M1", "A credible threat of violence is escalated to the safety team."),
    ("M2", "A slur or hate term is removed."),
    ("M3", "A known spam pattern from an account younger than 7 days is removed."),
    ("M4", "More than 5 user reports -> human review."),
    ("M5", "An external link from an account younger than 1 day has its reach limited."),
    ("M6", "Otherwise allow."),
]
MOD_DECISIONS = ["allow", "limit_reach", "flag_review", "remove", "escalate_safety"]
_MOD_REPLIES = {"allow": "Within policy.", "limit_reach": "Distribution limited pending review.",
                "flag_review": "Queued for human review.", "remove": "Removed for violating policy.", "escalate_safety": "Escalated to the safety team."}
MOD_POLICY_MD = "Moderate each post by applying these rules in order of severity; cite the rule used.\n" + "\n".join(f"{p}: {t}" for p, t in MOD_RULES)


def _mod_decide(x):
    if x.get("has_threat"):
        return "escalate_safety", ["M1"]
    if x.get("has_slur"):
        return "remove", ["M2"]
    age = x.get("account_age_days")
    if x.get("is_spam_pattern") and isinstance(age, int) and age < 7:
        return "remove", ["M3"]
    if isinstance(x.get("report_count"), int) and x["report_count"] > 5:
        return "flag_review", ["M4"]
    if x.get("has_external_link") and isinstance(age, int) and age < 1:
        return "limit_reach", ["M5"]
    return "allow", ["M6"]


def _mod_example(inp):
    d, c = _mod_decide(inp)
    return {"input": inp, "target": {"decision": d, "policy_citations": c, "internal_reasoning_brief": f"rule {c}", "customer_reply": _MOD_REPLIES[d]}, "metadata": {"slices": []}}


def _mod_post(i, rng):
    return {"ticket_id": f"P-{i:05d}", "has_threat": bool(rng.integers(0, 10) == 0), "has_slur": bool(rng.integers(0, 8) == 0),
            "is_spam_pattern": bool(rng.integers(0, 3) == 0), "has_external_link": bool(rng.integers(0, 2)),
            "report_count": int(rng.integers(0, 12)), "account_age_days": int(rng.integers(0, 400)),
            "requested_action": "moderate", "customer_message": "Post submitted for moderation."}


def make_moderation_splits(seed=0):
    rng = np.random.default_rng(seed)
    n = itertools.count()
    ex = lambda: _mod_example(_mod_post(next(n), rng))  # noqa: E731
    return {"train": [ex() for _ in range(120)], "validation": [ex() for _ in range(40)], "test": [ex() for _ in range(60)]}


# ─────────────────────────────────────────────────────────────────────────────
# triage — per-(service, alert) lookup + a customer-impact override
# ─────────────────────────────────────────────────────────────────────────────
TR_SERVICES = ["payments", "auth", "search", "email", "cdn", "database"]
TR_ALERTS = ["latency", "error_rate", "saturation", "down", "cert_expiry"]
TR_DECISIONS = ["page_oncall", "create_ticket", "auto_resolve", "escalate_incident"]
_TR_BASE = ["page_oncall", "create_ticket", "auto_resolve"]
_TR_TABLE = {
    (s, t): _TR_BASE[int(r)]
    for (s, t), r in zip(((s, t) for s in TR_SERVICES for t in TR_ALERTS), np.random.default_rng(11).integers(0, 3, size=30))
}
_TR_REPLIES = {"page_oncall": "Paging on-call.", "create_ticket": "Ticket created.", "auto_resolve": "Auto-resolving.", "escalate_incident": "Declaring an incident."}
TR_POLICY_MD = (
    "Triage each alert. Each (service, alert_type) pair has its own routing rule; cite it "
    "'service/alert_type'. The full table is not reproduced here. Override G1: any customer-facing "
    "alert affecting more than 1000 users is escalated as an incident. Decisions: page_oncall, "
    "create_ticket, auto_resolve, escalate_incident."
)


def _tr_decide(x):
    if bool(x.get("is_customer_facing")) and isinstance(x.get("affected_users"), int) and x["affected_users"] > 1000:
        return "escalate_incident", ["G1"]
    s, t = x.get("service"), x.get("alert_type")
    return _TR_TABLE.get((s, t), "create_ticket"), ([f"{s}/{t}"] if (s, t) in _TR_TABLE else [])


def _tr_example(inp):
    d, c = _tr_decide(inp)
    return {"input": inp, "target": {"decision": d, "policy_citations": c, "internal_reasoning_brief": f"rule {c}", "customer_reply": _TR_REPLIES[d]}, "metadata": {"slices": []}}


def _tr_ticket(i, s, t, rng, force=False):
    cf = bool(rng.integers(0, 2)) or force
    users = int(rng.integers(1001, 50000)) if force else int(rng.integers(0, 5000))
    return {"ticket_id": f"A-{i:05d}", "service": s, "alert_type": t, "is_customer_facing": cf, "affected_users": users,
            "region": ["us", "eu", "apac"][int(rng.integers(3))], "requested_action": "triage",
            "customer_message": f"{t} alert on {s} ({'customer-facing' if cf else 'internal'}, {users} users)."}


def make_triage_splits(seed=0):
    rng = np.random.default_rng(seed)
    n = itertools.count()
    cells = [_tr_example(_tr_ticket(next(n), s, t, rng)) for s in TR_SERVICES for t in TR_ALERTS]
    rnd = lambda: _tr_example(_tr_ticket(next(n), TR_SERVICES[int(rng.integers(6))], TR_ALERTS[int(rng.integers(5))], rng))  # noqa: E731
    return {"train": cells + [rnd() for _ in range(70)], "validation": [rnd() for _ in range(40)], "test": [rnd() for _ in range(60)]}


TASKS = {
    "marketplace": Task("marketplace", MK_POLICY_MD, make_marketplace_splits),
    "authz": Task("authz", AZ_POLICY_MD, make_authz_splits),
    "moderation": Task("moderation", MOD_POLICY_MD, make_moderation_splits),
    "triage": Task("triage", TR_POLICY_MD, make_triage_splits),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; expected one of {list(TASKS)}")
    return TASKS[name]
