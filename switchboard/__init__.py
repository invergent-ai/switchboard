"""Switchboard: extract the rule, let the model just read and write around it.

A small, self-contained reference implementation that accompanies the article. The core pieces:
  - detector  : discover the minimal key that determines the decision + a purity/coverage verdict
  - table     : the certified lookup table + coverage-aware routing (answer or abstain)
  - provider  : a minimal async chat client (OpenRouter / OpenAI-compatible)
  - tasks     : the four synthetic decision tasks
"""

from switchboard.detector import discover_key, purity, recommend
from switchboard.provider import Provider
from switchboard.scoring import get_decision, parse_json
from switchboard.table import ABSTAIN, learn_table, lookup
from switchboard.tasks import TASKS, get_task

__all__ = [
    "discover_key",
    "recommend",
    "purity",
    "learn_table",
    "lookup",
    "ABSTAIN",
    "Provider",
    "get_decision",
    "parse_json",
    "get_task",
    "TASKS",
]
