"""Prototype-level safety mechanisms.

These are deliberately simple and honest — the goal is to demonstrate *where* safety
hooks belong in an agent, not to claim production hardening:

* ``RateLimiter``    — caps external API calls (cost / abuse control).
* ``sanitize_query`` — bounds and cleans user input before it hits an API.
* ``grounding_check``— flags LLM summaries that may not be supported by the source
                       abstract (a lightweight hallucination tripwire).
* ``confirm_action`` — gates any outward / hard-to-reverse action behind a human y/n.
"""
from __future__ import annotations

import re
import time
from collections import deque
from typing import Deque, List, Tuple


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` within ``period`` seconds."""

    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        self.max_calls = max(1, int(max_calls))
        self.period = float(period)
        self._calls: Deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > self.period:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            sleep_for = self.period - (now - self._calls[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self._calls and now - self._calls[0] > self.period:
                self._calls.popleft()
        self._calls.append(time.monotonic())


def sanitize_query(query: str, max_len: int = 300) -> str:
    """Strip control characters and cap length before sending to an external API."""
    query = "".join(ch for ch in (query or "") if ch.isprintable())
    query = re.sub(r"\s+", " ", query).strip()
    return query[:max_len]


# A few content patterns we never want to surface in a "grounded" summary.
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def grounding_check(summary: str, abstract: str) -> Tuple[bool, List[str]]:
    """Heuristically decide whether ``summary`` is grounded in ``abstract``.

    Returns ``(ok, warnings)``. We flag two cheap red flags: (1) the summary is
    longer than the source (likely embellished), and (2) it introduces numeric
    claims absent from the abstract (a common hallucination pattern). This is a
    tripwire, not a proof of faithfulness.
    """
    warnings: List[str] = []
    summary = (summary or "").strip()
    abstract = (abstract or "").strip()
    if not summary:
        return True, warnings
    if len(summary) > len(abstract) + 40:
        warnings.append("summary longer than source abstract")
    src_nums = set(_NUM_RE.findall(abstract))
    new_nums = [n for n in _NUM_RE.findall(summary) if n not in src_nums]
    if new_nums:
        warnings.append(f"summary introduces numbers not in abstract: {new_nums}")
    return (len(warnings) == 0), warnings


def confirm_action(description: str, *, require_confirmation: bool = True, auto_yes: bool = False) -> bool:
    """Gate an outward / irreversible action behind explicit human approval.

    The prototype has no destructive actions, but this is the single chokepoint a
    future "open 20 PDFs" or "email me this digest" action would have to pass through.
    """
    if not require_confirmation or auto_yes:
        return True
    try:
        ans = input(f"[safety] {description} Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):  # non-interactive context -> refuse by default
        print("\n[safety] no confirmation available; declining action.")
        return False
    return ans in {"y", "yes"}
