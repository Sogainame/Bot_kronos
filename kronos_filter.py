"""Kronos filter for Sniper bot.

Reads kronos_signal.json (produced by kronos_service.py) and decides whether
to allow or block a trade based on Kronos's directional consensus.

Filter logic (conservative — never overrides bot, only blocks bad trades):
  1. No signal for current window → ALLOW (don't interfere with bot)
  2. Stale signal (different window_ts) → ALLOW
  3. consensus < 0.70 → BLOCK ("model not confident, risky window")
  4. direction == NEUTRAL → BLOCK (same reason)
  5. direction != bot's direction → BLOCK ("model disagrees")
  6. direction == bot's direction → ALLOW
"""
from __future__ import annotations

import json
import time
from pathlib import Path

SIGNAL_FILE = Path("kronos_signal.json")
MIN_CONSENSUS = 0.70   # require 14/20 votes for one side
MAX_AGE_SECS = 280     # signal older than this = stale


def load_signal() -> dict | None:
    if not SIGNAL_FILE.exists():
        return None
    try:
        return json.loads(SIGNAL_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def evaluate(bot_direction: str, current_window_ts: int) -> tuple[bool, str, dict | None]:
    """Decide whether to block the trade.

    Returns (block, reason, signal_dict).
    block=True means SKIP the trade.
    """
    sig = load_signal()
    if sig is None:
        return False, "kronos:no_file", None

    # Window mismatch — signal is for different window
    if sig.get("window_ts") != current_window_ts:
        return False, f"kronos:wrong_window({sig.get('window_ts')}!={current_window_ts})", sig

    # Age check
    try:
        from datetime import datetime
        gen_ts = datetime.fromisoformat(sig["generated_at"].replace("Z", "+00:00")).timestamp()
        age = time.time() - gen_ts
        if age > MAX_AGE_SECS:
            return False, f"kronos:stale({age:.0f}s)", sig
    except (KeyError, ValueError):
        pass

    consensus = sig.get("consensus", 0.0)
    direction = sig.get("direction", "NEUTRAL")

    if consensus < MIN_CONSENSUS:
        return True, f"kronos:low_consensus({consensus:.2f})", sig

    if direction == "NEUTRAL":
        return True, "kronos:neutral", sig

    if direction != bot_direction:
        return True, f"kronos:disagree({direction}vs{bot_direction})", sig

    return False, f"kronos:ok({direction}@{consensus:.2f})", sig


def format_signal_for_log(sig: dict | None) -> str:
    if sig is None:
        return "Kronos: no_signal"
    return (
        f"Kronos: dir={sig['direction']} "
        f"votes={sig['votes_up']}/{sig['n_paths']} "
        f"mean={sig['mean_pred_pct']:+.3f}% "
        f"spread={sig['spread_pct']:.3f}%"
    )
