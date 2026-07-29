"""Headless analysis runner for scheduled / containerized deployments.

The interactive CLI (``tradingagents``) drives every choice through terminal
prompts, which no scheduler can answer. This runner takes the run inputs from
the environment (or argv) instead, so a cron job — Render, GitHub Actions, a
plain crontab — can execute an analysis unattended:

    TRADINGAGENTS_TICKERS        comma-separated tickers to analyze (required
                                 unless passed as argv[1])
    TRADINGAGENTS_ANALYSIS_DATE  YYYY-MM-DD trade date (default: today, UTC;
                                 argv[2] overrides)
    TRADINGAGENTS_ASSET_TYPE     "stock" (default) or "crypto"

Every other knob — provider, models, debate depth, output language, data
vendors — comes from the same TRADINGAGENTS_* variables the package already
honors (see tradingagents/default_config.py), so a .env that works for the
CLI works here too.

Each ticker's final decision is printed to stdout so log-based platforms
capture it even when the results directory is ephemeral. Exits non-zero if
any ticker fails, so schedulers mark the run as failed instead of silently
green.

Usage:
    python scripts/run_analysis.py                  # env-driven
    python scripts/run_analysis.py NVDA,MSFT        # tickers from argv
    python scripts/run_analysis.py NVDA 2024-05-10  # explicit trade date
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone


def _resolve_tickers() -> list[str]:
    raw = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TRADINGAGENTS_TICKERS", "")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        print(
            "No tickers given. Set TRADINGAGENTS_TICKERS (comma-separated) or pass "
            "them as the first argument, e.g. `python scripts/run_analysis.py NVDA,MSFT`.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return tickers


def _resolve_trade_date() -> str:
    raw = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.environ.get("TRADINGAGENTS_ANALYSIS_DATE", "")
    ).strip()
    if not raw:
        # UTC keeps "today" stable regardless of the host's timezone; a job
        # scheduled after the US close (>= 21:00 UTC) still lands on the same
        # calendar day as the trading session it analyzes.
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid analysis date {raw!r}: expected YYYY-MM-DD.", file=sys.stderr)
        raise SystemExit(2) from None
    return raw


def main() -> int:
    tickers = _resolve_tickers()
    trade_date = _resolve_trade_date()
    asset_type = os.environ.get("TRADINGAGENTS_ASSET_TYPE", "stock").strip().lower() or "stock"

    # Import after arg validation so bad inputs fail fast without paying the
    # LangChain/provider import cost.
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph(config=DEFAULT_CONFIG.copy())

    failures = 0
    for ticker in tickers:
        print(f"=== {ticker} @ {trade_date} ({asset_type}) ===", flush=True)
        try:
            _, decision = graph.propagate(ticker, trade_date, asset_type=asset_type)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the batch
            failures += 1
            print(f"[FAILED] {ticker}: {exc}", file=sys.stderr, flush=True)
            continue
        print(f"Decision for {ticker}: {decision}", flush=True)

    if failures:
        print(f"{failures}/{len(tickers)} ticker(s) failed.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
