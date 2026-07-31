"""In-memory job store and serial analysis worker for the web frontend.

Analyses run on ONE background thread, strictly FIFO. Concurrency is not an
option here: ``TradingAgentsGraph.__init__`` mutates the process-global
dataflows vendor config (``dataflows/config.py``), and the reflection memory
log is a shared, unlocked file — two concurrent runs corrupt each other.
This mirrors the serial loop in ``scripts/run_analysis.py``.

The graph modules are imported lazily inside the worker so importing this
module (and the FastAPI app) stays fast and dependency-light, and API tests
never pay the LangChain import cost.
"""

from __future__ import annotations

import queue
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Jobs still queued or running. Terminal jobs don't count against the cap —
# they only occupy memory, which MAX_TOTAL_JOBS bounds.
MAX_ACTIVE_JOBS = 50
# Oldest terminal jobs are pruned past this, keeping a long-lived server's
# memory bounded (each job holds a few hundred KB of report markdown).
MAX_TOTAL_JOBS = 200

REPORT_KEYS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "investment_plan",
    "trader_investment_plan",
    "final_trade_decision",
)

_TERMINAL = ("done", "failed")


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_decision_fields(final_trade_decision: str, decision: str) -> dict:
    """Extract table-friendly fields from the PM decision markdown.

    ``render_pm_decision`` (agents/schemas.py) emits a deterministic shape —
    ``**Rating**: X`` / ``**Executive Summary**: ...`` with optional
    ``**Price Target**`` and ``**Time Horizon**`` lines — but the text has
    passed through an LLM pipeline, so every capture stays defensive.
    """
    from tradingagents.agents.utils.rating import RATINGS_5_TIER, parse_rating

    text = final_trade_decision or ""
    rating = decision if decision in RATINGS_5_TIER else parse_rating(text)

    def _capture(label: str) -> str | None:
        # Value runs until the next blank-line-then-bold section or the end.
        match = re.search(
            rf"\*\*{label}\*\*:\s*(.+?)(?=\n\s*\n\s*\*\*|\Z)", text, re.S
        )
        if not match:
            return None
        value = " ".join(match.group(1).split())
        return value or None

    price_target: float | None = None
    raw_target = _capture("Price Target")
    if raw_target is not None:
        try:
            price_target = float(raw_target.replace("$", "").replace(",", ""))
        except ValueError:
            price_target = None

    return {
        "rating": rating,
        "executive_summary": _capture("Executive Summary"),
        "price_target": price_target,
        "time_horizon": _capture("Time Horizon"),
    }


@dataclass
class Job:
    id: str
    ticker: str
    trade_date: str
    asset_type: str
    company: str | None = None
    status: str = "queued"  # queued | running | done | failed
    created_at: str = field(default_factory=_utcnow)
    started_at: str | None = None
    finished_at: str | None = None
    rating: str | None = None
    price_target: float | None = None
    time_horizon: str | None = None
    executive_summary: str | None = None
    reports: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    def to_summary(self) -> dict:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "company": self.company,
            "trade_date": self.trade_date,
            "asset_type": self.asset_type,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "rating": self.rating,
            "price_target": self.price_target,
            "time_horizon": self.time_horizon,
            "executive_summary": self.executive_summary,
            "error": self.error,
        }

    def to_detail(self) -> dict:
        return {**self.to_summary(), "reports": dict(self.reports)}


def _run_analysis(job: Job) -> tuple[dict, str]:
    """Run one analysis with a fresh graph. Module-level so tests can stub it."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph(config=DEFAULT_CONFIG.copy())
    return graph.propagate(job.ticker, job.trade_date, asset_type=job.asset_type)


class QueueFullError(Exception):
    """Raised when the number of queued+running jobs is at MAX_ACTIVE_JOBS."""


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None

    def submit(
        self,
        ticker: str,
        trade_date: str,
        asset_type: str,
        company: str | None = None,
    ) -> Job:
        with self._lock:
            active = sum(1 for j in self._jobs.values() if j.status not in _TERMINAL)
            if active >= MAX_ACTIVE_JOBS:
                raise QueueFullError(
                    f"queue full: {active} jobs already queued or running"
                )
            job_id = uuid.uuid4().hex[:8]
            while job_id in self._jobs:
                job_id = uuid.uuid4().hex[:8]
            job = Job(
                id=job_id,
                ticker=ticker,
                trade_date=trade_date,
                asset_type=asset_type,
                company=company,
            )
            self._jobs[job_id] = job
            self._prune_locked()
        self._queue.put(job_id)
        self.ensure_worker()
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [job.to_summary() for job in reversed(self._jobs.values())]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_detail(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_detail() if job else None

    def queue_stats(self) -> dict:
        with self._lock:
            queued = sum(1 for j in self._jobs.values() if j.status == "queued")
            running = next(
                (j.id for j in self._jobs.values() if j.status == "running"), None
            )
        return {
            "queued": queued,
            "running": running,
            "worker_alive": self._worker is not None and self._worker.is_alive(),
        }

    def ensure_worker(self) -> None:
        # Lazy start (nothing spawns at import time) and restart-on-submit if
        # a previous worker thread died on an unexpected error.
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_loop, name="tradingagents-worker", daemon=True
            )
            self._worker.start()

    def _prune_locked(self) -> None:
        overflow = len(self._jobs) - MAX_TOTAL_JOBS
        if overflow <= 0:
            return
        for job_id in [
            j.id for j in self._jobs.values() if j.status in _TERMINAL
        ][:overflow]:
            del self._jobs[job_id]

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                job = self.get(job_id)
                if job is None:  # pruned while queued
                    continue
                with self._lock:
                    job.status = "running"
                    job.started_at = _utcnow()
                try:
                    final_state, decision = _run_analysis(job)
                    reports = {
                        key: str(final_state.get(key) or "") for key in REPORT_KEYS
                    }
                    fields = parse_decision_fields(
                        reports["final_trade_decision"], decision
                    )
                    with self._lock:
                        job.reports = reports
                        job.rating = fields["rating"]
                        job.price_target = fields["price_target"]
                        job.time_horizon = fields["time_horizon"]
                        job.executive_summary = fields["executive_summary"]
                        job.status = "done"
                        job.finished_at = _utcnow()
                except Exception as exc:  # noqa: BLE001 - one bad run must not kill the worker
                    with self._lock:
                        job.error = f"{type(exc).__name__}: {exc}"[:2000]
                        job.status = "failed"
                        job.finished_at = _utcnow()
            finally:
                self._queue.task_done()
