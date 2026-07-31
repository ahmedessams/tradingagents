"""FastAPI application for the TradingAgents web frontend.

Serves a single-page UI (``static/index.html``) plus a small JSON API that
enqueues analyses on the serial worker in ``webapp.runner`` and reports
results for the table view.

Auth is a single shared password from ``TRADINGAGENTS_WEB_PASSWORD``,
accepted as ``Authorization: Bearer <pw>`` or ``X-API-Key: <pw>``. When the
variable is unset auth is disabled (local development) with a loud warning —
an unprotected public deployment lets anyone spend the LLM API budget.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from tradingagents.dataflows.utils import safe_ticker_component
from webapp.runner import JobStore, QueueFullError

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

MAX_TICKERS_PER_REQUEST = 20


class JobRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=MAX_TICKERS_PER_REQUEST)
    date: str | None = None
    asset_type: str = "stock"

    @field_validator("tickers")
    @classmethod
    def _validate_tickers(cls, tickers: list[str]) -> list[str]:
        cleaned: list[str] = []
        for raw in tickers:
            ticker = raw.strip().upper()
            if not ticker:
                continue
            try:
                safe_ticker_component(ticker)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            if ticker not in cleaned:
                cleaned.append(ticker)
        if not cleaned:
            raise ValueError("no valid tickers given")
        return cleaned

    @field_validator("date")
    @classmethod
    def _validate_date(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"invalid date {value!r}: expected YYYY-MM-DD") from exc
        return value

    @field_validator("asset_type")
    @classmethod
    def _validate_asset_type(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in ("stock", "crypto"):
            raise ValueError("asset_type must be 'stock' or 'crypto'")
        return value


def create_app(password: str | None = None) -> FastAPI:
    """Build the app. ``password=None`` reads TRADINGAGENTS_WEB_PASSWORD."""
    if password is None:
        password = os.environ.get("TRADINGAGENTS_WEB_PASSWORD") or None
    if password is None:
        logger.warning(
            "TRADINGAGENTS_WEB_PASSWORD is not set - auth is DISABLED; anyone "
            "who can reach this server can trigger LLM runs on your API key."
        )

    app = FastAPI(title="TradingAgents Web", docs_url=None, redoc_url=None)
    app.state.password = password
    app.state.store = JobStore()

    def require_auth(request: Request) -> None:
        if app.state.password is None:
            return
        supplied = request.headers.get("x-api-key", "")
        if not supplied:
            authorization = request.headers.get("authorization", "")
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer":
                supplied = token.strip()
        if not supplied or not secrets.compare_digest(supplied, app.state.password):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/api/auth/check")
    def auth_check(request: Request) -> dict:
        require_auth(request)
        return {"auth_required": app.state.password is not None}

    @app.post("/api/jobs", status_code=202)
    def create_jobs(body: JobRequest, request: Request) -> dict:
        require_auth(request)
        trade_date = body.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        jobs = []
        for ticker in body.tickers:
            try:
                job = app.state.store.submit(ticker, trade_date, body.asset_type)
            except QueueFullError as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            jobs.append(job.to_summary())
        return {"jobs": jobs}

    @app.get("/api/jobs")
    def list_jobs(request: Request) -> list[dict]:
        require_auth(request)
        return app.state.store.list()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict:
        require_auth(request)
        detail = app.state.store.get_detail(job_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="job not found")
        return detail

    @app.get("/api/queue")
    def queue_stats(request: Request) -> dict:
        require_auth(request)
        return app.state.store.queue_stats()

    return app


app = create_app()
