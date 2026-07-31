"""Tests for the web frontend's FastAPI endpoints: auth, validation, lifecycle.

All tests stub ``webapp.runner._run_analysis`` (or never reach it), so no
LangChain import, no network, and no real graph construction happens here.
"""

import sys
import time
import unittest
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from webapp import runner, server
from webapp.runner import REPORT_KEYS

AUTH = {"X-API-Key": "s3cret"}

DECISION_MD = (
    "**Rating**: Buy\n"
    "\n"
    "**Executive Summary**: Compelling entry point.\n"
    "\n"
    "**Investment Thesis**: Thesis.\n"
    "\n"
    "**Price Target**: 42.0"
)


def _client(password="s3cret"):
    return TestClient(server.create_app(password=password))


def _wait_for_status(client, headers, job_id, status, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if job["status"] == status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {status}")


@pytest.mark.unit
class TestAuth(unittest.TestCase):
    def test_healthz_and_index_always_open(self):
        client = _client()
        self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("TradingAgents", response.text)

    def test_api_requires_credential(self):
        client = _client()
        self.assertEqual(client.get("/api/jobs").status_code, 401)
        self.assertEqual(
            client.get("/api/jobs", headers={"X-API-Key": "wrong"}).status_code, 401
        )
        self.assertEqual(client.get("/api/auth/check").status_code, 401)

    def test_both_header_forms_accepted(self):
        client = _client()
        self.assertEqual(client.get("/api/jobs", headers=AUTH).status_code, 200)
        self.assertEqual(
            client.get(
                "/api/jobs", headers={"Authorization": "Bearer s3cret"}
            ).status_code,
            200,
        )

    def test_auth_disabled_when_password_unset(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("TRADINGAGENTS_WEB_PASSWORD", None)
            with self.assertLogs("webapp.server", level="WARNING"):
                client = TestClient(server.create_app())
        self.assertEqual(client.get("/api/jobs").status_code, 200)
        self.assertEqual(
            client.get("/api/auth/check").json(), {"auth_required": False}
        )

    def test_auth_check_valid_credential(self):
        client = _client()
        self.assertEqual(
            client.get("/api/auth/check", headers=AUTH).json(),
            {"auth_required": True},
        )


@pytest.mark.unit
class TestTickerList(unittest.TestCase):
    def test_tickers_endpoint_open_and_shaped(self):
        client = _client()
        response = client.get("/api/tickers")  # deliberately no auth header
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 1000)
        self.assertEqual(set(data[0]), {"s", "n", "x"})
        self.assertIn("max-age", response.headers.get("cache-control", ""))

    def test_missing_file_returns_empty_list(self):
        with mock.patch.object(
            server, "_STATIC_DIR", server._STATIC_DIR / "does-not-exist"
        ):
            client = _client()
            response = client.get("/api/tickers")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])


@pytest.mark.unit
class TestJobValidation(unittest.TestCase):
    def setUp(self):
        self.client = _client()

    def _post(self, body):
        return self.client.post("/api/jobs", json=body, headers=AUTH)

    def test_path_traversal_ticker_rejected(self):
        self.assertEqual(self._post({"tickers": ["../etc"]}).status_code, 422)

    def test_empty_and_oversized_lists_rejected(self):
        self.assertEqual(self._post({"tickers": []}).status_code, 422)
        self.assertEqual(
            self._post({"tickers": [f"T{i}" for i in range(21)]}).status_code, 422
        )

    def test_bad_date_rejected(self):
        for bad in ("2026-13-99", "July 4", "31/07/2026"):
            self.assertEqual(
                self._post({"tickers": ["NVDA"], "date": bad}).status_code, 422
            )

    def test_bad_asset_type_rejected(self):
        self.assertEqual(
            self._post({"tickers": ["NVDA"], "asset_type": "bond"}).status_code, 422
        )

    def test_tickers_deduped_and_uppercased(self):
        with mock.patch.object(runner.JobStore, "ensure_worker", lambda self: None):
            response = self._post({"tickers": [" nvda ", "NVDA", "msft"]})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            [j["ticker"] for j in response.json()["jobs"]], ["NVDA", "MSFT"]
        )

    def test_date_defaults_to_today_utc(self):
        from datetime import datetime, timezone

        with mock.patch.object(runner.JobStore, "ensure_worker", lambda self: None):
            response = self._post({"tickers": ["NVDA"]})
        expected = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(response.json()["jobs"][0]["trade_date"], expected)

    def test_queue_full_maps_to_429(self):
        with (
            mock.patch.object(runner.JobStore, "ensure_worker", lambda self: None),
            mock.patch.object(runner, "MAX_ACTIVE_JOBS", 1),
        ):
            self.assertEqual(self._post({"tickers": ["NVDA"]}).status_code, 202)
            self.assertEqual(self._post({"tickers": ["MSFT"]}).status_code, 429)


@pytest.mark.unit
class TestJobLifecycle(unittest.TestCase):
    def test_submit_poll_detail(self):
        final_state = {key: f"{key} md" for key in REPORT_KEYS}
        final_state["final_trade_decision"] = DECISION_MD

        with mock.patch.object(
            runner, "_run_analysis", return_value=(final_state, "Buy")
        ):
            client = _client()
            response = client.post(
                "/api/jobs",
                json={"tickers": ["NVDA"], "date": "2026-07-30"},
                headers=AUTH,
            )
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["jobs"][0]["id"]

            job = _wait_for_status(client, AUTH, job_id, "done")
            self.assertEqual(job["rating"], "Buy")
            self.assertEqual(job["price_target"], 42.0)
            self.assertEqual(job["executive_summary"], "Compelling entry point.")
            self.assertEqual(set(job["reports"]), set(REPORT_KEYS))

            rows = client.get("/api/jobs", headers=AUTH).json()
            self.assertEqual(rows[0]["id"], job_id)
            self.assertNotIn("reports", rows[0])  # summaries stay lightweight

            stats = client.get("/api/queue", headers=AUTH).json()
            self.assertEqual(stats["queued"], 0)

    def test_unknown_job_404(self):
        client = _client()
        self.assertEqual(
            client.get("/api/jobs/deadbeef", headers=AUTH).status_code, 404
        )

    def test_graph_never_imported_by_api_layer(self):
        # The lazy-import design must hold: exercising the API with a stubbed
        # runner never loads the LangGraph stack. Other test files may have
        # imported it already in a full-suite run, so assert on the delta.
        already_loaded = "tradingagents.graph.trading_graph" in sys.modules
        with mock.patch.object(
            runner, "_run_analysis", return_value=(dict.fromkeys(REPORT_KEYS, ""), "Hold")
        ):
            client = _client()
            response = client.post("/api/jobs", json={"tickers": ["SPY"]}, headers=AUTH)
            _wait_for_status(client, AUTH, response.json()["jobs"][0]["id"], "done")
        self.assertEqual(
            "tradingagents.graph.trading_graph" in sys.modules, already_loaded
        )


if __name__ == "__main__":
    unittest.main()
