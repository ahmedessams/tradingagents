"""Tests for the web frontend's job store, worker loop, and decision parsing.

The worker never touches the real graph here: ``webapp.runner._run_analysis``
is monkeypatched, keeping these tests network- and LangChain-free.
"""

import time
import unittest
from unittest import mock

import pytest

from webapp import runner
from webapp.runner import (
    REPORT_KEYS,
    JobStore,
    QueueFullError,
    parse_decision_fields,
)

DECISION_MD = (
    "**Rating**: Overweight\n"
    "\n"
    "**Executive Summary**: Strong data-center demand offsets valuation risk.\n"
    "\n"
    "**Investment Thesis**: Multi-paragraph thesis text here.\n"
    "\n"
    "**Price Target**: 210.5\n"
    "\n"
    "**Time Horizon**: 3-6 months"
)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.unit
class TestParseDecisionFields(unittest.TestCase):
    def test_full_decision_markdown(self):
        fields = parse_decision_fields(DECISION_MD, "Overweight")
        self.assertEqual(fields["rating"], "Overweight")
        self.assertEqual(
            fields["executive_summary"],
            "Strong data-center demand offsets valuation risk.",
        )
        self.assertEqual(fields["price_target"], 210.5)
        self.assertEqual(fields["time_horizon"], "3-6 months")

    def test_optional_lines_absent(self):
        md = "**Rating**: Hold\n\n**Executive Summary**: Wait and see.\n\n**Investment Thesis**: X."
        fields = parse_decision_fields(md, "Hold")
        self.assertIsNone(fields["price_target"])
        self.assertIsNone(fields["time_horizon"])
        self.assertEqual(fields["executive_summary"], "Wait and see.")

    def test_garbage_price_target_is_none(self):
        md = DECISION_MD.replace("210.5", "around fair value")
        self.assertIsNone(parse_decision_fields(md, "Buy")["price_target"])

    def test_dollar_and_comma_price_target(self):
        md = DECISION_MD.replace("210.5", "$1,250.00")
        self.assertEqual(parse_decision_fields(md, "Buy")["price_target"], 1250.0)

    def test_unknown_decision_falls_back_to_text_rating(self):
        fields = parse_decision_fields(DECISION_MD, "NOT-A-RATING")
        self.assertEqual(fields["rating"], "Overweight")

    def test_empty_text_defaults(self):
        fields = parse_decision_fields("", "Sell")
        self.assertEqual(fields["rating"], "Sell")
        self.assertIsNone(fields["executive_summary"])
        self.assertIsNone(fields["price_target"])
        self.assertIsNone(fields["time_horizon"])


@pytest.mark.unit
class TestJobStore(unittest.TestCase):
    def _store_without_worker(self):
        store = JobStore()
        # Keep the worker from starting so queue/list state stays inspectable.
        store.ensure_worker = lambda: None
        return store

    def test_submit_creates_queued_job(self):
        store = self._store_without_worker()
        job = store.submit("NVDA", "2026-07-31", "stock")
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.ticker, "NVDA")
        self.assertEqual(len(job.id), 8)
        self.assertIsNotNone(store.get(job.id))

    def test_list_newest_first(self):
        store = self._store_without_worker()
        first = store.submit("AAPL", "2026-07-31", "stock")
        second = store.submit("MSFT", "2026-07-31", "stock")
        listed = store.list()
        self.assertEqual([j["id"] for j in listed], [second.id, first.id])

    def test_get_unknown_returns_none(self):
        store = self._store_without_worker()
        self.assertIsNone(store.get("nope"))
        self.assertIsNone(store.get_detail("nope"))

    def test_queue_cap(self):
        store = self._store_without_worker()
        with mock.patch.object(runner, "MAX_ACTIVE_JOBS", 2):
            store.submit("A", "2026-07-31", "stock")
            store.submit("B", "2026-07-31", "stock")
            with self.assertRaises(QueueFullError):
                store.submit("C", "2026-07-31", "stock")

    def test_queue_stats(self):
        store = self._store_without_worker()
        store.submit("NVDA", "2026-07-31", "stock")
        stats = store.queue_stats()
        self.assertEqual(stats["queued"], 1)
        self.assertIsNone(stats["running"])
        self.assertFalse(stats["worker_alive"])


@pytest.mark.unit
class TestWorkerLoop(unittest.TestCase):
    def test_success_sets_fields_and_reports(self):
        final_state = {key: f"{key} content" for key in REPORT_KEYS}
        final_state["final_trade_decision"] = DECISION_MD

        def fake_run(job):
            return final_state, "Overweight"

        with mock.patch.object(runner, "_run_analysis", side_effect=fake_run):
            store = JobStore()
            job = store.submit("NVDA", "2026-07-31", "stock")
            self.assertTrue(_wait_until(lambda: store.get(job.id).status == "done"))

        done = store.get(job.id)
        self.assertEqual(done.rating, "Overweight")
        self.assertEqual(done.price_target, 210.5)
        self.assertEqual(set(done.reports), set(REPORT_KEYS))
        self.assertIsNotNone(done.started_at)
        self.assertIsNotNone(done.finished_at)

    def test_failure_records_error_and_next_job_still_runs(self):
        calls = []

        def fake_run(job):
            calls.append(job.ticker)
            if job.ticker == "BAD":
                raise RuntimeError("provider exploded")
            return dict.fromkeys(REPORT_KEYS, ""), "Hold"

        with mock.patch.object(runner, "_run_analysis", side_effect=fake_run):
            store = JobStore()
            bad = store.submit("BAD", "2026-07-31", "stock")
            good = store.submit("GOOD", "2026-07-31", "stock")
            self.assertTrue(
                _wait_until(
                    lambda: store.get(bad.id).status == "failed"
                    and store.get(good.id).status == "done"
                )
            )

        self.assertEqual(calls, ["BAD", "GOOD"])
        self.assertIn("RuntimeError: provider exploded", store.get(bad.id).error)
        self.assertIsNone(store.get(good.id).error)

    def test_construction_error_is_captured(self):
        # TradingAgentsGraph.__init__ itself can raise (bad provider config);
        # that must land in job.error, not kill the worker thread.
        def fake_run(job):
            raise ValueError("Unknown provider 'nope'")

        with mock.patch.object(runner, "_run_analysis", side_effect=fake_run):
            store = JobStore()
            job = store.submit("NVDA", "2026-07-31", "stock")
            self.assertTrue(_wait_until(lambda: store.get(job.id).status == "failed"))
            self.assertTrue(store.queue_stats()["worker_alive"])


if __name__ == "__main__":
    unittest.main()
