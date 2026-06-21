import importlib
import sys
import types
import unittest
from unittest.mock import patch


if "sentence_transformers" not in sys.modules:
    sentence_transformers = types.ModuleType("sentence_transformers")

    class _DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

    sentence_transformers.SentenceTransformer = _DummySentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers

pipeline = importlib.import_module("pipeline")


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FetchKalshiPaginationTests(unittest.TestCase):
    def _event_payload(self, idx: int):
        return {
            "events": [
                {
                    "title": f"event-{idx}",
                    "event_ticker": f"ticker-{idx}",
                    "category": "test",
                    "markets": [
                        {
                            "rules_primary": "rules",
                            "status": "active",
                            "expected_expiration_time": "2026-01-01T00:00:00Z",
                            "ticker": f"m-{idx}",
                            "yes_sub_title": "yes",
                            "yes_bid_dollars": 0.1,
                            "yes_ask_dollars": 0.2,
                            "no_bid_dollars": 0.3,
                            "no_ask_dollars": 0.4,
                            "expiration_time": "2026-01-02T00:00:00Z",
                            "volume": 1,
                        }
                    ],
                }
            ],
        }

    def test_fetch_kalshi_stops_at_page_limit(self):
        calls = {"count": 0}

        def fake_get(_url, timeout=60):
            calls["count"] += 1
            payload = self._event_payload(calls["count"])
            payload["cursor"] = f"cursor-{calls['count']}"  # endless cursor chain
            return _MockResponse(payload)

        with patch("pipeline.requests.get", side_effect=fake_get), patch("pipeline.time.sleep", return_value=None):
            df = pipeline.fetch_kalshi()

        self.assertEqual(calls["count"], pipeline.KALSHI_MAX_PAGES)
        self.assertEqual(len(df), pipeline.KALSHI_MAX_PAGES)


if __name__ == "__main__":
    unittest.main()
