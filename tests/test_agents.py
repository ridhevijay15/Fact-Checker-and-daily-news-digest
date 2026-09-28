import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agents


class AgentTests(unittest.TestCase):
    def test_gemini_rest_request_parses_json_without_sdk_client(self):
        response = unittest.mock.Mock()
        response.ok = True
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "{\"claims\": [\"Claim A\"]}"}]}}]
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "local-test-key"}), patch.object(
            agents.requests, "post", return_value=response
        ) as request:
            result = agents._generate_json("test prompt")

        self.assertEqual(result, {"claims": ["Claim A"]})
        self.assertEqual(request.call_args.kwargs["params"], {"key": "local-test-key"})
        self.assertEqual(request.call_args.kwargs["timeout"], 45)

    def test_gemini_falls_back_when_model_is_overloaded(self):
        busy = unittest.mock.Mock(status_code=503)
        ready = unittest.mock.Mock(
            status_code=200,
            ok=True,
            json=lambda: {"candidates": [{"content": {"parts": [{"text": "{\"ok\": true}"}]}}]},
        )
        with patch.dict("os.environ", {"GEMINI_API_KEY": "local-test-key"}), patch.object(
            agents.requests, "post", side_effect=[busy, ready]
        ) as request:
            result = agents._generate_json("test prompt")

        self.assertEqual(result, {"ok": True})
        self.assertIn("gemini-3.8-flash", request.call_args_list[0].args[0])
        self.assertIn("gemini-3.7-flash", request.call_args_list[1].args[0])

    def test_gemini_reports_temporary_capacity_after_all_fallbacks(self):
        busy = unittest.mock.Mock(status_code=503)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "local-test-key"}), patch.object(
            agents.requests, "post", return_value=busy
        ) as request:
            with self.assertRaisesRegex(agents.AgentError, "temporarily at high capacity"):
                agents._generate_json("test prompt")

        self.assertEqual(request.call_count, len(agents.GEMINI_MODELS))

    def test_extract_claims_trims_and_deduplicates(self):
        with patch.object(agents, "_generate_json", return_value={"claims": [" Claim A ", "Claim A", "Claim B"]}):
            self.assertEqual(agents.extract_claims("Some text"), ["Claim A", "Claim B"])

    def test_fact_check_uses_unverifiable_for_unknown_verdict(self):
        evidence = [{"title": "Source", "url": "https://example.com", "content": "Excerpt", "search_direction": "supporting"}]
        with patch.object(agents, "_search_both_sides", return_value=evidence), patch.object(agents, "_find_related_checks", return_value=[]), patch.object(
            agents,
            "_generate_json",
            return_value={"verdict": "Likely true", "confidence": 140, "summary": "Not certain", "needs_more_evidence": False},
        ):
            result = agents.fact_check_claim("A test claim")

        self.assertEqual(result["verdict"], "Unverifiable")
        self.assertEqual(result["confidence"], 100)
        self.assertEqual(result["sources"], evidence)

    def test_markdown_export_includes_source_link(self):
        report = agents.export_markdown([{
            "claim": "A test claim",
            "verdict": "True",
            "confidence": 90,
            "sources": [{"title": "Source", "url": "https://example.com", "search_direction": "supporting"}],
        }])

        self.assertIn("A test claim", report)
        self.assertIn("[Source](https://example.com)", report)
        self.assertIn("90%", report)

    def test_saved_claims_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(agents, "DATA_DIR", root), patch.object(agents, "CLAIMS_FILE", root / "claims.json"):
                agents.save_checked_claims("science", [{"claim": "A claim", "verdict": "Unverifiable", "checked_at": "2026-09-28T00:00:00+00:00"}])
                saved = agents.get_saved_claims()

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["topic"], "science")
        self.assertEqual(saved[0]["claim"], "A claim")


if __name__ == "__main__":
    unittest.main()