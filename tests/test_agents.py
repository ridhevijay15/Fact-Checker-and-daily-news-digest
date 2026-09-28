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
            agents.requests, "post", side_effect=[busy, busy, ready]
        ) as request, patch.object(agents.time, "sleep") as wait:
            result = agents._generate_json("test prompt")

        self.assertEqual(result, {"ok": True})
        self.assertIn("gemini-3.8-flash", request.call_args_list[0].args[0])
        self.assertIn("gemini-3.8-flash", request.call_args_list[1].args[0])
        self.assertIn("gemini-3.7-flash", request.call_args_list[2].args[0])
        wait.assert_called_once_with(1.0)

    def test_gemini_reports_temporary_capacity_after_all_fallbacks(self):
        busy = unittest.mock.Mock(status_code=503)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "local-test-key"}), patch.object(
            agents.requests, "post", return_value=busy
        ) as request, patch.object(agents.time, "sleep"):
            with self.assertRaisesRegex(agents.AgentError, "temporarily at high capacity"):
                agents._generate_json("test prompt")

        self.assertEqual(request.call_count, len(agents.GEMINI_MODELS) + 1)

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
        self.assertEqual(result["sources"][0]["url"], evidence[0]["url"])
        self.assertIsNone(result["sources"][0]["trusted_source"])

    def test_trusted_source_match_rejects_lookalike_domains(self):
        source = agents._trusted_source_for_url("https://www.who.int/news/example")

        self.assertEqual(source["name"], "World Health Organization")
        self.assertIn("publication date", source["credibility_note"])
        self.assertIsNone(agents._trusted_source_for_url("https://who.int.attacker.example/news"))

    def test_judge_receives_trusted_source_and_retrieved_archive_context(self):
        prompts = []
        archived = [{
            "claim": "Coffee drinking is associated with mortality",
            "verdict": "Unverifiable",
            "summary": "Prior check found mixed evidence.",
            "checked_at": "2026-09-28T00:00:00+00:00",
            "sources": [{"title": "Study", "url": "https://pubmed.ncbi.nlm.nih.gov/123", "search_direction": "supporting"}],
        }]
        evidence = [{"title": "Health report", "url": "https://who.int/report", "content": "Study summary", "search_direction": "supporting"}]

        with patch.object(agents, "_generate_json", side_effect=lambda prompt: prompts.append(prompt) or {"verdict": "Unverifiable"}):
            agents._judge_claim("Coffee claim", evidence, archived)

        self.assertIn("World Health Organization", prompts[0])
        self.assertIn("credibility_note", prompts[0])
        self.assertIn("Prior check found mixed evidence.", prompts[0])
        self.assertIn("https://pubmed.ncbi.nlm.nih.gov/123", prompts[0])

    def test_archive_retrieval_ranks_relevant_checks_with_bm25(self):
        archive = [
            {"claim": "Coffee drinking and mortality risk", "summary": "A short prior review.", "checked_at": "2026-01-01"},
            {"claim": "Coffee farming and crop yield", "summary": "Coffee farms grow beans.", "checked_at": "2026-02-01"},
            {"claim": "Mars has two moons", "summary": "A space science fact.", "checked_at": "2026-03-01"},
        ]
        with patch.object(agents, "get_saved_claims", return_value=archive):
            matches = agents._find_related_checks("coffee mortality risk study")

        self.assertEqual(matches[0]["claim"], "Coffee drinking and mortality risk")
        self.assertNotIn(archive[2], matches)

    def test_claim_batch_graph_extracts_checks_and_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = [
                {"claim": "Claim one", "verdict": "True", "confidence": 90, "checked_at": "2026-09-28T00:00:00+00:00"},
                {"claim": "Claim two", "verdict": "Unverifiable", "confidence": 0, "checked_at": "2026-09-28T00:00:00+00:00"},
            ]
            with patch.object(agents, "DATA_DIR", root), patch.object(agents, "CLAIMS_FILE", root / "claims.json"), patch.object(
                agents, "extract_claims", return_value=["Claim one", "Claim two"]
            ), patch.object(agents, "fact_check_claim", side_effect=results):
                batch = agents.run_claim_checks("Two factual claims", "science")
                archived = agents.get_saved_claims()

        self.assertEqual(batch["claims"], ["Claim one", "Claim two"])
        self.assertEqual(len(batch["results"]), 2)
        self.assertEqual(len(archived), 2)
        self.assertTrue(all(item["topic"] == "science" for item in archived))

    def test_daily_digest_graph_collects_checks_writes_and_archives(self):
        article = {"title": "A story", "description": "A detail", "source": "Example", "url": "https://example.com", "published_at": "2026-09-28"}
        check = {"claim": "A story. A detail", "verdict": "True", "confidence": 90, "summary": "Supported", "sources": [], "checked_at": "2026-09-28T00:00:00+00:00"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(agents, "DATA_DIR", root), patch.object(agents, "CLAIMS_FILE", root / "claims.json"), patch.object(
                agents, "DIGESTS_FILE", root / "digests.json"
            ), patch.object(agents, "news_collector_agent", return_value=[article]), patch.object(
                agents, "fact_check_claim", return_value=check
            ), patch.object(agents, "digest_agent", return_value="# Daily digest\n\nA verified story."):
                record = agents.run_daily_digest("science", limit=1)
                saved_digests = agents.get_saved_digests()
                saved_claims = agents.get_saved_claims()

        self.assertEqual(record["digest"], "# Daily digest\n\nA verified story.")
        self.assertEqual(len(saved_digests), 1)
        self.assertEqual(len(saved_claims), 1)

    def test_langgraph_runs_at_most_two_followup_retrieval_rounds(self):
        judgment = {"verdict": "Unverifiable", "confidence": 20, "summary": "More evidence needed.", "needs_more_evidence": True}
        critic = {"needs_more_evidence": True, "followup_queries": ["independent follow-up"]}
        responses = [judgment, critic, judgment, critic, judgment, critic]
        initial = [{"title": "Initial", "url": "https://example.com/initial", "content": "Evidence", "search_direction": "supporting"}]

        def search_result(query, max_results):
            index = search.call_count
            return [{"title": f"Follow-up {index}", "url": f"https://example.com/followup-{index}", "content": "More evidence"}]

        with patch.object(agents, "_search_both_sides", return_value=initial), patch.object(
            agents, "_find_related_checks", return_value=[]
        ), patch.object(agents, "_generate_json", side_effect=responses) as generate, patch.object(
            agents, "search_web", side_effect=search_result
        ) as search:
            result = agents.fact_check_claim("A claim needing review")

        self.assertEqual(generate.call_count, 6)
        self.assertEqual(search.call_count, agents.MAX_CRITIC_ROUNDS)
        self.assertEqual(len(result["sources"]), 3)

    def test_critic_can_request_evidence_after_judge_is_done(self):
        responses = [
            {"verdict": "True", "confidence": 80, "summary": "Supported.", "needs_more_evidence": False},
            {"needs_more_evidence": True, "followup_queries": ["independent confirmation"]},
            {"verdict": "True", "confidence": 90, "summary": "Corroborated.", "needs_more_evidence": False},
            {"needs_more_evidence": False, "followup_queries": []},
        ]
        initial = [{"title": "WHO report", "url": "https://who.int/report", "content": "Guidance", "search_direction": "supporting"}]
        followup = [{"title": "Reuters report", "url": "https://reuters.com/fact-check/story", "content": "Independent report"}]

        with patch.object(agents, "_search_both_sides", return_value=initial), patch.object(
            agents, "_find_related_checks", return_value=[]
        ), patch.object(agents, "_generate_json", side_effect=responses) as generate, patch.object(
            agents, "search_web", return_value=followup
        ) as search:
            result = agents.fact_check_claim("A supported health claim")

        self.assertEqual(generate.call_count, 4)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(result["sources"][0]["trusted_source"]["name"], "World Health Organization")
        self.assertEqual(result["sources"][1]["trusted_source"]["name"], "Reuters")

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