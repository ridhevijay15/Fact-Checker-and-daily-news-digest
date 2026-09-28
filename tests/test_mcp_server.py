import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agents
import mcp_server
from mcp.shared.memory import create_connected_server_and_client_session


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_tools_save_retrieve_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(agents, "DATA_DIR", root),
                patch.object(agents, "CLAIMS_FILE", root / "claims.json"),
                patch.object(agents, "DIGESTS_FILE", root / "digests.json"),
                patch.object(mcp_server, "EXPORT_DIR", root / "exports"),
            ):
                agents.save_checked_claims("science", [{
                    "claim": "A sample claim",
                    "verdict": "True",
                    "confidence": 90,
                    "checked_at": "2026-09-28T00:00:00+00:00",
                    "sources": [],
                }])
                async with create_connected_server_and_client_session(mcp_server.mcp) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.assertEqual(
                        {tool.name for tool in tools.tools},
                        {"save_digest", "get_saved_claims", "export_markdown"},
                    )

                    saved = await session.call_tool("save_digest", {
                        "topic": "science",
                        "digest": "# Science digest\n\nA sourced story.",
                    })
                    self.assertFalse(saved.isError)

                    claims = await session.call_tool("get_saved_claims", {
                        "topic": "science",
                        "verdict": "True",
                    })
                    self.assertFalse(claims.isError)
                    self.assertIn("A sample claim", claims.content[0].text)

                    exported = await session.call_tool("export_markdown", {
                        "record_type": "digest",
                        "filename": "science-digest.md",
                    })
                    self.assertFalse(exported.isError)

                output = root / "exports" / "science-digest.md"
                self.assertTrue(output.exists())
                self.assertIn("# Science digest", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()