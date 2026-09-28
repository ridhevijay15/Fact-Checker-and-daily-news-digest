# Fact-Checker & Daily News Digest

An agentic Streamlit application for checking factual claims against live sources and compiling sourced daily news digests.

## Project overview

The app combines recent news retrieval, AI-assisted claim extraction, web evidence searches, cautious verdicts, and source-linked summaries. A local archive stores checked claims and generated digests for later reference.

## Architecture

```text
Claim-check LangGraph:
Claim Extractor -> Claim Checker -> Fact-check subgraph -> Claim Archive
              Evidence Retriever -> Verdict Judge -> Critic
                ^                               |
                +--- at most two searches ------+

Daily-digest LangGraph:
NewsData Collector -> Story Fact Checker -> Digest Writer -> Digest Archive

Retrieval context: BM25-ranked saved checks + trusted-source domain notes
MCP tools: save_digest, get_saved_claims, export_markdown
```

Verdicts are `True`, `False`, `Misleading`, or `Unverifiable`. Missing or inconclusive evidence is reported as `Unverifiable`, not guessed. Archived checks are historical context, not a substitute for current sources.

## RAG knowledge base

The retriever uses BM25 to rank previously checked claims across claim text, summary, and topic, then supplies relevant verdicts, dates, and source links as historical context to the judge. The curated `knowledge/trusted_sources.json` registry records publisher domains, source types, and scope/limitation notes. Matching registry entries are included with live evidence and shown with citations. Registry notes are review guidance, not proof or an automatic reliability score; maintain the list as sources and coverage change.

## Requirements

- Python 3.10 or newer
- A Gemini API key
- A Tavily API key
- A NewsData.io API key for daily digests

## Setup on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your keys to `.env`. The app reads `GEMINI_API_KEY`, `TAVILY_API_KEY`, and `NEWSDATA_API_KEY`; the `.env` file is ignored by Git.

## Run

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit.

## MCP server

The project includes a separate MCP server using stdio transport. It exposes three tools:

- `save_digest`: store a digest and its topic, articles, and checks in the local archive.
- `get_saved_claims`: retrieve saved checks, optionally filtering by topic and verdict.
- `export_markdown`: write a saved claim report or digest to `exports/`.

Run it directly with the project environment:

```powershell
.\.venv\Scripts\python.exe mcp_server.py
```

To connect it from VS Code, add a server entry to `.vscode/mcp.json` (adjust the paths if the project is elsewhere):

```json
{
  "servers": {
    "fact-check-digest": {
      "type": "stdio",
      "command": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "args": ["${workspaceFolder}\\mcp_server.py"]
    }
  }
}
```

## Workflows

- **Fact-check claims:** paste a statement or forwarded message; Gemini separates factual claims, Tavily searches for supporting and challenging evidence, and a bounded critic pass can request more sources. Each result has a conservative verdict, confidence, evidence summary, and source links.
- **Daily digest:** NewsData.io retrieves recent articles for a topic. Each story is checked against live web evidence before Gemini compiles a dated Markdown digest.
- **Saved work:** checked claims and digests are stored locally under `data/`, can be searched as historical context, and can be exported as Markdown.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests use mocked model responses, local temporary files, and an in-memory MCP client/server session; they do not make external API calls.
