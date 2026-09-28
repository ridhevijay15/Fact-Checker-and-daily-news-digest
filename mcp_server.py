from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from agents import (
    export_markdown as render_claims_markdown,
    get_saved_claims as read_saved_claims,
    get_saved_digests,
    save_digest as store_digest,
)

mcp = FastMCP("fact-check-daily-digest")
EXPORT_DIR = Path(__file__).resolve().parent / "exports"


@mcp.tool()
def save_digest(
    topic: str,
    digest: str,
    articles: list[dict[str, Any]] | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Save a compiled news digest with its topic, articles, and fact checks."""
    if not topic.strip():
        raise ValueError("Topic must not be empty.")
    if not digest.strip():
        raise ValueError("Digest must not be empty.")
    record = store_digest(topic.strip(), digest.strip(), articles or [], checks or [])
    return {"id": record["id"], "topic": record["topic"], "date": record["date"]}


@mcp.tool()
def get_saved_claims(topic: str | None = None, verdict: str | None = None) -> list[dict[str, Any]]:
    """Return saved fact checks, optionally filtered by topic text and verdict."""
    claims = read_saved_claims()
    if topic:
        topic_filter = topic.casefold()
        claims = [claim for claim in claims if topic_filter in str(claim.get("topic", "")).casefold()]
    if verdict:
        verdict_filter = verdict.casefold()
        claims = [claim for claim in claims if verdict_filter == str(claim.get("verdict", "")).casefold()]
    return claims


@mcp.tool()
def export_markdown(
    record_type: Literal["claims", "digest"] = "claims",
    record_id: str | None = None,
    filename: str | None = None,
    title: str | None = None,
) -> dict[str, str]:
    """Export saved claim checks or a saved digest to a Markdown file under exports/."""
    if record_type == "claims":
        records = read_saved_claims()
        if record_id:
            records = [record for record in records if record.get("id") == record_id]
        if not records:
            raise ValueError("No matching saved claim checks were found.")
        content = render_claims_markdown(records, title or "Fact-check report")
        default_name = "fact-check-report.md"
    else:
        records = get_saved_digests()
        if record_id:
            records = [record for record in records if record.get("id") == record_id]
        if not records:
            raise ValueError("No matching saved digests were found.")
        record = records[0]
        content = str(record.get("digest", "")).strip()
        if not content:
            raise ValueError("The saved digest contains no Markdown content.")
        default_name = f"news-digest-{record.get('date', 'saved')}.md"

    requested_name = filename or default_name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(requested_name).name).strip(".-_")
    if not safe_name:
        raise ValueError("Filename must contain letters or numbers.")
    if not safe_name.lower().endswith(".md"):
        safe_name += ".md"

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXPORT_DIR / safe_name
    output_path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return {"path": str(output_path), "filename": safe_name, "record_type": record_type}


if __name__ == "__main__":
    mcp.run(transport="stdio")