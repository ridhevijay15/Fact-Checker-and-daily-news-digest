from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import streamlit as st

from agents import (
    AgentError,
    configured_keys,
    export_markdown,
    get_saved_claims,
    get_saved_digests,
    run_daily_digest,
    run_claim_checks,
)

st.set_page_config(
    page_title="Fact-check desk",
    page_icon=":material/fact_check:",
    layout="wide",
)

st.title("Fact-check desk")
st.caption("Check claims against live sources. Build and save a sourced daily news digest.")

with st.sidebar:
    st.subheader("Connected services")
    for key_name, available in configured_keys().items():
        label = key_name.removesuffix("_API_KEY").replace("_", " ").title()
        st.write(f"{'Ready' if available else 'Missing'} · {label}")
    st.caption("Keys are loaded from your local .env file and are never shown here.")

def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No live sources were available for this check.")
        return
    for source in sources:
        direction = source.get("search_direction", "evidence")
        st.markdown(f"[{source.get('title', 'Source')}]({source.get('url', '')})")
        date_label = f" · {source['published_date']}" if source.get("published_date") else ""
        st.caption(f"{direction.title()} search{date_label}")
        registry_entry = source.get("trusted_source")
        if registry_entry:
            st.caption(
                f"Registry: {registry_entry['name']} · {registry_entry['type']}. "
                f"{registry_entry['credibility_note']}"
            )
        else:
            st.caption("Not listed in the trusted-source registry; assess using evidence and corroboration.")
        if source.get("content"):
            st.write(source["content"])


def render_check(result: dict[str, Any]) -> None:
    st.subheader(result.get("claim", "Claim"))
    first, second = st.columns(2)
    first.metric("Verdict", result.get("verdict", "Unverifiable"))
    second.metric("Confidence", f"{result.get('confidence', 0)}%")
    st.write(result.get("summary", ""))
    supporting = result.get("supporting_evidence", [])
    challenging = result.get("challenging_evidence", [])
    if supporting:
        st.markdown("**Evidence supporting the claim**")
        for item in supporting:
            st.write(f"- {item}")
    if challenging:
        st.markdown("**Evidence challenging the claim**")
        for item in challenging:
            st.write(f"- {item}")
    with st.expander(f"Sources ({len(result.get('sources', []))})"):
        render_sources(result.get("sources", []))
    if result.get("archive_matches"):
        with st.expander("Related checks in the archive"):
            for archived in result["archive_matches"]:
                st.write(f"**{archived.get('verdict', 'Unverifiable')}** · {archived.get('claim', '')}")
                st.caption("Historical check; not used as live evidence.")


fact_tab, digest_tab, archive_tab = st.tabs(["Fact-check claims", "Daily digest", "Saved work"])

with fact_tab:
    st.subheader("Check a claim or forwarded message")
    with st.form("claim_check_form"):
        claim_text = st.text_area(
            "Text to check",
            placeholder="Paste a claim, article excerpt, or forwarded message. The checker will separate it into factual claims.",
            height=160,
        )
        claim_topic = st.text_input("Topic label (optional)", placeholder="Health, elections, science…")
        submitted = st.form_submit_button(
            "Extract claims and check evidence",
            type="primary",
            icon=":material/search:",
        )

    if submitted:
        missing = [name for name in ("GEMINI_API_KEY", "TAVILY_API_KEY") if not configured_keys()[name]]
        if missing:
            st.error("Configure " + " and ".join(missing) + " in .env before checking claims.")
        elif not claim_text.strip():
            st.warning("Paste some text or enter a claim first.")
        else:
            st.session_state["claim_results"] = []
            try:
                with st.status("Extracting and checking claims", expanded=True) as progress:
                    batch = run_claim_checks(claim_text, claim_topic.strip() or "Pasted text")
                    claims = batch.get("claims", [])
                    if not claims:
                        progress.update(label="No checkable factual claims found", state="complete")
                        st.info("Try a message containing a specific factual statement.")
                    else:
                        results = batch.get("results", [])
                        st.session_state["claim_results"] = results
                        progress.update(label=f"Checked and saved {len(results)} claim(s)", state="complete")
            except AgentError as error:
                st.error(str(error))

    for result in st.session_state.get("claim_results", []):
        st.divider()
        render_check(result)

with digest_tab:
    st.subheader("Compile a verified news digest")
    st.write("NewsData gathers recent articles; Tavily finds evidence for and against each story; Gemini writes the sourced digest.")
    with st.form("daily_digest_form"):
        topic = st.text_input("News topic", placeholder="Technology, public health, climate…")
        article_limit = st.slider("Number of stories", min_value=1, max_value=5, value=3)
        digest_submitted = st.form_submit_button(
            "Build today's digest",
            type="primary",
            icon=":material/newspaper:",
        )

    if digest_submitted:
        required_keys = ("GEMINI_API_KEY", "NEWSDATA_API_KEY", "TAVILY_API_KEY")
        missing = [name for name in required_keys if not configured_keys()[name]]
        if missing:
            st.error("Configure " + ", ".join(missing) + " in .env before building a digest.")
        elif not topic.strip():
            st.warning("Enter a topic for today's digest.")
        else:
            try:
                with st.status("Building the daily digest", expanded=True) as progress:
                    progress.update(label="Collecting articles and checking each story against live sources")
                    record = run_daily_digest(topic.strip(), article_limit)
                    st.session_state["latest_digest"] = record
                    progress.update(label="Digest checked and saved", state="complete")
            except AgentError as error:
                st.error(str(error))

    latest = st.session_state.get("latest_digest")
    if latest:
        st.divider()
        st.markdown(latest["digest"])
        st.download_button(
            "Download digest as Markdown",
            data=latest["digest"],
            file_name=f"news-digest-{date.today().isoformat()}.md",
            mime="text/markdown",
            icon=":material/download:",
        )
        with st.expander(f"Reviewed stories ({len(latest.get('articles', []))})"):
            for article, check in zip(latest.get("articles", []), latest.get("checks", [])):
                st.markdown(f"**{article['title']}** · {check['verdict']} · {check['confidence']}%")
                if article.get("url"):
                    st.markdown(f"[Original article]({article['url']})")
                render_sources(check.get("sources", []))

with archive_tab:
    st.subheader("Saved checks and digests")
    saved_claims = get_saved_claims()
    saved_digests = get_saved_digests()
    st.caption(f"{len(saved_claims)} saved claim checks · {len(saved_digests)} saved digests")

    this_week = st.checkbox("Show checks from the last 7 days only")
    if this_week:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        saved_claims = [
            record for record in saved_claims
            if record.get("checked_at")
            and datetime.fromisoformat(record["checked_at"].replace("Z", "+00:00")) >= cutoff
        ]

    if saved_claims:
        claim_ids = [record["id"] for record in saved_claims]
        selected_claim_id = st.selectbox(
            "Saved claim checks",
            claim_ids,
            format_func=lambda item_id: next(
                f"{record.get('verdict', 'Unverifiable')} · {record.get('claim', '')[:100]}"
                for record in saved_claims if record["id"] == item_id
            ),
        )
        selected_claim = next(record for record in saved_claims if record["id"] == selected_claim_id)
        render_check(selected_claim)
        st.download_button(
            "Export selected check",
            data=export_markdown([selected_claim]),
            file_name="fact-check-report.md",
            mime="text/markdown",
            icon=":material/download:",
        )
    else:
        st.info("No claim checks found in this date range.")

    if saved_digests:
        digest_ids = [record["id"] for record in saved_digests]
        selected_digest_id = st.selectbox(
            "Saved digests",
            digest_ids,
            format_func=lambda item_id: next(
                f"{record.get('date', '')} · {record.get('topic', '')}"
                for record in saved_digests if record["id"] == item_id
            ),
        )
        selected_digest = next(record for record in saved_digests if record["id"] == selected_digest_id)
        st.markdown(selected_digest.get("digest", ""))
        st.download_button(
            "Export selected digest",
            data=selected_digest.get("digest", ""),
            file_name=f"news-digest-{selected_digest.get('date', date.today().isoformat())}.md",
            mime="text/markdown",
            icon=":material/download:",
        )
    else:
        st.info("Daily digests you build will appear here.")