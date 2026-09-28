import streamlit as st
import requests
import os
from dotenv import load_dotenv
import google.generativeai as genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY")


# ============================================================
# CHECK API KEYS
# ============================================================

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is missing from .env")

if not NEWSDATA_API_KEY:
    st.error("❌ NEWSDATA_API_KEY is missing from .env")


# ============================================================
# CONFIGURE GEMINI
# ============================================================

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )


# ============================================================
# AGENT 1 — NEWS COLLECTOR
# ============================================================

def news_collector_agent(topic):

    url = "https://newsdata.io/api/1/latest"

    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": topic,
        "language": "en"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        # Debug information
        print("NewsData status:", response.status_code)
        print("NewsData response:", response.text)

        # API error
        if response.status_code != 200:

            st.error(
                f"❌ NewsData.io API Error: "
                f"{response.status_code}"
            )

            st.code(response.text)

            return []

        data = response.json()

        # Check API response
        if data.get("status") == "error":

            st.error("❌ NewsData.io returned an error.")

            st.code(str(data))

            return []

        articles = []

        results = data.get("results", [])

        for article in results[:5]:

            articles.append({

                "title": article.get(
                    "title",
                    "No title available"
                ),

                "description": article.get(
                    "description",
                    "No description available"
                ),

                "source": article.get(
                    "source_name",
                    "Unknown source"
                ),

                "url": article.get(
                    "link",
                    ""
                )
            })

        return articles

    except requests.exceptions.Timeout:

        st.error(
            "⏱️ NewsData.io request timed out. "
            "Please try again."
        )

        return []

    except requests.exceptions.ConnectionError:

        st.error(
            "🌐 Could not connect to NewsData.io. "
            "Check your internet connection."
        )

        return []

    except Exception as e:

        st.error(
            f"❌ Unexpected NewsData error: {e}"
        )

        return []


# ============================================================
# AGENT 2 — FACT CHECKER
# ============================================================

def fact_checker_agent(articles):

    if not articles:

        return "No articles available for fact checking."


    # Combine articles
    news_text = ""

    for i, article in enumerate(articles, 1):

        news_text += f"""

ARTICLE {i}

Title:
{article["title"]}

Source:
{article["source"]}

Description:
{article["description"]}

URL:
{article["url"]}

----------------------------------------
"""


    prompt = f"""
You are the Fact-Checking Agent of an Agentic AI system.

Analyze the news articles provided below.

Your tasks:

1. Identify the major factual claims.
2. Compare information between the available articles.
3. Classify each claim as one of:
   VERIFIED
   CONTRADICTED
   INSUFFICIENT EVIDENCE

4. Give a short explanation.
5. Mention the sources supporting the claim.

IMPORTANT RULES:

- Do not invent facts.
- Do not invent sources.
- Use only the information provided.
- If the evidence is insufficient, say so.

NEWS ARTICLES:

{news_text}
"""


    try:

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:

        return f"""
❌ Gemini Fact Checker Error:

{str(e)}
"""


# ============================================================
# AGENT 3 — DAILY DIGEST GENERATOR
# ============================================================

def digest_agent(articles, fact_check):

    if not articles:

        return "No news available."


    news_text = ""

    for article in articles:

        news_text += f"""

Title:
{article["title"]}

Source:
{article["source"]}

Description:
{article["description"]}

URL:
{article["url"]}

"""


    prompt = f"""
You are the Daily News Digest Agent.

Create a concise and readable news digest.

For each important story include:

📰 HEADLINE

📝 SUMMARY

🔎 FACT-CHECK STATUS

📚 SOURCE

Use the fact-checking report provided below.

Possible fact-check statuses:

VERIFIED
CONTRADICTED
INSUFFICIENT EVIDENCE

Do not invent information.

NEWS:

{news_text}

FACT-CHECK REPORT:

{fact_check}
"""


    try:

        response = model.generate_content(prompt)

        return response.text

    except Exception as e:

        return f"""
❌ Gemini Digest Error:

{str(e)}
"""


# ============================================================
# ORCHESTRATOR
# ============================================================

def run_agents(topic):

    # ------------------------------------
    # STEP 1
    # ------------------------------------

    st.info(
        "🔎 Agent 1: Collecting latest news..."
    )

    articles = news_collector_agent(topic)


    if not articles:

        return [], "No articles found.", "No digest available."


    # ------------------------------------
    # STEP 2
    # ------------------------------------

    st.info(
        "🛡️ Agent 2: Fact-checking information..."
    )

    fact_check = fact_checker_agent(
        articles
    )


    # ------------------------------------
    # STEP 3
    # ------------------------------------

    st.info(
        "📝 Agent 3: Generating news digest..."
    )

    digest = digest_agent(
        articles,
        fact_check
    )


    return articles, fact_check, digest


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Fact-Checker & Daily News Digest",

    page_icon="📰",

    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📰 Fact-Checker & Daily News Digest"
)

st.write(
    "An Agentic AI system that collects news, "
    "fact-checks information, and generates "
    "a concise daily digest."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🤖 Agentic AI Pipeline"
)

st.sidebar.markdown(
"""
### 🔎 Agent 1
**News Collector Agent**

Collects recent news using NewsData.io.

---

### 🛡️ Agent 2
**Fact Checker Agent**

Analyzes claims and compares available sources using Gemini.

---

### 📝 Agent 3
**Digest Generator Agent**

Creates the final concise news digest.

---

### 🔄 Orchestrator

Coordinates the complete agent workflow.
"""
)


# ============================================================
# USER INPUT
# ============================================================

topic = st.text_input(

    "🔍 Enter a news topic",

    placeholder="Example: Artificial Intelligence"
)


# ============================================================
# BUTTON
# ============================================================

generate = st.button(

    "🚀 Generate News Digest",

    type="primary"
)


# ============================================================
# RUN AGENTS
# ============================================================

if generate:

    if not topic.strip():

        st.warning(
            "⚠️ Please enter a news topic."
        )

    elif not GEMINI_API_KEY or not NEWSDATA_API_KEY:

        st.error(
            "❌ Please check your API keys in the .env file."
        )

    else:

        articles, fact_check, digest = run_agents(
            topic
        )


        # ====================================================
        # COLLECTED NEWS
        # ====================================================

        st.header(
            "📰 Collected News"
        )


        if articles:

            for article in articles:

                st.subheader(
                    article["title"]
                )

                st.write(
                    f"**Source:** {article['source']}"
                )

                if article["description"]:

                    st.write(
                        article["description"]
                    )

                if article["url"]:

                    st.markdown(
                        f"🔗 [Read Original Article]"
                        f"({article['url']})"
                    )

                st.divider()

        else:

            st.warning(
                "No news articles were found."
            )


        # ====================================================
        # FACT CHECK
        # ====================================================

        st.header(
            "🛡️ Fact-Checking Report"
        )

        st.markdown(
            fact_check
        )


        # ====================================================
        # DAILY DIGEST
        # ====================================================

        st.header(
            "📋 Daily News Digest"
        )

        st.markdown(
            digest
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Agentic AI Pipeline: "
    "News Collection → Fact Checking → "
    "Digest Generation"
)