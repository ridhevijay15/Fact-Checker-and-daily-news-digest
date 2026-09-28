import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# --------------------------------------------------
# AGENT 1: NEWS COLLECTOR
# --------------------------------------------------

def news_collector_agent(topic):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": topic,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return []

    data = response.json()

    articles = []

    for article in data.get("articles", []):

        articles.append({
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "source": article.get("source", {}).get("name"),
            "url": article.get("url")
        })

    return articles


# --------------------------------------------------
# AGENT 2: FACT CHECKER
# --------------------------------------------------

def fact_checker_agent(articles):

    if not articles:
        return "No news articles were found."

    article_text = ""

    for i, article in enumerate(articles, 1):

        article_text += f"""
ARTICLE {i}

Title: {article['title']}
Source: {article['source']}
Description: {article['description']}
Content: {article['content']}
URL: {article['url']}

-----------------------
"""

    prompt = f"""
You are a Fact Checking Agent.

Analyze the following news articles.

Your tasks:

1. Identify the major factual claims.
2. Compare claims across the different sources.
3. Determine whether the claims are:
   - VERIFIED
   - CONTRADICTED
   - INSUFFICIENT EVIDENCE
4. Explain your reasoning briefly.
5. Mention which sources support each claim.

IMPORTANT:
Do not invent facts or sources.
Only use information present in the provided articles.

NEWS ARTICLES:

{article_text}
"""

    response = client.chat.completions.create(
        model="gpt-5.6",
        messages=[
            {
                "role": "system",
                "content": "You are a careful news fact-checking agent."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# --------------------------------------------------
# AGENT 3: DIGEST GENERATOR
# --------------------------------------------------

def digest_agent(articles, fact_check):

    article_text = ""

    for article in articles:

        article_text += f"""
Title: {article['title']}
Source: {article['source']}
Description: {article['description']}
URL: {article['url']}

"""

    prompt = f"""
You are a Daily News Digest Agent.

Create a concise and readable news digest using the articles
and fact-checking report below.

For every important story provide:

📰 Headline
📝 Short Summary
🔎 Fact-check Status
📚 Sources

Do not invent information.

NEWS:

{article_text}

FACT CHECK REPORT:

{fact_check}
"""

    response = client.chat.completions.create(
        model="gpt-5.6",
        messages=[
            {
                "role": "system",
                "content": "You create concise and factual news digests."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# --------------------------------------------------
# ORCHESTRATOR
# --------------------------------------------------

def run_news_agents(topic):

    articles = news_collector_agent(topic)

    if not articles:
        return [], "No articles found.", "No digest available."

    fact_check = fact_checker_agent(articles)

    digest = digest_agent(
        articles,
        fact_check
    )

    return articles, fact_check, digest