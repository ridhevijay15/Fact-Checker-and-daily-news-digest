# 📰 Fact-Checker & Daily News Digest

## 📌 Project Overview

Fact-Checker & Daily News Digest is an Agentic AI application that collects recent news based on a user-provided topic, analyzes the information using multiple AI agents, performs cross-source fact checking, and generates a concise news digest.

The system combines news retrieval and generative AI to automate the process of collecting, analyzing, verifying, and summarizing news.

---

## 🎯 Objectives

- Collect recent news based on a user-selected topic.
- Analyze information from multiple news articles.
- Identify and compare important factual claims.
- Classify claims as Verified, Contradicted, or Insufficient Evidence.
- Generate a concise and readable daily news digest.
- Provide links to the original news sources.

---

## 🤖 Agentic AI Architecture

The project uses multiple specialized agents coordinated through an orchestrated workflow.

```text
                    USER
                      │
                      ▼
              ┌──────────────┐
              │ ORCHESTRATOR │
              └──────┬───────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  NEWS COLLECTOR AGENT  │
        └────────────┬───────────┘
                     │
                     ▼
                NewsData.io
                     │
                     ▼
        ┌────────────────────────┐
        │   FACT CHECKER AGENT   │
        └────────────┬───────────┘
                     │
                     ▼
                   Gemini
                     │
                     ▼
        ┌────────────────────────┐
        │  DIGEST GENERATOR AGENT│
        └────────────┬───────────┘
                     │
                     ▼
              FINAL NEWS DIGEST
