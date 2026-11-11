<div align="center">

# Axiom Dispatch — AI Research Intelligence Aggregator
### Autonomous Multi-Source Research Synthesis, Spoken Transcript Mining & IEEE Briefing Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL 16](https://img.shields.io/badge/postgresql-16-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![LLM: Gemini 3.5 / 3 Flash & OpenAI](https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenAI-4285F4.svg?logo=google&logoColor=white)](https://aistudio.google.com/)
[![Tests: 122 Passed](https://img.shields.io/badge/tests-122%20passed-success.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*An autonomous multi-agent pipeline that scrapes frontier AI lab feeds and technical YouTube video transcripts, stores structured literature in PostgreSQL, synthesizes deep technical digests with Gemini / OpenAI, ranks research against each subscriber's profile, and delivers responsive IEEE-styled HTML emails with attached A4 PDF compendiums.*

</div>

---

## Branches at a Glance

This repository maintains two dedicated, production-hardened branches for different runtime environments:

| Feature / Aspect | `main` Branch (Local Computer Pipeline) | `deployment-part-2` Branch (Render Cloud Deployment) |
|:---|:---|:---|
| **Designed For** | Running full end-to-end briefings directly from your local workstation | Zero-maintenance cloud hosting & 24-hour scheduling on Render |
| **Primary Email Transport** | Gmail SMTP TLS with automatic port failover (`587` STARTTLS -> `465` SSL) | Resend HTTPS API (`443`) primary (avoids cloud outbound SMTP blocks) |
| **Backup Email Transport** | Automatic Resend HTTPS API (`443`) fallback if all SMTP ports are blocked and `RESEND_API_KEY` is set | Automatic Gmail SMTP (`587`/`465`) fallback if `RESEND_API_KEY` is unset |
| **One-Command Dispatch** | `uv run python scripts/manual_send.py --full` & `--preflight` | Portal UI **Trigger Pipeline Now** button + 24h APScheduler |
| **Infrastructure Config** | Docker PostgreSQL (`docker-compose.yml`) + local `.env` | Managed Render Blueprint (`render.yaml`: Docker Web Service + PostgreSQL 16) |
| **Web Portal Security** | Auto-selects free local port (`8000`–`8010`) | Automatically redacts subscriber PII on public `GET /api/config` (`RENDER=true`) |
| **YouTube Cloud Blocking Mitigation** | Runs from residential/local IP directly | Supports optional `YOUTUBE_PROXY` env var for datacenter IPs |

---

## System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Sourcing & Ingestion"]
        YT["YouTube Technical Channels (RSS)"] --> Runner["Ingestion Runner (app/scrapers/runner.py)"]
        OA["OpenAI News & Research RSS"] --> Runner
        ANT["Anthropic News & Engineering RSS"] --> Runner
        Runner --> Extract["Content & Transcript Extraction (yt-dlp, youtube-transcript-api, r.jina.ai fallback)"]
        Extract --> ArticlesDB[("PostgreSQL: articles")]
    end

    subgraph Synthesis["LLM Synthesis & Storage"]
        ArticlesDB --> Summarizer["DigestAgent (app/agent/summarizer.py)"]
        Summarizer --> Gemini["DualLLMClient (Gemini model cascade -> OpenAI fallback)"]
        Gemini --> DigestDB[("PostgreSQL: digest")]
    end

    subgraph Personalization["Multi-Agent Ranking"]
        DigestDB --> Ranker["UserRankingAgent (app/agent/ranking_agent.py)"]
        SubDB[("PostgreSQL: subscribers")] --> Ranker
        Ranker --> Tier1["Tier 1: Featured Strategic Deep Dives (max 2/source)"]
        Ranker --> Tier2["Tier 2: Ecosystem Radar & Quick Takes"]
        Tier1 --> RankingsDB[("PostgreSQL: user_digest_rankings")]
        Tier2 --> RankingsDB
    end

    subgraph Delivery["IEEE Publication & Delivery"]
        RankingsDB --> EmailAgent["EmailDeliveryAgent (app/agent/email_agent.py)"]
        EmailAgent --> PDF["ReportLab A4 PDF Generator"]
        EmailAgent --> HTML["IEEE Transactions HTML Builder"]
        PDF --> EmailSvc["EmailService (Resend HTTPS 443 <-> SMTP 587/465)"]
        HTML --> EmailSvc
    end
```

---

## Relational Database Schema

Managed via SQLAlchemy 2.0 on PostgreSQL 16 (with `init_db` auto-creating schema migrations and seeding defaults):

- `sources`: Monitored YouTube channels, OpenAI feeds, and Anthropic feeds (`is_active`, `last_scraped_at`).
- `articles`: Deduplicated ingested articles and spoken transcripts (`external_id`, `raw_content`, `is_summarized`).
- `digest`: 1:1 LLM-synthesized half-page digests (`summary`, `problem_statement`, `solution_statement`, `key_insights`, `model_used`).
- `subscribers`: Subscriber email, display name, role (`bio_or_role`), expertise level, interests, and custom ranking directives.
- `user_digest_rankings`: Personalized `0–100` calibrated score, rank position, tier (`featured` | `quick_bite`), and practitioner takeaways per subscriber and `batch_date`.
- `digests`: Legacy historical compiled briefings archive table.

---

## Key Reliability Features

- **Resilient Gemini Model Cascade**: Automatically fails over from `gemini-3.5-flash` -> `gemini-3-flash-preview` -> `gemini-3.5-flash-lite` on both transient `503`/`504` overloads and `429` rate limits, with per-call HTTP timeouts (`30s` summarization, `45s` ranking) and OpenAI / offline fallbacks.
- **Anti-Bot Blog Extraction**: Automatically falls back to the Jina Reader endpoint (`https://r.jina.ai/{url}`) when upstream article sites return HTTP `403` or `429` (`ENABLE_READER_FALLBACK=true`).
- **YouTube Datacenter Proxy Support**: Set optional `YOUTUBE_PROXY` in cloud hosts when datacenter IPs hit YouTube caption blocks.
- **Web Healthchecks & Cloud Privacy**: Exposes `GET /healthz` and `HEAD /` for Render healthchecks, and automatically redacts subscriber PII in `GET /api/config` when running on Render (`RENDER=true`).

---

## 1. Cloud Deployment & Receiving Email on Render (`deployment-part-2` branch)

Render deployments track the `deployment-part-2` branch using [`render.yaml`](render.yaml), which provisions a **Docker Web Service** (`ai-research-aggregator`) + **Managed PostgreSQL 16** (`ai-news-db`) and routes live email over HTTPS via **Resend** (because free cloud hosts block outbound SMTP ports `587`/`465`).

### Deployment Setup Steps

1. In your [Render Dashboard](https://dashboard.render.com/), click **New +** -> **Blueprint**.
2. Connect repository `https://github.com/Dhruv-Chugh/AI-research-aggregator` and select branch **`deployment-part-2`**.
3. Set the required secret Environment Variables in Render **Environment** (all marked `sync: false` in `render.yaml` so zero secrets or personal emails live in Git):

| Env Key | Required? | Purpose |
|:---|:---|:---|
| `GEMINI_API_KEY` | **Yes** | Google AI Studio key for digest synthesis & ranking |
| `RESEND_API_KEY` | **Yes** | Resend HTTPS API key for cloud email delivery (`443`) |
| `RECIPIENT_EMAIL` | **Yes** | Destination email address receiving daily briefings |
| `SUBSCRIBER_NAME` | **Yes** | Subscriber display name shown in greeting & PDF cover |
| `RESEND_FROM_EMAIL` | Preset | Defaults to `The AI Dispatch <onboarding@resend.dev>` |
| `GEMINI_MODEL` | Preset | Primary model (`gemini-3.5-flash`) |
| `GEMINI_FALLBACK_MODELS` | Preset | Comma-separated fallbacks (`gemini-3-flash-preview,gemini-3.5-flash-lite`) |
| `YOUTUBE_PROXY` | Optional | HTTP/HTTPS proxy URL if cloud datacenter IP hits YouTube transcript limits |
| `OPENAI_API_KEY` | Optional | Secondary LLM provider fallback |
| `SMTP_USER` / `SMTP_PASSWORD` | Optional | Fallback SMTP credentials if `RESEND_API_KEY` is removed |

> **Resend Sandbox Note (`onboarding@resend.dev`):** Resend's default testing sender (`onboarding@resend.dev`) only delivers emails to the email address registered on your Resend account. To email arbitrary addresses, verify your custom domain inside Resend and update `RESEND_FROM_EMAIL`.

### How to Trigger & Receive an Email from Render

1. Wait for Render logs to print `Your service is live` and `Started 24-hour background scheduler daemon`.
2. Open your live Render URL (`https://<your-service>.onrender.com`). *(In cloud deployments where `RENDER=true`, subscriber email and name are automatically masked on `GET /api/config` while keeping your saved target intact when updating preferences.)*
3. Scroll to the bottom control bar and click **Trigger Pipeline Now** (or wait for the 24-hour APScheduler timer).
4. Monitor Render **Logs** to confirm end-to-end execution:
   ```text
   Sending briefing via Resend HTTP API ...
   Live email successfully sent via Resend (id=...) to: <RECIPIENT_EMAIL> with attachment Daily_AI_Briefing_YYYY-MM-DD.pdf
   ```

---

## 2. Run Full End-to-End Pipeline Locally (`main` branch)

To run scraping, database storage, LLM summarization, personalized ranking, PDF compilation, and live email delivery **directly on your local computer**, use the **`main`** branch:

### Step 1: Clone & Switch to `main`

Prerequisites: **Python 3.12+**, **[uv](https://github.com/astral-sh/uv)**, and **Docker Desktop**.

```bash
git clone https://github.com/Dhruv-Chugh/AI-research-aggregator.git
cd AI-research-aggregator
git checkout main

# Install dependencies into .venv
uv sync
```

### Step 2: Configure `.env`

```bash
cp .env.example .env
```

Set your local credentials inside `.env` (`.env` is gitignored and never leaves your computer):

```ini
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_news_db
RECIPIENT_EMAIL=your_email@gmail.com
SUBSCRIBER_NAME=Your Name

# Primary Local Email Delivery: Gmail SMTP TLS (16-char Google App Password)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop

# Optional Backup Email Delivery (automatic fallback on main if 587/465 are blocked)
# RESEND_API_KEY=re_your_resend_api_key
# RESEND_FROM_EMAIL=The AI Dispatch <onboarding@resend.dev>

GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 3: Start PostgreSQL & Initialize Database

```bash
docker compose up -d postgres
# Or if starting a pre-existing container: docker start ai_news_postgres

uv run python -m app.database.init_db --seed
```

### Step 4: Verify & Run Full Pipeline

```bash
# 1. Verify PostgreSQL, API keys, and email ports (587 / 465 / Resend 443)
uv run python scripts/manual_send.py --preflight

# 2. Run FULL pipeline (Scrape -> PostgreSQL -> Summarize -> Rank -> PDF -> Live Email)
uv run python scripts/manual_send.py --full

# Optional: widen lookback to 7 days (168h) on quiet news cycles
uv run python scripts/manual_send.py --full --hours 168

# Optional: send immediately using briefings already stored in PostgreSQL
uv run python scripts/manual_send.py

# Optional: save HTML & PDF locally under exports/briefings/ without emailing
uv run python scripts/manual_send.py --full --dry-run

# Optional: launch Web Configuration Portal (auto-picks free port 8000-8010)
uv run ai-form
```

### Individual Stage CLI Reference (`main`)

| Command | Purpose |
|:---|:---|
| `uv run python scripts/manual_send.py --preflight` | Verify PostgreSQL, LLM keys, SMTP `587`/`465`, and Resend readiness |
| `uv run python scripts/manual_send.py --full` | End-to-end run: scrape, summarize, rank, build PDF, and email live |
| `uv run python -m app.scrapers.runner` | Scrape YouTube transcripts + OpenAI/Anthropic feeds into PostgreSQL |
| `uv run python -m app.agent.summarizer` | Generate technical digests into table `digest` |
| `uv run python -m app.agent.ranking_agent --run-now` | Score and rank digests into table `user_digest_rankings` |
| `uv run python -m app.agent.email_agent --send` | Build IEEE HTML + ReportLab PDF and send live emails |
| `uv run python -m app.agent.email_agent --dry-run` | Export local HTML + PDF previews to `exports/briefings/` |

---

## 3. Running Headless Scheduler with Docker Compose

`docker-compose.yml` provisions `ai_postgres` and a headless scheduler container (`ai_aggregator`, with port `8000` unbound so it never collides with local `uv run ai-form`):

```bash
docker compose up --build -d
docker compose logs -f ai-aggregator
```

---

## 4. Automated Test Suite

Tests cover scrapers, content extraction (`r.jina.ai` fallback), PostgreSQL repositories, Gemini failover/rate-limit cascade, personalized ranking, ReportLab PDF generation, Resend/SMTP delivery, and FastAPI portal endpoints:

```bash
# Run core unit & integration suite
uv run python -m pytest tests/ --ignore=tests/test_web_server.py

# Run FastAPI web portal & PII-masking suite
uv run python -m pytest tests/test_web_server.py
```

---

## Security & Privacy

- **Secrets Never Committed**: `.env`, `*.db`, `*.sqlite3`, `*.log`, and generated PDFs/HTMLs under `exports/` are excluded via [`.gitignore`](.gitignore).
- **Cloud Portal Privacy**: On Render (`RENDER=true` or `MASK_PUBLIC_PII=true`), `/api/config` redacts the configured email (`u***@domain.com`) and omits `name` from public GET responses while preserving the target email during preference updates.
- **No Personal Defaults in Repo**: Codebase defaults use neutral placeholders (`subscriber@example.com`) and resolve real subscriber credentials exclusively from runtime environment variables (`.env` locally, Render Environment dashboard in cloud).

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
