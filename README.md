<div align="center">

# Axiom Dispatch — AI Research Intelligence Aggregator
### Autonomous Multi-Source Research Synthesis, Spoken Transcript Mining & IEEE Briefing Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL 16](https://img.shields.io/badge/postgresql-16-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![LLM: Gemini 3.5 / 3 Flash & OpenAI](https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenAI-4285F4.svg?logo=google&logoColor=white)](https://aistudio.google.com/)
[![Tests: 125 Passed](https://img.shields.io/badge/tests-125%20passed-success.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
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
        PDF --> EmailSvc["EmailService (SMTP 587/465 <-> Resend HTTPS 443)"]
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
- **Quiet-Day Window Expansion**: When running a full scrape with lookback `<= 48h` yields zero items (e.g., quiet news cycles), `AggregationRunner` (`expand_if_empty=True`) automatically widens the window to `168h` (7 days).
- **Zero-Conflict Port & Email Failover**:
  - `app/web/server.py` automatically selects the next open local port (`8000`–`8010`) if port `8000` is busy, and exposes `HEAD /` and `GET /healthz` for cloud healthchecks.
  - `EmailService` on `main` fails over `587` -> `465` SSL -> Resend `443` HTTPS when outbound SMTP is filtered.

---

## 1. Run Full End-to-End Pipeline Locally (`main` branch)

Follow these steps to run scraping, database storage, LLM summarization, personalized ranking, PDF compilation, and **receive a live email briefing directly from your local machine**.

### Step 1: Clone & Install Dependencies

Prerequisites: **Python 3.12+**, **[uv](https://github.com/astral-sh/uv)**, and **Docker Desktop** (for PostgreSQL).

```bash
git clone https://github.com/Dhruv-Chugh/AI-research-aggregator.git
cd AI-research-aggregator
git checkout main

# Install all dependencies and CLI entry points into .venv
uv sync
```

### Step 2: Create & Configure `.env`

Copy the template file (`.env` is gitignored and strictly stays on your machine):

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```ini
# PostgreSQL (matches docker-compose.yml defaults)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_news_db

# Recipient & Display Name
RECIPIENT_EMAIL=your_email@gmail.com
SUBSCRIBER_NAME=Your Name

# Primary Local Email Delivery: Gmail SMTP TLS (16-char Google App Password)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_gmail@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop

# Optional Backup Email Delivery (used automatically on main if ports 587/465 are blocked)
# RESEND_API_KEY=re_your_resend_api_key
# RESEND_FROM_EMAIL=The AI Dispatch <onboarding@resend.dev>

# LLM API Key (Google AI Studio: https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Gmail App Password Tip:** Standard Gmail passwords do not work for SMTP. Enable 2-Step Verification and generate a 16-character App Password at `https://myaccount.google.com/apppasswords`.

### Step 3: Start PostgreSQL & Initialize Tables

```bash
# Start PostgreSQL container
docker compose up -d postgres
# (If you already created a standalone 'ai_news_postgres' container earlier, run: docker start ai_news_postgres)

# Create tables and seed sources/subscriber from config/ and .env
uv run python -m app.database.init_db --seed
```

### Step 4: Verify Prerequisites (Preflight Check)

Confirm PostgreSQL connectivity, API keys, and email ports (`587` / `465` / Resend `443`) are healthy before running:

```bash
uv run python scripts/manual_send.py --preflight
```

### Step 5: Run Full End-to-End Pipeline & Send Email

Run the entire pipeline from fresh web scraping to live Gmail/Resend delivery in **one command**:

```bash
# Full end-to-end pipeline (Scrape -> PostgreSQL -> Summarize -> Rank -> IEEE PDF -> Live Email)
uv run python scripts/manual_send.py --full
```

Helpful variations:

```bash
# Expand lookback window to 7 days (168 hours) if last 24h has few articles
uv run python scripts/manual_send.py --full --hours 168

# Re-send or deliver using briefings already stored in PostgreSQL (auto-fills missing summaries/rankings)
uv run python scripts/manual_send.py

# Preview locally only: saves HTML & PDF inside exports/briefings/ without sending an email
uv run python scripts/manual_send.py --full --dry-run
```

Every run exports archival copies of the compiled HTML email and A4 PDF report to `exports/briefings/` (gitignored).

### Step 6: Customize Preferences via the Local Web Portal (Optional)

Launch the FastAPI configuration portal (`http://127.0.0.1:8000`, automatically selecting `8001`–`8010` if `8000` is occupied) to add/remove YouTube channels, toggle OpenAI/Anthropic sources, tune ranking directives, or trigger runs visually:

```bash
uv run ai-form
# Or on Windows PowerShell / CMD:
.\run_portal.bat
```

### Individual Stage CLI Reference (`main`)

| Command | Purpose |
|:---|:---|
| `uv run python scripts/manual_send.py --preflight` | Check PostgreSQL, Gemini/OpenAI keys, SMTP ports `587`/`465`, and Resend readiness |
| `uv run python scripts/manual_send.py --full --hours 168` | Complete end-to-end run: scrape, summarize, rank, build PDF, and send live email |
| `uv run python scripts/manual_send.py` | Build PDF and send live email from existing PostgreSQL data |
| `uv run python -m app.scrapers.runner` | Ingest new YouTube transcripts and OpenAI/Anthropic articles into PostgreSQL |
| `uv run python -m app.agent.summarizer` | Summarize unsummarized PostgreSQL articles into table `digest` |
| `uv run python -m app.agent.ranking_agent --run-now` | Score and rank digests into table `user_digest_rankings` |
| `uv run python -m app.agent.email_agent --send` | Render IEEE HTML + ReportLab PDF and send live emails |
| `uv run python -m app.agent.email_agent --dry-run` | Export IEEE HTML + PDF previews to `exports/briefings/` without emailing |
| `uv run ai-form` | Start configuration portal + 24h APScheduler locally |

---

## 2. Cloud Deployment & Receiving Email on Render (`deployment-part-2` branch)

Render deployments track the `deployment-part-2` branch using [`render.yaml`](https://github.com/Dhruv-Chugh/AI-research-aggregator/blob/deployment-part-2/render.yaml), which provisions a **Docker Web Service** (`ai-research-aggregator`) + **Managed PostgreSQL 16** (`ai-news-db`) and routes live email over HTTPS via **Resend** (because free cloud hosts block outbound SMTP ports `587`/`465`).

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
2. Open your live Render URL (`https://<your-service>.onrender.com`). *(Note: In cloud deployments where `RENDER=true`, subscriber email and name are automatically masked on `GET /api/config` while keeping your saved target intact when updating preferences.)*
3. Scroll to the bottom control bar and click **Trigger Pipeline Now** (or wait for the 24-hour APScheduler timer).
4. Monitor Render **Logs** to confirm end-to-end execution:
   ```text
   Sending briefing via Resend HTTP API ...
   Live email successfully sent via Resend (id=...) to: <RECIPIENT_EMAIL> with attachment Daily_AI_Briefing_YYYY-MM-DD.pdf
   ```

---

## 3. Running Headless Scheduler with Docker Compose (Local)

`docker-compose.yml` provisions `ai_postgres` and a headless scheduler container (`ai_aggregator`, with port `8000` unbound so it never collides with `uv run ai-form`):

```bash
docker compose up --build -d
docker compose logs -f ai-aggregator
```

*(Note: If you already run a standalone `ai_news_postgres` container on port `5432`, stop it before running `docker compose up -d`, or use `docker start ai_news_postgres` directly with the CLI commands above.)*

---

## 4. Automated Test Suite

Tests cover scrapers, content extraction (`r.jina.ai` fallback), PostgreSQL repositories, Gemini failover/rate-limit cascade, personalized ranking, ReportLab PDF generation, email transport failover, and FastAPI portal endpoints:

```bash
# Run core unit & integration suite
uv run python -m pytest tests/ --ignore=tests/test_web_server.py

# Run FastAPI web portal & PII-masking suite
uv run python -m pytest tests/test_web_server.py
```

*(Running `test_web_server.py` separately avoids Windows sandbox socketpair conflicts when using restricted shells.)*

---

## Security & Privacy

- **Secrets Never Committed**: `.env`, `*.db`, `*.sqlite3`, `*.log`, and generated PDFs/HTMLs under `exports/` are excluded via [`.gitignore`](.gitignore).
- **Cloud Portal Privacy**: On Render (`RENDER=true` or `MASK_PUBLIC_PII=true`), `/api/config` redacts the configured email (`u***@domain.com`) and omits `name` from public GET responses while preserving the target email during preference updates.
- **No Personal Defaults in Repo**: Codebase defaults use neutral placeholders (`subscriber@example.com`) and resolve real subscriber credentials exclusively from runtime environment variables (`.env` locally, Render Environment dashboard in cloud).

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
