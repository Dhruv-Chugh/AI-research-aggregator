<div align="center">

# ⚡ AI Research Intelligence Aggregator
### Autonomous AI Research Synthesis, Spoken Transcript Mining & IEEE Briefing Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PostgreSQL 16](https://img.shields.io/badge/postgresql-16-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg?logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![LLM: Gemini & OpenAI](https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenAI-4285F4.svg?logo=google&logoColor=white)](https://aistudio.google.com/)
[![Tests Passing](https://img.shields.io/badge/tests-102%20passed-success.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*An autonomous, multi-agent computational pipeline that monitors frontier AI laboratory preprints, engineering blogs, and technical lecture transcripts, synthesizes knowledge-dense architectural analyses, and delivers personalized IEEE-standard Technical Memoranda directly to subscriber inboxes.*

---

</div>

## 📌 Executive Overview

**AI Research Intelligence Aggregator** transforms chaotic streams of AI research announcements and hours of technical videos into concise, publication-grade intelligence briefings. 

Rather than producing generic summaries, the platform features a **Personalized User Interest Ranking Agent** that scores every piece of research against the subscriber's specific technical role, domain focus areas (e.g., *autonomous agents, inference scaling, foundation alignment*), and custom prioritization directives.

Each morning, subscribers receive:
1. **An IEEE Transactions–Styled Responsive Email**: Featuring formal Abstract & Index Terms, citation-indexed literature sections (`[A1]`, `[A2]`, `[B1]`), and zero marketing fluff.
2. **An Attached Publication-Grade Visual PDF**: A comprehensive, 7-page ReportLab compendium with running headers, executive metadata grids, problem vs. solution comparative matrices, and tailored practitioner takeaways.

---

## 🏛 System Architecture

The platform operates as a decoupled, multi-agent pipeline backed by PostgreSQL:

```mermaid
flowchart TD
    subgraph Sourcing & Ingestion
        YT["YouTube Technical Feeds<br/>(Two Minute Papers, Yannic Kilcher, etc.)"] --> Runner["Ingestion Runner<br/>(app/scrapers/runner.py)"]
        OA["OpenAI Research & Engineering<br/>(Web & RSS Feeds)"] --> Runner
        ANT["Anthropic Research & Engineering<br/>(Web & RSS Feeds)"] --> Runner
        Runner --> Transcripts["Transcript Extraction<br/>(yt-dlp & youtube-transcript-api)"]
        Transcripts --> ArticlesDB[("PostgreSQL: articles<br/>(Deduplicated via external_id)")]
    end

    subgraph LLM Synthesis & Extraction
        ArticlesDB --> SummarizerAgent["Analytical Summarizer Agent<br/>(DualLLMClient: Gemini / OpenAI)"]
        SummarizerAgent --> DigestDB[("PostgreSQL: digest<br/>(Half-page analyses, problem-solution & insights)")]
    end

    subgraph Multi-Agent Personalization
        DigestDB --> UserRankingAgent["Personalized Ranking Agent<br/>(Evaluates against Subscriber Profile)"]
        SubscribersDB[("PostgreSQL: subscribers<br/>(Role, interests, expertise)")] --> UserRankingAgent
        UserRankingAgent --> Classifier["Two-Tier Content Classifier<br/>(max_featured_per_source=2)"]
        Classifier --> Tier1["🌟 Featured Strategic Deep Dives"]
        Classifier --> Tier2["⚡ Ecosystem Radar & Preprints"]
        Tier1 --> UserRankingsDB[("PostgreSQL: user_digest_rankings")]
        Tier2 --> UserRankingsDB
    end

    subgraph Publication & Delivery
        UserRankingsDB --> EmailDeliveryAgent["Email Delivery Agent<br/>(app/agent/email_agent.py)"]
        EmailDeliveryAgent --> PDFGen["ReportLab PDF Generator<br/>(IEEE Technical Compendium)"]
        EmailDeliveryAgent --> EmailSvc["MIME Email Service<br/>(Responsive IEEE HTML Template)"]
        PDFGen --> EmailSvc
        EmailSvc --> SMTP["Authenticated SMTP TLS Delivery<br/>(Gmail / Custom Mail Server)"]
    end
```

---

## 🗄 Relational Database Schema

The database architecture is fully normalized and managed via SQLAlchemy 2.0 with PostgreSQL 16:

```mermaid
erDiagram
    SOURCES ||--o{ ARTICLES : "monitors / ingests"
    ARTICLES ||--o| DIGEST : "synthesized into (1:1)"
    DIGEST ||--o{ USER_DIGEST_RANKINGS : "evaluated in"
    SUBSCRIBERS ||--o{ USER_DIGEST_RANKINGS : "receives"

    SOURCES {
        int id PK
        string name
        string source_type "youtube | openai | anthropic"
        string url
        string channel_id
        boolean is_active
        datetime last_scraped_at
    }

    ARTICLES {
        int id PK
        int source_id FK
        string external_id UK "Unique hash / video ID"
        string title
        string url
        string content_type "article | transcript | video"
        text raw_content "Full transcript or scraped text"
        boolean is_summarized
        datetime published_at
    }

    DIGEST {
        int id PK
        int article_id FK "1:1 Unique Reference"
        string title
        string url
        string source_name
        string source_type
        text summary "Knowledge-dense technical analysis"
        text problem_statement "Identified architectural bottleneck"
        text solution_statement "Breakthrough methodology"
        json key_insights "Actionable takeaways list"
        string model_used
        int tokens_used
    }

    SUBSCRIBERS {
        int id PK
        string email UK
        string name
        string bio_or_role "e.g. AI Software Architect"
        string expertise_level "advanced | expert"
        json interests "Domain focus topics"
        json preferred_sources
        text ranking_instructions "Custom user directives"
        boolean is_active
    }

    USER_DIGEST_RANKINGS {
        int id PK
        int subscriber_id FK
        int digest_id FK
        int rank_position "Priority order"
        float rank_score "Calibrated 0-100 metric"
        text relevance_reasoning "Profile-specific rationale"
        json key_takeaways_for_user
        string tier "featured | quick_bite"
        text quick_summary "3-4 line tactical brief"
        date batch_date
    }
```

---

## ✨ Core Features

- **Multi-Source Ingestion & Transcript Extraction**:
  - Automatically scrapes OpenAI Research/Engineering and Anthropic News feeds.
  - Monitors YouTube channels using RSS and downloads full spoken transcripts via `youtube-transcript-api` and `yt-dlp`.
  - Filters out YouTube Shorts and irrelevant media automatically.
- **Dual-LLM Structured Synthesis**:
  - Uses Google Gemini 2.5 (`gemini-2.5-flash`) via the modern `google-genai` SDK and OpenAI Responses API.
  - Generates structured problem/solution breakdowns, core insights, and technical summaries.
  - Offline deterministic fallback ensures pipeline resilience if external APIs are unreachable.
- **Two-Tier Daily Briefing Model**:
  - **Tier 1 (Featured Deep Dives)**: High-conviction architectural reports with problem-solution comparison tables and role takeaways (maximum 2 per source for diversity).
  - **Tier 2 (Ecosystem Radar & Preprints)**: Rapid 3–4 line tactical summaries ensuring zero information loss across auxiliary news.
- **IEEE Technical Memorandum Formatting**:
  - **Email**: Clean academic layout featuring IEEE double hairline masthead, Abstract & Index Terms, citation indexing (`[A1]`, `[B1]`), and zero marketing emojis or AI disclaimers.
  - **PDF Report**: Publication-grade A4 document built with ReportLab, complete with two-pass `NumberedCanvas` headers, executive metadata grids, and visual relevance meters.
- **24-Hour Autonomous Scheduler**:
  - Integrated APScheduler daemon executing automated ingestion, evaluation, and delivery every 24 hours.

---

## 🚀 Quickstart Guide

### Prerequisites
- Python 3.12+
- [Astral uv](https://github.com/astral-sh/uv) (fast Python package installer)
- Docker & Docker Compose (for local PostgreSQL)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Dhruv-Chugh/AI-research-aggregator.git
cd AI-research-aggregator

# Install dependencies into isolated .venv using uv
uv sync
```

### 2. Configure Environment Variables
Copy the template and configure your local settings:
```bash
cp .env.example .env
```
Edit `.env` with your preferred settings:
```ini
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ai_news_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_news_db

# Delivery Credentials
RECIPIENT_EMAIL=your_email@gmail.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_dispatch_sender@gmail.com
SMTP_PASSWORD=your_16_char_app_password

# LLM API Key (Free tier from Google AI Studio)
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start PostgreSQL with Docker
```bash
docker compose up postgres -d
```

### 4. Initialize Database Tables & Seed Sources
```bash
uv run python -m app.database.init_db --seed
```

---

## 💻 CLI Command Reference

| Command | Action |
|:---|:---|
| `uv run python -m app.web.server` | Launch the Google/Microsoft Forms-styled configuration portal on `http://127.0.0.1:8000`. |
| `.\run_portal.bat` | Convenient one-click launcher for the configuration portal. |
| `uv run python -m app.scrapers.runner` | Ingest latest articles and video transcripts from all configured sources. |
| `uv run python -m app.agent.summarizer` | Run LLM synthesis agent to generate analytical digests for unsummarized articles. |
| `uv run python -m app.agent.ranking_agent --run-now` | Execute one-shot personalized interest ranking for all active subscribers. |
| `uv run python -m app.agent.ranking_agent --daemon --interval-hours 24` | Launch 24-hour autonomous scheduler daemon. |
| `uv run python -m app.agent.email_agent --dry-run` | Generate and export IEEE HTML and PDF briefings locally to `exports/briefings/`. |
| `uv run python -m app.agent.email_agent --send` | Generate briefings and dispatch live via SMTP TLS to subscriber inboxes. |

---

## 🚀 One-Click Cloud Deployment (Render.com)

The repository includes a production-ready **Render Blueprint** ([`render.yaml`](render.yaml)) configured for the `deploy` branch that provisions:
1. **Managed PostgreSQL Database** (`ai-news-db`): Dedicated PostgreSQL 16 database with persistent storage and auto-SSL.
2. **Web & Scheduling Service** (`ai-research-aggregator`): Docker container running the Google/Microsoft Forms configuration portal on port `8000` alongside an automated 24-hour background scheduler.

### 1-Click Deployment Steps:
1. Log into your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** → **Blueprint**.
3. Select this repository: `https://github.com/Dhruv-Chugh/AI-research-aggregator` (Branch: `deploy`).
4. Render automatically parses `render.yaml` and links the PostgreSQL database to the web service via `DATABASE_URL`.
5. Enter your environment secrets in the Render dashboard:
   - `GEMINI_API_KEY`: Your Gemini API key (from [Google AI Studio](https://aistudio.google.com/)).
   - `OPENAI_API_KEY`: *(Optional)* OpenAI fallback key.
   - `SMTP_USER`: Your Gmail address.
   - `SMTP_PASSWORD`: Your 16-character Gmail App Password.
   - `RECIPIENT_EMAIL`: Recipient email address for IEEE technical briefings.
6. Click **Apply**.
   - Render builds the image, creates database schema tables, launches the web portal, and activates the 24-hour autonomous pipeline.
   - Access your portal anytime at: `https://ai-research-aggregator.onrender.com`.

---

## 🐳 Production Container Deployment

The platform is packaged with a lightweight multi-stage `Dockerfile` and `docker-compose.yml`:

```bash
# Build and run PostgreSQL and the autonomous dispatch scheduler in the background
docker compose up --build -d
```

View live execution logs:
```bash
docker compose logs -f ai-aggregator
```

---

## 🧪 Automated Test Suite

The project includes 102 unit and integration tests covering scrapers, LLM clients, database repositories, ranking logic, PDF compilation, and email dispatching:

```bash
# Run complete test suite
uv run pytest
```

---

## 🔒 Security & Privacy

- **Local Secrets Isolation**: `.env` is strictly excluded from version control via `.gitignore`.
- **Zero Third-Party Telemetry**: Scraping, storage, and compilation run locally on your host or private infrastructure.
- **TLS Authentication**: Email dispatches require authenticated TLS with dedicated Google App Passwords.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

<div align="center">
<b>AI Research Intelligence Aggregator</b> &bull; Built with precision for AI engineers, researchers, and systems architects.
</div>
