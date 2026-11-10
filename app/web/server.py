"""FastAPI Web Configuration Server for Axiom Dispatch.

Provides an authentic Google/Microsoft Forms-styled configuration interface for:
- Monitoring, adding, and removing YouTube channels
- Calibrating subscriber profile and technical expertise
- Defining multi-agent LLM ranking directives
- Toggling frontier lab feeds (OpenAI & Anthropic)
- Managing delivery settings and triggering pipeline test runs
- Running an integrated 24-hour autonomous scheduler daemon (APScheduler)
"""

from __future__ import annotations

import sys
import types

# Windows Application Control / AppLocker compatibility guard:
# When Windows blocks _ctypes.pyd from user AppData directories, safely mock click._winconsole
# so modern Windows terminal output continues using native sys.stdout/sys.stderr without ctypes.
if sys.platform.startswith("win"):
    try:
        import ctypes  # noqa: F401
    except ImportError:
        _mock_winconsole = types.ModuleType("click._winconsole")
        _mock_winconsole._get_windows_console_stream = lambda f, enc, err: None
        sys.modules["click._winconsole"] = _mock_winconsole

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import (
    DEFAULT_LOOKBACK_HOURS,
    DEFAULT_SOURCES_PATH,
    SourcesConfig,
    add_channel,
    get_sources_config,
    resolve_channel_info,
)
from app.models.youtube import ChannelConfig

logger = logging.getLogger("ai_news.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

PREFERENCES_PATH = Path("config/subscriber_preferences.json")
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"

# Global scheduler and execution tracking state
_scheduler: Optional[BackgroundScheduler] = None
_pipeline_state: Dict[str, Any] = {
    "is_running": False,
    "last_run_at": None,
    "last_run_status": "idle",
    "last_run_error": None,
    "briefings_delivered": 0,
}

# Default subscriber preferences
DEFAULT_SUBSCRIBER = {
    "email": "",
    "name": "",
    "is_active": True,
    "bio_or_role": "AI Software Architect & Researcher",
    "expertise_level": "advanced",
    "interests": [],
    "preferred_sources": [],
    "ranking_instructions": (
        "Prioritize production architectures, open-weights releases, and practical implementation details. "
        "De-prioritize high-level corporate hype and financing rounds."
    ),
    "lookback_hours": 24,
    "max_featured_per_source": 2,
    "email_delivery_enabled": True,
}


def load_local_preferences() -> Dict[str, Any]:
    """Load cached subscriber preferences from disk with default fallback."""
    if PREFERENCES_PATH.exists():
        try:
            data = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_SUBSCRIBER)
            merged.update(data)
            return merged
        except Exception as e:
            logger.warning("Error reading local preferences: %s. Using defaults.", e)
    return dict(DEFAULT_SUBSCRIBER)


def save_local_preferences(prefs: Dict[str, Any]) -> None:
    """Save subscriber preferences to disk."""
    PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCES_PATH.write_text(json.dumps(prefs, indent=2, ensure_ascii=False), encoding="utf-8")


def auto_init_database() -> bool:
    """Auto-create tables and seed default subscriber on startup if DB is reachable."""
    try:
        from app.database import check_connection, create_tables
        from app.database.init_db import seed_initial_data
        from app.database.session import engine

        if check_connection():
            logger.info("PostgreSQL database detected. Running schema migrations and seeding...")
            create_tables(engine)
            seed_initial_data(engine)

            logger.info("Database initialized successfully.")
            return True
        else:
            logger.info("PostgreSQL not reachable at startup. Operating with local config fallback.")
            return False
    except Exception as e:
        logger.warning("Database auto-init warning: %s", e)
        return False


def execute_full_pipeline(hours: Optional[int] = None, send_live: Optional[bool] = None) -> Dict[str, Any]:
    """Execute complete pipeline: scrape -> summarize -> rank -> deliver."""
    global _pipeline_state
    if _pipeline_state["is_running"]:
        logger.warning("Pipeline is already executing. Skipping concurrent run.")
        return {"status": "skipped", "reason": "Already running"}

    _pipeline_state["is_running"] = True
    _pipeline_state["last_run_status"] = "running"
    _pipeline_state["last_run_error"] = None

    try:
        from app.scrapers.runner import run_news_aggregation

        pref = load_local_preferences()
        effective_hours = hours
        if effective_hours is None:
            effective_hours = int(pref.get("lookback_hours", os.getenv("DEFAULT_LOOKBACK_HOURS", "24")))

        live_send = send_live
        if live_send is None:
            live_send = os.getenv("SEND_LIVE_EMAIL", "true").lower() in ("true", "1", "yes")

        logger.info("Starting autonomous news, ranking & delivery pipeline (hours=%d, live_send=%s)...", effective_hours, live_send)
        res = run_news_aggregation(
            hours=effective_hours,
            include_youtube=True,
            include_openai=True,
            include_anthropic=True,
            fetch_transcripts=True,
            fetch_full_content=True,
            sync_to_db=True,
            summarize=True,
            rank=True,
            email_briefing=True,
            send_live_email=live_send,
        )

        # Resilient fallback: If 24h-48h yielded 0 items (e.g. over weekends or quiet research days),
        # automatically expand lookback to 168 hours (7 days) to ensure subscribers get an intelligence briefing.
        if res.total_items == 0 and effective_hours <= 48:
            logger.info(
                "0 items discovered in %dh window. Expanding lookback to 168 hours (7 days) to ingest recent frontier research...",
                effective_hours,
            )
            res = run_news_aggregation(
                hours=168,
                include_youtube=True,
                include_openai=True,
                include_anthropic=True,
                fetch_transcripts=True,
                fetch_full_content=True,
                sync_to_db=True,
                summarize=True,
                rank=True,
                email_briefing=True,
                send_live_email=live_send,
            )

        _pipeline_state["last_run_at"] = datetime.now(timezone.utc).isoformat()
        _pipeline_state["last_run_status"] = "success"
        _pipeline_state["briefings_delivered"] = res.briefings_delivered
        logger.info("Pipeline completed: %d items processed, %d briefings delivered.", res.total_items, res.briefings_delivered)
        return {"status": "success", "total_items": res.total_items, "briefings": res.briefings_delivered}
    except Exception as e:
        logger.error("Pipeline execution failed: %s", e, exc_info=True)
        _pipeline_state["last_run_status"] = "error"
        _pipeline_state["last_run_error"] = str(e)
        return {"status": "error", "error": str(e)}
    finally:
        _pipeline_state["is_running"] = False


def get_db_status_and_subscriber() -> tuple[bool, Dict[str, Any], int, Optional[str]]:
    """Probe database status and retrieve subscriber profile if available."""
    try:
        from app.database import check_connection, get_db_session
        from app.database.models import Article, Subscriber
        from sqlalchemy import func, select

        if not check_connection():
            return False, load_local_preferences(), 0, None

        with get_db_session() as session:
            count_stmt = select(func.count(Article.id))
            article_count = session.execute(count_stmt).scalar() or 0

            latest_stmt = select(func.max(Article.created_at))
            latest_dt = session.execute(latest_stmt).scalar()
            latest_str = latest_dt.isoformat() if latest_dt else None

            pref = load_local_preferences()
            sub_email = pref.get("email", DEFAULT_SUBSCRIBER["email"])
            sub_stmt = select(Subscriber).where(Subscriber.email == sub_email)
            subscriber = session.execute(sub_stmt).scalar_one_or_none()

            if subscriber:
                clean_email = subscriber.email or ""
                if any(x in clean_email.lower() for x in ["dhruv", "chugh", "byteme"]):
                    clean_email = ""
                clean_name = subscriber.name or pref.get("name") or ""
                if "dhruv" in clean_name.lower():
                    clean_name = ""

                sub_data = {
                    "email": clean_email,
                    "name": clean_name,
                    "is_active": subscriber.is_active,
                    "bio_or_role": subscriber.bio_or_role or pref.get("bio_or_role"),
                    "expertise_level": subscriber.expertise_level or pref.get("expertise_level"),
                    "interests": subscriber.interests if subscriber.interests is not None else pref.get("interests", []),
                    "preferred_sources": subscriber.preferred_sources if subscriber.preferred_sources is not None else pref.get("preferred_sources", []),
                    "ranking_instructions": subscriber.ranking_instructions or pref.get("ranking_instructions"),
                    "lookback_hours": pref.get("lookback_hours", 24),
                    "max_featured_per_source": pref.get("max_featured_per_source", 2),
                    "email_delivery_enabled": pref.get("email_delivery_enabled", True),
                }
                return True, sub_data, article_count, latest_str

            clean_pref = dict(pref)
            if any(x in str(clean_pref.get("email", "")).lower() for x in ["dhruv", "chugh", "byteme"]):
                clean_pref["email"] = ""
            if "dhruv" in str(clean_pref.get("name", "")).lower():
                clean_pref["name"] = ""

            return True, clean_pref, article_count, latest_str
    except Exception as e:
        logger.debug("Database probe fallback: %s", e)
        return False, load_local_preferences(), 0, None


def sync_subscriber_to_db(prefs: Dict[str, Any]) -> bool:
    """Sync subscriber preferences to PostgreSQL if connected."""
    try:
        from app.database import check_connection, get_db_session
        from app.database.models import Subscriber
        from sqlalchemy import select

        if not check_connection():
            return False

        with get_db_session() as session:
            email = prefs.get("email", DEFAULT_SUBSCRIBER["email"]).strip().lower()
            stmt = select(Subscriber).where(Subscriber.email == email)
            subscriber = session.execute(stmt).scalar_one_or_none()

            if subscriber:
                subscriber.name = prefs.get("name")
                subscriber.is_active = prefs.get("is_active", True)
                subscriber.bio_or_role = prefs.get("bio_or_role")
                subscriber.expertise_level = prefs.get("expertise_level", "advanced")
                subscriber.interests = prefs.get("interests", [])
                subscriber.preferred_sources = prefs.get("preferred_sources", [])
                subscriber.ranking_instructions = prefs.get("ranking_instructions")
            else:
                subscriber = Subscriber(
                    email=email,
                    name=prefs.get("name"),
                    is_active=prefs.get("is_active", True),
                    bio_or_role=prefs.get("bio_or_role"),
                    expertise_level=prefs.get("expertise_level", "advanced"),
                    interests=prefs.get("interests", []),
                    preferred_sources=prefs.get("preferred_sources", []),
                    ranking_instructions=prefs.get("ranking_instructions"),
                )
                session.add(subscriber)

            session.commit()
            return True
    except Exception as e:
        logger.debug("Failed to sync subscriber to DB: %s", e)
        return False


# --- Pydantic Request/Response Models ---

class AddChannelRequest(BaseModel):
    identifier: str = Field(..., description="YouTube @handle, Channel URL, or Channel ID")
    name: Optional[str] = Field(None, description="Optional custom channel name override")


class ToggleChannelRequest(BaseModel):
    is_active: bool = Field(..., description="Active state for channel")


class LabSourcesRequest(BaseModel):
    openai_active: bool = Field(default=True, description="Enable OpenAI scraping")
    anthropic_active: bool = Field(default=True, description="Enable Anthropic scraping")


class UpdatePreferencesRequest(BaseModel):
    email: str = Field(..., description="Subscriber email address")
    name: Optional[str] = Field(None, description="Subscriber full name")
    is_active: bool = Field(default=True, description="Subscriber active state")
    bio_or_role: Optional[str] = Field(None, description="Professional role or appointment")
    expertise_level: str = Field(default="advanced", description="Technical expertise level")
    interests: List[str] = Field(default_factory=list, description="List of technical interest topics")
    preferred_sources: List[str] = Field(default_factory=list, description="Preferred source names")
    ranking_instructions: Optional[str] = Field(None, description="Personalized ranking directives")
    lookback_hours: int = Field(default=24, description="Lookback window in hours")
    max_featured_per_source: int = Field(default=2, description="Two-tier strategic limit")
    email_delivery_enabled: bool = Field(default=True, description="Automated delivery toggle")


# --- FastAPI Lifecycle & App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Auto-init database schema and seed data
    auto_init_database()

    # 2. Launch background scheduler if enabled
    scheduler_enabled = os.getenv("ENABLE_SCHEDULER", "true").lower() in ("true", "1", "yes")
    interval_hours = int(os.getenv("SCHEDULER_INTERVAL_HOURS", "24"))

    global _scheduler
    if scheduler_enabled:
        try:
            _scheduler = BackgroundScheduler()
            _scheduler.add_job(
                execute_full_pipeline,
                trigger=IntervalTrigger(hours=interval_hours),
                id="24h_autonomous_pipeline",
                name="24-Hour Axiom Dispatch Pipeline",
                kwargs={"hours": interval_hours},
                replace_existing=True,
            )
            _scheduler.start()
            logger.info("Started 24-hour background scheduler daemon (interval: every %d hours).", interval_hours)
        except Exception as e:
            logger.warning("Could not start background scheduler: %s", e)

    yield

    # Shutdown
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background scheduler stopped.")


app = FastAPI(
    title="Axiom Dispatch Configuration Portal",
    description="Authentic Google/Microsoft Forms styled settings portal for AI Research Aggregator.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- HTML Frontend & Healthcheck Routes ---

@app.get("/healthz")
async def healthcheck():
    """Ultra-fast, zero-overhead healthcheck endpoint for cloud orchestrators (Render)."""
    return {"status": "ok", "service": "axiom-dispatch"}


@app.get("/", response_class=HTMLResponse)
async def serve_form_ui():
    """Serve the single-page Google/Microsoft Forms styled interface."""
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Template file not found at {INDEX_HTML_PATH}",
        )
    return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))


# --- API Routes ---

@app.get("/api/config")
async def get_full_configuration():
    """Retrieve full configuration: monitored channels, lab feeds, and subscriber preferences."""
    sources_cfg = get_sources_config(DEFAULT_SOURCES_PATH)
    db_connected, subscriber_data, article_count, last_scraped = get_db_status_and_subscriber()

    # When DB is connected, use it as the authoritative source for YouTube channels
    # so that channels added via the UI persist across container restarts.
    if db_connected:
        try:
            from app.database import get_db_session
            from app.database.models import Source
            from sqlalchemy import select

            with get_db_session() as session:
                stmt = select(Source).where(
                    Source.source_type == "youtube",
                    Source.is_active.is_not(None),
                )
                db_sources = session.execute(stmt).scalars().all()
                if db_sources:
                    active_yt = [
                        {
                            "name": s.name,
                            "channel_id": s.channel_id or "",
                            "handle": s.handle or "",
                            "url": s.url or "",
                            "feed_url": s.feed_url or "",
                            "is_active": s.is_active if s.is_active is not None else True,
                        }
                        for s in db_sources
                    ]
                else:
                    # DB connected but no channel rows yet — fall back to file config
                    active_yt = [c.model_dump() for c in sources_cfg.youtube_channels]
        except Exception as db_err:
            logger.debug("DB channel read fallback: %s", db_err)
            active_yt = [c.model_dump() for c in sources_cfg.youtube_channels]
    else:
        active_yt = [c.model_dump() for c in sources_cfg.youtube_channels]

    return {
        "status": "success",
        "system": {
            "db_connected": db_connected,
            "total_articles": article_count,
            "last_scraped_at": last_scraped,
            "server_time": datetime.now(timezone.utc).isoformat(),
        },
        "sources": {
            "youtube_channels": active_yt,
            "openai": sources_cfg.openai.model_dump(),
            "anthropic": sources_cfg.anthropic.model_dump(),
        },
        "subscriber": subscriber_data,
    }


@app.post("/api/config")
async def save_configuration(payload: UpdatePreferencesRequest):
    """Save updated subscriber preferences and pipeline settings."""
    prefs = payload.model_dump()
    save_local_preferences(prefs)
    db_synced = sync_subscriber_to_db(prefs)

    return {
        "status": "success",
        "message": "Preferences saved successfully.",
        "db_synced": db_synced,
        "subscriber": prefs,
    }


@app.get("/api/channels")
async def list_channels():
    """List all monitored YouTube channels."""
    cfg = get_sources_config(DEFAULT_SOURCES_PATH)
    return {
        "status": "success",
        "count": len(cfg.youtube_channels),
        "channels": [c.model_dump() for c in cfg.youtube_channels],
    }


@app.post("/api/channels")
async def add_new_channel(payload: AddChannelRequest):
    """Resolve and add a YouTube channel by handle (@creator), URL, or Channel ID."""
    raw_ident = payload.identifier.strip()
    if not raw_ident:
        raise HTTPException(status_code=400, detail="Identifier cannot be empty.")

    try:
        resolved = resolve_channel_info(raw_ident, name=payload.name)
    except Exception as e:
        logger.warning("Channel resolution failed for '%s': %s", raw_ident, e)
        raise HTTPException(
            status_code=400,
            detail=f"Unable to resolve YouTube channel '{raw_ident}'. Ensure handle, URL, or Channel ID is valid.",
        )

    cfg = get_sources_config(DEFAULT_SOURCES_PATH)
    added = cfg.add_youtube_channel(resolved)
    if not added:
        raise HTTPException(
            status_code=409,
            detail=f"Channel '{resolved.name}' ({resolved.channel_id}) is already being monitored.",
        )

    cfg.save_to_file(DEFAULT_SOURCES_PATH)
    logger.info("Saved new channel '%s' to %s", resolved.name, DEFAULT_SOURCES_PATH)

    # Sync to DB if connected
    try:
        from app.database import check_connection, get_db_session
        from app.database.repository import NewsRepository

        if check_connection():
            with get_db_session() as session:
                repo = NewsRepository(session)
                repo.sources.upsert(
                    name=resolved.name,
                    source_type="youtube",
                    url=str(resolved.url),
                    feed_url=resolved.feed_url,
                    channel_id=resolved.channel_id,
                    handle=resolved.handle,
                    is_active=resolved.is_active,
                )
                session.commit()
    except Exception as e:
        logger.debug("Database channel sync note: %s", e)

    return {
        "status": "success",
        "message": f"Successfully added channel '{resolved.name}'.",
        "channel": resolved.model_dump(),
    }


@app.delete("/api/channels/{channel_id}")
async def delete_channel(channel_id: str):
    """Remove a YouTube channel from the monitored list."""
    target_id = channel_id.strip()
    cfg = get_sources_config(DEFAULT_SOURCES_PATH)
    removed = cfg.remove_youtube_channel(target_id)

    if not removed:
        raise HTTPException(status_code=404, detail=f"Channel with ID '{target_id}' not found.")

    cfg.save_to_file(DEFAULT_SOURCES_PATH)
    logger.info("Removed channel '%s' from %s", target_id, DEFAULT_SOURCES_PATH)

    # Sync DB if connected
    try:
        from app.database import check_connection, get_db_session
        from app.database.models import Source
        from sqlalchemy import select

        if check_connection():
            with get_db_session() as session:
                stmt = select(Source).where(Source.channel_id == target_id)
                source = session.execute(stmt).scalar_one_or_none()
                if source:
                    source.is_active = False
                    session.commit()
    except Exception as e:
        logger.debug("Database channel removal sync note: %s", e)

    return {
        "status": "success",
        "message": f"Channel '{target_id}' removed from monitored list.",
        "channel_id": target_id,
    }


@app.patch("/api/channels/{channel_id}/toggle")
async def toggle_channel_state(channel_id: str, payload: ToggleChannelRequest):
    """Toggle a YouTube channel active/inactive without deleting it."""
    target_id = channel_id.strip().lower()
    cfg = get_sources_config(DEFAULT_SOURCES_PATH)
    found = False

    for c in cfg.youtube_channels:
        if c.channel_id.lower() == target_id or (c.handle and c.handle.lower() == target_id):
            c.is_active = payload.is_active
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Channel '{channel_id}' not found.")

    cfg.save_to_file(DEFAULT_SOURCES_PATH)

    # Sync DB
    try:
        from app.database import check_connection, get_db_session
        from app.database.models import Source
        from sqlalchemy import select

        if check_connection():
            with get_db_session() as session:
                stmt = select(Source).where(Source.channel_id == channel_id)
                source = session.execute(stmt).scalar_one_or_none()
                if source:
                    source.is_active = payload.is_active
                    session.commit()
    except Exception as e:
        logger.debug("Database toggle sync note: %s", e)

    return {
        "status": "success",
        "message": f"Channel status set to {'active' if payload.is_active else 'inactive'}.",
        "channel_id": channel_id,
        "is_active": payload.is_active,
    }


@app.patch("/api/sources/labs")
async def toggle_lab_sources(payload: LabSourcesRequest):
    """Toggle OpenAI and Anthropic feed ingestion."""
    cfg = get_sources_config(DEFAULT_SOURCES_PATH)
    cfg.openai.is_active = payload.openai_active
    cfg.anthropic.is_active = payload.anthropic_active
    cfg.save_to_file(DEFAULT_SOURCES_PATH)

    # Sync DB
    try:
        from app.database import check_connection, get_db_session
        from app.database.models import Source
        from sqlalchemy import select

        if check_connection():
            with get_db_session() as session:
                stmt_oa = select(Source).where(Source.source_type == "openai")
                oa = session.execute(stmt_oa).scalar_one_or_none()
                if oa:
                    oa.is_active = payload.openai_active

                stmt_ant = select(Source).where(Source.source_type == "anthropic")
                ant = session.execute(stmt_ant).scalar_one_or_none()
                if ant:
                    ant.is_active = payload.anthropic_active

                session.commit()
    except Exception as e:
        logger.debug("Database lab sources sync note: %s", e)

    return {
        "status": "success",
        "message": "Frontier laboratory sources updated.",
        "openai_active": payload.openai_active,
        "anthropic_active": payload.anthropic_active,
    }


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Query current pipeline execution state and scheduler settings."""
    interval_hours = int(os.getenv("SCHEDULER_INTERVAL_HOURS", "24"))
    scheduler_active = _scheduler is not None and _scheduler.running
    return {
        "status": "success",
        "pipeline": _pipeline_state,
        "scheduler": {
            "active": scheduler_active,
            "interval_hours": interval_hours,
        },
    }


class PipelineTriggerPayload(BaseModel):
    hours: Optional[int] = Field(default=None, ge=1, le=720, description="Optional custom hours lookback")
    send_live: Optional[bool] = Field(default=None, description="Force live SMTP send toggle")


@app.post("/api/pipeline/trigger-now")
async def trigger_pipeline_now(payload: Optional[PipelineTriggerPayload] = None):
    """Trigger immediate execution of the pipeline in the background."""
    if _pipeline_state["is_running"]:
        return JSONResponse(
            status_code=409,
            content={"status": "already_running", "message": "Pipeline is already executing."},
        )

    req_hours = payload.hours if payload else None
    req_live = payload.send_live if payload else None

    # Launch in a daemon thread so HTTP response returns immediately
    thread = threading.Thread(
        target=execute_full_pipeline,
        kwargs={"hours": req_hours, "send_live": req_live},
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "message": f"Autonomous pipeline launched in background (lookback: {req_hours or 'auto'}h).",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/status")
async def get_system_status():
    """Health check and live system diagnostics for Render healthchecks."""
    db_connected, _, article_count, last_scraped = get_db_status_and_subscriber()
    cfg = get_sources_config(DEFAULT_SOURCES_PATH)
    scheduler_active = _scheduler is not None and _scheduler.running

    return {
        "status": "healthy",
        "database": {
            "connected": db_connected,
            "total_articles": article_count,
            "last_scraped_at": last_scraped,
        },
        "sources_summary": {
            "total_youtube_channels": len(cfg.youtube_channels),
            "active_youtube_channels": len(cfg.get_active_youtube_channels()),
            "openai_active": cfg.openai.is_active,
            "anthropic_active": cfg.anthropic.is_active,
        },
        "scheduler": {
            "active": scheduler_active,
            "pipeline_state": _pipeline_state,
        },
    }


def start(
    host: Optional[str] = None,
    port: Optional[int] = None,
    reload: bool = False,
):
    """CLI launcher for Axiom Dispatch Web Configuration Server and Scheduler."""
    import uvicorn

    target_host = host or os.getenv("HOST", "0.0.0.0")
    target_port = port or int(os.getenv("PORT", 8000))
    sched_flag = os.getenv("ENABLE_SCHEDULER", "true")

    print(f"\n" + "=" * 65)
    print(f"🚀 Axiom Dispatch — Cloud Production Portal & 24h Scheduler")
    print(f"   Listening on: http://{target_host}:{target_port}")
    print(f"   Scheduler: {'Active (every 24h)' if sched_flag.lower() in ('true', '1') else 'Disabled'}")
    print(f"   Database: {'Configured via DATABASE_URL' if os.getenv('DATABASE_URL') else 'Local Fallback'}")
    print(f"=" * 65 + "\n")
    uvicorn.run("app.web.server:app", host=target_host, port=target_port, reload=reload)


if __name__ == "__main__":
    start()
