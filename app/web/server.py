"""FastAPI Web Configuration Server for Axiom Dispatch.

Provides an authentic Google/Microsoft Forms-styled configuration interface for:
- Monitoring, adding, and removing YouTube channels
- Calibrating subscriber profile and technical expertise
- Defining multi-agent LLM ranking directives
- Toggling frontier lab feeds (OpenAI & Anthropic)
- Managing delivery settings and triggering pipeline test runs
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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, status
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

# Default subscriber preferences
DEFAULT_SUBSCRIBER = {
    "email": "dhruvchugh2007@gmail.com",
    "name": "Dhruv Chugh",
    "is_active": True,
    "bio_or_role": "AI Software Architect & Full-Stack Engineer",
    "expertise_level": "advanced",
    "interests": [
        "Autonomous Multi-Agent Systems",
        "Inference Scaling & LLM Reasoning",
        "Foundation Model Alignment & Post-Training",
        "Distributed Neural Architectures & Compute Scaling",
        "CUDA, Triton & Hardware Acceleration",
    ],
    "preferred_sources": [
        "Think School",
        "Figuring Out by Raj Shamani",
        "OpenAI",
        "Anthropic",
    ],
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


def get_db_status_and_subscriber() -> tuple[bool, Dict[str, Any], int, Optional[str]]:
    """Probe database status and retrieve subscriber profile if available."""
    try:
        from app.database import check_connection, get_db_session
        from app.database.models import Article, Subscriber
        from sqlalchemy import func, select

        if not check_connection():
            return False, load_local_preferences(), 0, None

        with get_db_session() as session:
            # Query article count
            count_stmt = select(func.count(Article.id))
            article_count = session.execute(count_stmt).scalar() or 0

            # Query last article scrape timestamp
            latest_stmt = select(func.max(Article.created_at))
            latest_dt = session.execute(latest_stmt).scalar()
            latest_str = latest_dt.isoformat() if latest_dt else None

            # Retrieve subscriber
            pref = load_local_preferences()
            sub_email = pref.get("email", DEFAULT_SUBSCRIBER["email"])
            sub_stmt = select(Subscriber).where(Subscriber.email == sub_email)
            subscriber = session.execute(sub_stmt).scalar_one_or_none()

            if subscriber:
                sub_data = {
                    "email": subscriber.email,
                    "name": subscriber.name or pref.get("name"),
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

            return True, pref, article_count, latest_str
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
    logger.info("Axiom Dispatch Web Configuration Service initialized.")
    yield
    logger.info("Axiom Dispatch Web Configuration Service stopped.")


app = FastAPI(
    title="Axiom Dispatch Configuration Portal",
    description="Authentic Google/Microsoft Forms styled settings portal for AI Research Aggregator.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- HTML Frontend Route ---

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

    return {
        "status": "success",
        "system": {
            "db_connected": db_connected,
            "total_articles": article_count,
            "last_scraped_at": last_scraped,
            "server_time": datetime.now(timezone.utc).isoformat(),
        },
        "sources": {
            "youtube_channels": [c.model_dump() for c in sources_cfg.youtube_channels],
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


@app.get("/api/status")
async def get_system_status():
    """Health check and live system diagnostics."""
    db_connected, _, article_count, last_scraped = get_db_status_and_subscriber()
    cfg = get_sources_config(DEFAULT_SOURCES_PATH)

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
    }


def start(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """CLI launcher for Axiom Dispatch Web Configuration Server."""
    import uvicorn

    print(f"\n" + "=" * 65)
    print(f"🚀 Axiom Dispatch — Web Configuration Portal")
    print(f"   Listening on: http://{host}:{port}")
    print(f"   Google/Microsoft Forms Interface Active")
    print(f"=" * 65 + "\n")
    uvicorn.run("app.web.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    start()
