"""Unit and Integration Tests for Axiom Dispatch Web Configuration Server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.config import ChannelConfig, SourcesConfig, DEFAULT_SOURCES_PATH
from app.web.server import app, load_local_preferences, save_local_preferences


def _run_async(coro):
    """Helper to run coroutines in sync test functions without anyio threading issues."""
    return asyncio.run(coro)


def test_healthcheck():
    """Verify GET /healthz returns 200 OK for Render healthcheck."""
    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/healthz")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"

    _run_async(_test())


def test_serve_form_ui():
    """Verify that root endpoint serves the Google/Microsoft Forms HTML page."""
    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            html_content = response.text
            assert "AI Research Intelligence Aggregator" in html_content
            assert "Subscriber Identity & Distribution Appointment" in html_content
            assert "Monitored YouTube Channels" in html_content
            assert "Domain Focus & Technical Interests" in html_content
            assert "Multi-Agent Ranking Directives" in html_content

    _run_async(_test())


def test_get_configuration():
    """Verify GET /api/config returns complete sources and subscriber profile."""
    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/config")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "system" in data
            assert "sources" in data
            assert "subscriber" in data
            assert isinstance(data["sources"]["youtube_channels"], list)
            assert "openai" in data["sources"]
            assert "anthropic" in data["sources"]
            assert "email" in data["subscriber"]

    _run_async(_test())


def test_update_preferences(tmp_path, monkeypatch):
    """Verify POST /api/config saves subscriber updates."""
    test_prefs_path = tmp_path / "test_prefs.json"
    monkeypatch.setattr("app.web.server.PREFERENCES_PATH", test_prefs_path)

    payload = {
        "email": "test_engineer@example.com",
        "name": "Alex Mercer",
        "is_active": True,
        "bio_or_role": "Principal Distributed Systems Engineer",
        "expertise_level": "expert",
        "interests": ["Inference Scaling & LLM Reasoning", "CUDA, Triton & Hardware Acceleration"],
        "preferred_sources": ["OpenAI", "Anthropic"],
        "ranking_instructions": "Focus strictly on GPU memory hierarchy and Triton kernels.",
        "lookback_hours": 48,
        "max_featured_per_source": 3,
        "email_delivery_enabled": False,
    }

    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/config", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["subscriber"]["email"] == "test_engineer@example.com"
            assert data["subscriber"]["expertise_level"] == "expert"

            # Verify persisted in file
            assert test_prefs_path.exists()
            saved = json.loads(test_prefs_path.read_text(encoding="utf-8"))
            assert saved["name"] == "Alex Mercer"
            assert saved["lookback_hours"] == 48

    _run_async(_test())


def test_list_channels():
    """Verify GET /api/channels lists monitored YouTube channels."""
    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/channels")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "channels" in data
            assert isinstance(data["channels"], list)
            assert data["count"] == len(data["channels"])

    _run_async(_test())


def test_add_channel_validation():
    """Verify POST /api/channels rejects empty identifier."""
    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/channels", json={"identifier": ""})
            assert response.status_code == 400

    _run_async(_test())


def test_add_and_delete_channel_mocked(tmp_path, monkeypatch):
    """Verify adding, toggling, and removing a channel using mocked config."""
    test_sources_path = tmp_path / "test_sources.json"
    initial_config = SourcesConfig(
        youtube_channels=[
            ChannelConfig(
                name="Existing Channel",
                channel_id="UC1111111111111111111111",
                handle="@existing",
                url="https://youtube.com/@existing",
                is_active=True,
            )
        ]
    )
    initial_config.save_to_file(test_sources_path)

    monkeypatch.setattr("app.web.server.DEFAULT_SOURCES_PATH", test_sources_path)
    monkeypatch.setattr("app.config.DEFAULT_SOURCES_PATH", test_sources_path)

    # Mock channel resolver so it doesn't make external network calls
    mock_resolved = ChannelConfig(
        name="Two Minute Papers",
        channel_id="UCbfYPyITQ-EY78ICISEnzNw",
        handle="@TwoMinutePapers",
        url="https://youtube.com/@TwoMinutePapers",
        is_active=True,
    )
    monkeypatch.setattr("app.web.server.resolve_channel_info", lambda ident, name=None: mock_resolved)

    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Add channel
            add_resp = await client.post("/api/channels", json={"identifier": "@TwoMinutePapers"})
            assert add_resp.status_code == 200
            assert add_resp.json()["channel"]["name"] == "Two Minute Papers"

            # Verify saved
            updated_cfg = SourcesConfig.load_from_file(test_sources_path)
            assert len(updated_cfg.youtube_channels) == 2

            # 2. Toggle active state
            toggle_resp = await client.patch(
                "/api/channels/UCbfYPyITQ-EY78ICISEnzNw/toggle",
                json={"is_active": False},
            )
            assert toggle_resp.status_code == 200
            assert toggle_resp.json()["is_active"] is False

            toggled_cfg = SourcesConfig.load_from_file(test_sources_path)
            target = next(c for c in toggled_cfg.youtube_channels if c.channel_id == "UCbfYPyITQ-EY78ICISEnzNw")
            assert target.is_active is False

            # 3. Delete channel
            del_resp = await client.delete("/api/channels/UCbfYPyITQ-EY78ICISEnzNw")
            assert del_resp.status_code == 200

            deleted_cfg = SourcesConfig.load_from_file(test_sources_path)
            assert len(deleted_cfg.youtube_channels) == 1
            assert deleted_cfg.youtube_channels[0].channel_id == "UC1111111111111111111111"

    _run_async(_test())


def test_toggle_lab_sources(tmp_path, monkeypatch):
    """Verify PATCH /api/sources/labs updates OpenAI and Anthropic toggles."""
    test_sources_path = tmp_path / "test_sources_labs.json"
    cfg = SourcesConfig.get_default()
    cfg.save_to_file(test_sources_path)

    monkeypatch.setattr("app.web.server.DEFAULT_SOURCES_PATH", test_sources_path)
    monkeypatch.setattr("app.config.DEFAULT_SOURCES_PATH", test_sources_path)

    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.patch("/api/sources/labs", json={"openai_active": False, "anthropic_active": True})
            assert response.status_code == 200
            data = response.json()
            assert data["openai_active"] is False
            assert data["anthropic_active"] is True

            saved = SourcesConfig.load_from_file(test_sources_path)
            assert saved.openai.is_active is False
            assert saved.anthropic.is_active is True

    _run_async(_test())


def test_get_system_status():
    """Verify GET /api/status returns health diagnostics."""
    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/status")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "database" in data
            assert "sources_summary" in data
            assert "total_youtube_channels" in data["sources_summary"]

    _run_async(_test())


def test_get_pipeline_status():
    """Verify GET /api/pipeline/status returns pipeline execution state."""
    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/pipeline/status")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "pipeline" in data
            assert "scheduler" in data
            assert data["scheduler"]["interval_hours"] == 24

    _run_async(_test())


def test_trigger_pipeline_now_mocked(monkeypatch):
    """Verify POST /api/pipeline/trigger-now launches background execution."""
    mock_run = MagicMock(return_value={"status": "success", "total_items": 5, "briefings": 1})
    monkeypatch.setattr("app.web.server.execute_full_pipeline", mock_run)

    async def _test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/pipeline/trigger-now")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "started"
            assert "Autonomous pipeline launched in background" in data["message"]

    _run_async(_test())
