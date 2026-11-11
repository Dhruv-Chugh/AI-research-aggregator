"""Unit tests for Unified Aggregation Runner Pipeline & Database Persistence."""

from datetime import datetime, timedelta, timezone
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import SourcesConfig
from app.database.models import Base
from app.database.repository import NewsRepository
from app.models.anthropic import AnthropicArticle
from app.models.openai import OpenAIArticle
from app.models.youtube import ChannelConfig, VideoTranscript, YouTubeVideo
from app.scrapers.runner import (
    AggregationResult,
    AggregationRunner,
    run_news_aggregation,
)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """In-memory SQLite database session for runner tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def sample_aggregation_data():
    now = datetime.now(timezone.utc)
    transcript = VideoTranscript(
        video_id="vid_1",
        available=True,
        text="Full transcript of Think School analysis on geopolitics.",
        language="English",
        word_count=8,
    )
    vids = [
        YouTubeVideo(
            video_id="vid_1",
            title="Think School Analysis",
            url="https://youtube.com/watch?v=vid_1",
            channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
            channel_name="Think School",
            published_at=now - timedelta(hours=2),
            description="Deep dive case study.",
            thumbnail_url="https://i.ytimg.com/vi/vid_1/hqdefault.jpg",
            transcript=transcript,
        )
    ]
    oai = [
        OpenAIArticle(
            article_id="oai_1",
            title="OpenAI Alignment Research",
            url="https://openai.com/index/oai-alignment",
            published_at=now - timedelta(hours=5),
            summary="New alignment benchmark",
            content="# OpenAI Alignment\nFull markdown research body...",
            word_count=500,
            categories=["Research", "Safety"],
            author="OpenAI Alignment Team",
        )
    ]
    ant = [
        AnthropicArticle(
            article_id="ant_1",
            title="Anthropic Drug Discovery",
            url="https://anthropic.com/research/drug-discovery",
            published_at=now - timedelta(hours=8),
            summary="Accelerating drug discovery",
            content="# Drug Discovery\nAccelerating biological pipelines...",
            word_count=750,
            categories=["Research"],
            feed_source="research",
            author="Anthropic Research",
        )
    ]
    return vids, oai, ant


def test_aggregation_result_properties(sample_aggregation_data):
    """Test counts, summary, and properties on AggregationResult."""
    vids, oai, ant = sample_aggregation_data
    result = AggregationResult(
        hours_back=24,
        youtube_videos=vids,
        openai_articles=oai,
        anthropic_articles=ant,
        db_synced=True,
        db_items_saved=3,
        db_new_items=3,
        db_updated_items=0,
    )

    assert result.total_items == 3
    assert result.has_items is True
    assert result.db_synced is True
    assert result.db_items_saved == 3
    assert result.summary_counts == {
        "youtube": 1,
        "openai": 1,
        "anthropic": 1,
        "total": 3,
    }


def test_aggregation_runner_full_orchestration(sample_aggregation_data):
    """Test AggregationRunner orchestrating all 3 scrapers."""
    vids, oai, ant = sample_aggregation_data

    with patch("app.scrapers.runner.YouTubeService") as mock_yt_cls, \
         patch("app.scrapers.runner.OpenAIScraperService") as mock_oai_cls, \
         patch("app.scrapers.runner.AnthropicScraperService") as mock_ant_cls:

        mock_yt_inst = MagicMock()
        mock_yt_inst.get_recent_videos.return_value = vids
        mock_yt_cls.return_value.__enter__.return_value = mock_yt_inst

        mock_oai_inst = MagicMock()
        mock_oai_inst.get_recent_articles.return_value = oai
        mock_oai_cls.return_value.__enter__.return_value = mock_oai_inst

        mock_ant_inst = MagicMock()
        mock_ant_inst.get_articles.return_value = ant
        mock_ant_cls.return_value.__enter__.return_value = mock_ant_inst

        with AggregationRunner() as runner:
            result = runner.run(
                hours_back=24,
                sync_to_db=False,  # Skip DB in unit test without DB session
            )

        assert result.total_items == 3
        assert len(result.youtube_videos) == 1
        assert len(result.openai_articles) == 1
        assert len(result.anthropic_articles) == 1


def test_aggregation_runner_database_sync(sample_aggregation_data, db_session):
    """Test that runner saves scraped items, rich metadata, and handles deduplication in database."""
    vids, oai, ant = sample_aggregation_data
    repo = NewsRepository(session=db_session)

    with patch("app.scrapers.runner.YouTubeService") as mock_yt_cls, \
         patch("app.scrapers.runner.OpenAIScraperService") as mock_oai_cls, \
         patch("app.scrapers.runner.AnthropicScraperService") as mock_ant_cls:

        mock_yt_inst = MagicMock()
        mock_yt_inst.get_recent_videos.return_value = vids
        mock_yt_cls.return_value.__enter__.return_value = mock_yt_inst

        mock_oai_inst = MagicMock()
        mock_oai_inst.get_recent_articles.return_value = oai
        mock_oai_cls.return_value.__enter__.return_value = mock_oai_inst

        mock_ant_inst = MagicMock()
        mock_ant_inst.get_articles.return_value = ant
        mock_ant_cls.return_value.__enter__.return_value = mock_ant_inst

        # First run: should persist 3 new items into DB
        with AggregationRunner(db_repo=repo) as runner:
            res1 = runner.run(hours_back=24, sync_to_db=True)

        assert res1.db_synced is True
        assert res1.db_items_saved == 3
        assert res1.db_new_items == 3
        assert res1.db_updated_items == 0
        assert repo.articles.count() == 3

        # Verify YouTube article and metadata
        yt_db = repo.articles.get_by_external_id("vid_1")
        assert yt_db is not None
        assert yt_db.title == "Think School Analysis"
        assert yt_db.content_type == "transcript"
        assert "Full transcript of Think School" in yt_db.raw_content
        assert yt_db.metadata_json["thumbnail_url"] == "https://i.ytimg.com/vi/vid_1/hqdefault.jpg"
        assert yt_db.metadata_json["has_transcript"] is True

        # Verify OpenAI article and metadata
        oai_db = repo.articles.get_by_external_id("oai_1")
        assert oai_db is not None
        assert oai_db.author == "OpenAI Alignment Team"
        assert "Research" in oai_db.category
        assert oai_db.word_count == 500
        assert "Full markdown research body" in oai_db.raw_content

        # Verify Anthropic article and metadata
        ant_db = repo.articles.get_by_external_id("ant_1")
        assert ant_db is not None
        assert ant_db.category == "research"
        assert ant_db.word_count == 750

        # Second run: Same items should be deduplicated (0 new, 3 updated)
        with AggregationRunner(db_repo=repo) as runner:
            res2 = runner.run(hours_back=24, sync_to_db=True)

        assert res2.db_synced is True
        assert res2.db_items_saved == 3
        assert res2.db_new_items == 0
        assert res2.db_updated_items == 3
        # Total count must NOT increase
        assert repo.articles.count() == 3


def test_aggregation_runner_toggling_sources(sample_aggregation_data):
    """Verify that disabling specific sources skips their execution."""
    vids, oai, ant = sample_aggregation_data

    with patch("app.scrapers.runner.YouTubeService") as mock_yt_cls, \
         patch("app.scrapers.runner.OpenAIScraperService") as mock_oai_cls, \
         patch("app.scrapers.runner.AnthropicScraperService") as mock_ant_cls:

        mock_oai_inst = MagicMock()
        mock_oai_inst.get_recent_articles.return_value = oai
        mock_oai_cls.return_value.__enter__.return_value = mock_oai_inst

        with AggregationRunner() as runner:
            # Disable YouTube and Anthropic, skip DB
            result = runner.run(
                hours_back=24,
                include_youtube=False,
                include_openai=True,
                include_anthropic=False,
                sync_to_db=False,
            )

        mock_yt_cls.assert_not_called()
        mock_ant_cls.assert_not_called()
        assert len(result.youtube_videos) == 0
        assert len(result.openai_articles) == 1
        assert len(result.anthropic_articles) == 0


def test_aggregation_runner_error_resilience(sample_aggregation_data):
    """Ensure that if one source fails, others succeed and error is captured as warning."""
    vids, oai, ant = sample_aggregation_data

    with patch("app.scrapers.runner.YouTubeService") as mock_yt_cls, \
         patch("app.scrapers.runner.OpenAIScraperService") as mock_oai_cls, \
         patch("app.scrapers.runner.AnthropicScraperService") as mock_ant_cls:

        mock_yt_cls.side_effect = Exception("YouTube connection timed out")

        mock_oai_inst = MagicMock()
        mock_oai_inst.get_recent_articles.return_value = oai
        mock_oai_cls.return_value.__enter__.return_value = mock_oai_inst

        mock_ant_inst = MagicMock()
        mock_ant_inst.get_articles.return_value = ant
        mock_ant_cls.return_value.__enter__.return_value = mock_ant_inst

        with AggregationRunner() as runner:
            result = runner.run(hours_back=24, sync_to_db=False)

        assert len(result.youtube_videos) == 0
        assert len(result.openai_articles) == 1
        assert len(result.anthropic_articles) == 1
        assert len(result.warnings) == 1
        assert "YouTube aggregation failed" in result.warnings[0]


def test_run_news_aggregation_streamlined_function(sample_aggregation_data):
    """Verify run_news_aggregation helper function."""
    vids, oai, ant = sample_aggregation_data

    with patch("app.scrapers.runner.AggregationRunner.run") as mock_run:
        mock_run.return_value = AggregationResult(
            hours_back=24,
            youtube_videos=vids,
            openai_articles=oai,
            anthropic_articles=ant,
            db_synced=True,
            db_items_saved=3,
        )

        res = run_news_aggregation(hours_back=24)
        assert res.total_items == 3
        assert res.db_synced is True
        mock_run.assert_called_once()


def test_aggregation_runner_expands_empty_short_window(sample_aggregation_data):
    """Verify expand_if_empty=True widens 24h to 168h when initial scrape finds 0 items."""
    vids, oai, ant = sample_aggregation_data

    with patch("app.scrapers.runner.YouTubeService") as mock_yt_cls, \
         patch("app.scrapers.runner.OpenAIScraperService") as mock_oai_cls, \
         patch("app.scrapers.runner.AnthropicScraperService") as mock_ant_cls:

        mock_yt_inst = MagicMock()
        mock_yt_inst.get_recent_videos.side_effect = [[], vids]
        mock_yt_cls.return_value.__enter__.return_value = mock_yt_inst

        mock_oai_inst = MagicMock()
        mock_oai_inst.get_recent_articles.side_effect = [[], oai]
        mock_oai_cls.return_value.__enter__.return_value = mock_oai_inst

        mock_ant_inst = MagicMock()
        mock_ant_inst.get_articles.side_effect = [[], ant]
        mock_ant_cls.return_value.__enter__.return_value = mock_ant_inst

        with AggregationRunner() as runner:
            res = runner.run(hours_back=24, sync_to_db=False, expand_if_empty=True)

        assert res.hours_back == 168
        assert res.total_items == 3

