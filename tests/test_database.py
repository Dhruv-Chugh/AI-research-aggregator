"""Unit tests for Database Models, Repositories, and Table Initialization."""

from datetime import date, datetime, timedelta, timezone
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.init_db import create_tables, seed_initial_data
from app.database.models import Article, Base, Digest, Source, Subscriber
from app.database.repository import (
    ArticleRepository,
    DigestRepository,
    NewsRepository,
    SourceRepository,
    SubscriberRepository,
)
from app.models.anthropic import AnthropicArticle
from app.models.openai import OpenAIArticle
from app.models.youtube import VideoTranscript, YouTubeVideo


@pytest.fixture
def test_engine():
    """Create an isolated, in-memory SQLite engine for tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(test_engine) -> Generator[Session, None, None]:
    """Yield a database session bound to the in-memory test engine."""
    session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


# -------------------------------------------------------------------------
# 1. Source Repository Tests
# -------------------------------------------------------------------------


def test_source_repository_crud(db_session: Session):
    repo = SourceRepository(db_session)

    # 1. Create
    src = repo.create(
        name="Think School",
        source_type="youtube",
        url="https://www.youtube.com/@thinkschool",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        handle="@thinkschool",
        is_active=True,
    )
    db_session.commit()
    assert src.id is not None
    assert src.name == "Think School"

    # 2. Get by ID, Name, Channel ID
    assert repo.get_by_id(src.id).name == "Think School"
    assert repo.get_by_name("Think School").channel_id == "UCKZozRVHRYsYHGEyNKuhhdA"
    assert repo.get_by_channel_id("UCKZozRVHRYsYHGEyNKuhhdA").handle == "@thinkschool"

    # 3. Active filter
    repo.create(
        name="Inactive Channel",
        source_type="youtube",
        url="https://youtube.com/inactive",
        is_active=False,
    )
    db_session.commit()

    active_sources = repo.get_active()
    assert len(active_sources) == 1
    assert active_sources[0].name == "Think School"

    # 4. Update
    updated = repo.update(src.id, handle="@thinkschool_updated")
    db_session.commit()
    assert updated.handle == "@thinkschool_updated"

    # 5. Upsert existing
    upserted, created = repo.upsert(
        name="Think School",
        source_type="youtube",
        url="https://www.youtube.com/@thinkschool",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        handle="@thinkschool_final",
    )
    db_session.commit()
    assert created is False
    assert upserted.id == src.id
    assert upserted.handle == "@thinkschool_final"

    # 6. Delete
    assert repo.delete(src.id) is True
    assert repo.get_by_id(src.id) is None


# -------------------------------------------------------------------------
# 2. Article Repository Tests
# -------------------------------------------------------------------------


def test_article_repository_crud_and_dedup(db_session: Session):
    src_repo = SourceRepository(db_session)
    art_repo = ArticleRepository(db_session)

    source = src_repo.create(
        name="OpenAI",
        source_type="openai",
        url="https://openai.com/news",
    )
    db_session.commit()

    now = datetime.now(timezone.utc)

    # 1. Create Article
    article = art_repo.create(
        source_id=source.id,
        external_id="oai-123",
        title="OpenAI Breakthrough",
        url="https://openai.com/news/breakthrough",
        published_at=now - timedelta(hours=3),
        summary="A significant research breakthrough.",
        raw_content="Full article body content here.",
        content_type="article",
        word_count=5,
    )
    db_session.commit()

    assert article.id is not None
    assert art_repo.exists("oai-123") is True
    assert art_repo.exists("non-existent") is False

    # 2. Upsert existing (deduplication & update)
    updated_art, created = art_repo.upsert(
        external_id="oai-123",
        title="OpenAI Breakthrough (Revised)",
        url="https://openai.com/news/breakthrough",
        published_at=now - timedelta(hours=3),
        summary="Updated summary.",
    )
    db_session.commit()

    assert created is False
    assert updated_art.id == article.id
    assert updated_art.title == "OpenAI Breakthrough (Revised)"
    assert art_repo.count() == 1

    # 3. Recent articles filter
    art_repo.create(
        source_id=source.id,
        external_id="oai-old",
        title="Old News",
        url="https://openai.com/news/old",
        published_at=now - timedelta(hours=48),
    )
    db_session.commit()

    recent_24h = art_repo.get_recent(hours=24)
    assert len(recent_24h) == 1
    assert recent_24h[0].external_id == "oai-123"

    recent_all = art_repo.get_recent(hours=None)
    assert len(recent_all) == 2

    # 4. Summarization workflow
    unsummarized = art_repo.get_unsummarized()
    assert len(unsummarized) == 2

    art_repo.mark_summarized(article.id, summary="LLM Generated Short Takeaway")
    db_session.commit()

    refreshed = art_repo.get_by_id(article.id)
    assert refreshed.is_summarized is True
    assert refreshed.summary == "LLM Generated Short Takeaway"


# -------------------------------------------------------------------------
# 3. Digest Repository Tests
# -------------------------------------------------------------------------


def test_digest_repository_crud(db_session: Session):
    repo = DigestRepository(db_session)
    today = date.today()

    # 1. Create Digest
    digest = repo.create(
        digest_date=today,
        title="AI News Daily Digest",
        content_markdown="# Daily Digest\n- AI updates",
        content_html="<h1>Daily Digest</h1>",
        article_count=5,
    )
    db_session.commit()

    assert digest.id is not None
    assert digest.sent_at is None

    # 2. Get by date & latest
    by_date = repo.get_by_date(today)
    assert by_date is not None
    assert by_date.id == digest.id

    latest = repo.get_latest()
    assert latest.title == "AI News Daily Digest"

    # 3. Mark sent
    repo.mark_sent(digest.id)
    db_session.commit()

    updated = repo.get_by_id(digest.id)
    assert updated.sent_at is not None


# -------------------------------------------------------------------------
# 4. Subscriber Repository Tests
# -------------------------------------------------------------------------


def test_subscriber_repository_crud(db_session: Session):
    repo = SubscriberRepository(db_session)

    # 1. Create Subscriber
    sub = repo.create(email="DhruvChugh2007@gmail.com", name="Dhruv Chugh")
    db_session.commit()

    # Automatic lowercasing
    assert sub.email == "dhruvchugh2007@gmail.com"
    assert sub.is_active is True

    # 2. Lookup
    found = repo.get_by_email("dhruvchugh2007@gmail.com")
    assert found is not None
    assert found.name == "Dhruv Chugh"

    # 3. Toggle Active
    repo.toggle_active("dhruvchugh2007@gmail.com", is_active=False)
    db_session.commit()

    assert len(repo.get_active()) == 0

    repo.toggle_active("dhruvchugh2007@gmail.com", is_active=True)
    db_session.commit()
    assert len(repo.get_active()) == 1

    # 4. Delete
    assert repo.delete("dhruvchugh2007@gmail.com") is True
    assert repo.get_by_email("dhruvchugh2007@gmail.com") is None


# -------------------------------------------------------------------------
# 5. Master NewsRepository & Pydantic Bridge Tests
# -------------------------------------------------------------------------


def test_news_repository_bridges(db_session: Session):
    repo = NewsRepository(session=db_session)

    now = datetime.now(timezone.utc)

    # 1. Save YouTubeVideo Pydantic model
    transcript_obj = VideoTranscript(
        video_id="yt-test-001",
        available=True,
        text="Hello and welcome to this video transcript.",
        word_count=7,
        language="English",
    )
    yt_vid = YouTubeVideo(
        video_id="yt-test-001",
        title="Think School Case Study",
        url="https://www.youtube.com/watch?v=yt-test-001",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
        description="Great case study.",
        transcript=transcript_obj,
    )
    art, created = repo.save_youtube_video(yt_vid)
    db_session.commit()

    assert created is True
    assert art.external_id == "yt-test-001"
    assert art.content_type == "transcript"
    assert art.raw_content == yt_vid.transcript.text

    # 2. Save OpenAIArticle Pydantic model
    oai_art = OpenAIArticle(
        article_id="oai-test-001",
        title="Introducing GPT Next",
        url="https://openai.com/index/introducing-gpt-next",
        published_at=now,
        author="OpenAI Team",
        categories=["Research"],
        summary="Next gen AI model",
        content="# Full Content\nDetails of the model...",
        word_count=500,
    )
    art2, created2 = repo.save_openai_article(oai_art)
    db_session.commit()

    assert created2 is True
    assert art2.external_id == "oai-test-001"
    assert art2.word_count == 500

    # 3. Save AnthropicArticle Pydantic model
    ant_art = AnthropicArticle(
        article_id="ant-test-001",
        title="Claude 3.7 Sonnet Released",
        url="https://www.anthropic.com/news/claude-3-7-sonnet",
        published_at=now,
        categories=["News"],
        feed_source="news",
        summary="Anthropic announces new reasoning capabilities.",
        content="# Claude 3.7\nBreakthrough reasoning architecture...",
        word_count=800,
    )
    art3, created3 = repo.save_anthropic_article(ant_art)
    db_session.commit()

    assert created3 is True
    assert art3.external_id == "ant-test-001"
    assert art3.category == "news"

    # Total articles in DB
    assert repo.articles.count() == 3


# -------------------------------------------------------------------------
# 6. Table Creation and Seed Script Test
# -------------------------------------------------------------------------


def test_create_tables_and_seed(test_engine):
    # Recreate tables and seed default data
    create_tables(target_engine=test_engine, drop_first=True, seed_defaults=True)

    session_factory = sessionmaker(bind=test_engine)
    with session_factory() as session:
        repo = NewsRepository(session=session)

        # Sources should be seeded from config/sources.json
        sources = repo.sources.get_all()
        assert len(sources) >= 2
        names = [s.name for s in sources]
        assert "Think School" in names
        assert "Figuring Out by Raj Shamani" in names

        # Subscriber should be seeded
        sub = repo.subscribers.get_by_email("dhruvchugh2007@gmail.com")
        assert sub is not None
        assert sub.is_active is True


# -------------------------------------------------------------------------
# 7. Model Serialization (to_dict) & Repository Stats
# -------------------------------------------------------------------------


def test_models_to_dict_and_repo_stats(db_session: Session):
    repo = NewsRepository(session=db_session)
    now = datetime.now(timezone.utc)

    # 1. Source to_dict
    src = repo.sources.create(
        name="Think School",
        source_type="youtube",
        url="https://youtube.com/@thinkschool",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        handle="@thinkschool",
    )
    db_session.commit()
    src_dict = src.to_dict()
    assert src_dict["name"] == "Think School"
    assert src_dict["channel_id"] == "UCKZozRVHRYsYHGEyNKuhhdA"
    assert "created_at" in src_dict

    # 2. Article to_dict & source filter
    art = repo.articles.create(
        source_id=src.id,
        external_id="vid-dict-1",
        title="Think School Analysis",
        url="https://youtube.com/watch?v=vid-dict-1",
        published_at=now,
        raw_content="Transcript content",
        content_type="transcript",
    )
    db_session.commit()
    art_dict = art.to_dict()
    assert art_dict["external_id"] == "vid-dict-1"
    assert art_dict["content_type"] == "transcript"

    by_src = repo.articles.get_by_source_id(src.id)
    assert len(by_src) == 1
    assert by_src[0].id == art.id

    # 3. Digest Candidates
    candidates = repo.get_digest_candidates(hours=24)
    assert len(candidates) == 1
    assert candidates[0].id == art.id

    # 4. System Stats
    st = repo.stats()
    assert st["sources"]["total"] == 1
    assert st["sources"]["active"] == 1
    assert st["articles"]["total"] == 1
    assert st["articles"]["unsummarized"] == 1
    assert st["articles"]["summarized"] == 0




# -------------------------------------------------------------------------
# 8. YouTube Shorts Exclusion & Database Cleanup Tests
# -------------------------------------------------------------------------


def test_remove_shorts_from_database(db_session: Session):
    """Verify that repo.remove_shorts() cleans up shorts and preserves long videos."""
    repo = NewsRepository(session=db_session)
    now = datetime.now(timezone.utc)

    # 1. Add normal video
    normal_art = repo.articles.create(
        external_id="long-video-1",
        title="Comprehensive AI Roadmap",
        url="https://www.youtube.com/watch?v=long-video-1",
        published_at=now,
        category="YouTube",
        content_type="transcript",
        raw_content="Long transcript content...",
    )

    # 2. Add Short via URL
    short_art_1 = repo.articles.create(
        external_id="short-video-1",
        title="Quick AI Tip",
        url="https://www.youtube.com/shorts/short-video-1",
        published_at=now,
        category="YouTube",
        content_type="video",
    )

    # 3. Add Short via hashtag in title
    short_art_2 = repo.articles.create(
        external_id="short-video-2",
        title="Mindblowing Claude feature #shorts",
        url="https://www.youtube.com/watch?v=short-video-2",
        published_at=now,
        category="YouTube",
        content_type="transcript",
    )

    # 4. Add Short via metadata
    short_art_3 = repo.articles.create(
        external_id="short-video-3",
        title="Fast tip",
        url="https://www.youtube.com/watch?v=short-video-3",
        published_at=now,
        category="YouTube",
        metadata_json={"is_short": True},
    )
    db_session.commit()

    assert repo.articles.count() == 4

    # Execute removal
    removed_count = repo.remove_shorts()
    assert removed_count == 3
    assert repo.articles.count() == 1

    remaining = repo.articles.get_by_external_id("long-video-1")
    assert remaining is not None
    assert remaining.title == "Comprehensive AI Roadmap"


def test_save_youtube_video_skips_shorts(db_session: Session):
    """save_youtube_video must refuse to persist YouTube Shorts."""
    repo = NewsRepository(session=db_session)
    now = datetime.now(timezone.utc)

    short_vid = YouTubeVideo(
        video_id="short_reject_1",
        title="Short Tip #shorts",
        url="https://www.youtube.com/shorts/short_reject_1",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
        is_short=True,
    )

    art, created = repo.save_youtube_video(short_vid)
    assert art is None
    assert created is False
    assert repo.articles.count() == 0
