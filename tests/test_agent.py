"""Unit tests for AI News Digest Agent, Dual-Provider LLM Client, and 'digest' Table."""

from datetime import datetime, timezone
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.llm_client import DualLLMClient, clean_json_text, estimate_tokens
from app.agent.schemas import AgentRunReport, ArticleDigestOutput, DigestResult
from app.agent.summarizer import DigestAgent
from app.database.models import Article, ArticleDigest, Base, Source
from app.database.repository import NewsRepository


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


@pytest.fixture
def sample_sources(db_session: Session) -> dict[str, Source]:
    """Create test sources for all 3 supported providers."""
    repo = NewsRepository(session=db_session)
    yt = repo.sources.create(
        name="Figuring Out by Raj Shamani",
        source_type="youtube",
        url="https://youtube.com/@rajshamani",
        channel_id="UCzwCEE_PchiBULMnAJqhGVg",
    )
    oa = repo.sources.create(
        name="OpenAI",
        source_type="openai",
        url="https://openai.com/news",
        feed_url="https://openai.com/news/rss.xml",
    )
    an = repo.sources.create(
        name="Anthropic",
        source_type="anthropic",
        url="https://anthropic.com/news",
    )
    db_session.commit()
    return {"youtube": yt, "openai": oa, "anthropic": an}


# =========================================================================
# 1. Schema & Token Estimator Tests
# =========================================================================


def test_article_digest_output_schema():
    """Verify ArticleDigestOutput validation and serialization."""
    data = {
        "title": "Sam Altman on AGI Future",
        "summary": "A detailed half-page summary of the conversation discussing superintelligence...",
        "problem_statement": "Energy compute bottlenecks and infrastructure scaling.",
        "solution_statement": "Massive capital deployment in datacenters and modular nuclear power.",
        "key_insights": [
            "Compute scaling curves remain intact.",
            "Agents will revolutionize software development within 24 months.",
        ],
    }
    output = ArticleDigestOutput.model_validate(data)
    assert output.title == "Sam Altman on AGI Future"
    assert len(output.key_insights) == 2
    assert "energy" in output.problem_statement.lower()


def test_clean_json_text():
    """Verify markdown code fence stripping from LLM responses."""
    raw = "```json\n{\"title\": \"Test\", \"summary\": \"Ok\", \"problem_statement\": \"P\", \"solution_statement\": \"S\", \"key_insights\": []}\n```"
    cleaned = clean_json_text(raw)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")


def test_estimate_tokens():
    """Verify token estimation heuristics."""
    short_text = "Hello world this is a test"
    assert estimate_tokens(short_text) > 0

    long_text = "word " * 1000
    est = estimate_tokens(long_text)
    assert 1000 <= est <= 1500


# =========================================================================
# 2. DualLLMClient Fallback & Routing Tests
# =========================================================================


def test_dual_llm_client_routing_large_tokens():
    """Verify that inputs exceeding the token threshold route to Gemini first."""
    client = DualLLMClient(
        openai_api_key="mock-openai-key",
        gemini_api_key="mock-gemini-key",
        openai_token_threshold=100,  # low threshold for testing
        preferred_provider="auto",
        allow_offline_fallback=False,
    )

    # 500 words will estimate to ~665 tokens (> 100)
    large_input = "AI scaling laws and frontier architectures " * 500

    with patch.object(client, "_call_gemini_interactions") as mock_gemini, \
         patch.object(client, "_call_openai_responses") as mock_openai:
        dummy_output = ArticleDigestOutput(
            title="T", summary="S", problem_statement="P", solution_statement="Sol", key_insights=[]
        )
        mock_gemini.return_value = (dummy_output, "gemini:interactions/gemini-2.5-flash", 800)

        out, model, tokens = client.generate_digest(large_input, {"title": "Test Large"})

        assert mock_gemini.called
        assert not mock_openai.called
        assert "gemini" in model


def test_dual_llm_client_fallback_openai_to_gemini():
    """Verify seamless fallback from OpenAI to Gemini when OpenAI encounters an error."""
    client = DualLLMClient(
        openai_api_key="mock-openai-key",
        gemini_api_key="mock-gemini-key",
        openai_token_threshold=5000,
        preferred_provider="auto",
        allow_offline_fallback=False,
    )

    small_input = "Short article content."

    with patch.object(client, "_call_openai_responses", side_effect=Exception("OpenAI RateLimit / Token Exceeded")), \
         patch.object(client, "_call_gemini_interactions") as mock_gemini:
        dummy_output = ArticleDigestOutput(
            title="Gemini Title",
            summary="Gemini summary half page...",
            problem_statement="Prob",
            solution_statement="Sol",
            key_insights=["Insight 1"],
        )
        mock_gemini.return_value = (dummy_output, "gemini:interactions/gemini-2.5-flash", 350)

        out, model, tokens = client.generate_digest(small_input, {"title": "Short Article"})

        assert mock_gemini.called
        assert out.title == "Gemini Title"
        assert "gemini" in model


def test_dual_llm_client_fallback_gemini_to_openai():
    """Verify fallback from Gemini to OpenAI when Gemini errors."""
    client = DualLLMClient(
        openai_api_key="mock-openai-key",
        gemini_api_key="mock-gemini-key",
        preferred_provider="gemini",
        allow_offline_fallback=False,
    )

    with patch.object(client, "_call_gemini_interactions", side_effect=Exception("Gemini Service Unavailable")), \
         patch.object(client, "_call_openai_responses") as mock_openai:
        dummy_output = ArticleDigestOutput(
            title="OpenAI Title",
            summary="OpenAI half-page summary...",
            problem_statement="Prob",
            solution_statement="Sol",
            key_insights=["Insight 1"],
        )
        mock_openai.return_value = (dummy_output, "openai:responses/gpt-4o-mini", 420)

        out, model, tokens = client.generate_digest("Some content", {"title": "Test"})

        assert mock_openai.called
        assert out.title == "OpenAI Title"
        assert "openai" in model


def test_dual_llm_client_offline_fallback(monkeypatch):
    """Verify intelligent deterministic fallback when no API keys are present."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = DualLLMClient(
        openai_api_key=None,
        gemini_api_key=None,
        allow_offline_fallback=True,
    )

    content = "OpenAI announced a new system for enterprise developers to build agents reliably.\n" * 10
    out, model, tokens = client.generate_digest(content, {"title": "Offline Test", "source_name": "OpenAI"})

    assert out.title == "Offline Test"
    assert len(out.summary) > 100
    assert len(out.key_insights) >= 3
    assert "deterministic" in model


# =========================================================================
# 3. 'digest' Table Model & Repository Tests
# =========================================================================


def test_article_digest_repository_crud(db_session: Session, sample_sources):
    """Verify ArticleDigestRepository operations on table 'digest'."""
    repo = NewsRepository(session=db_session)

    # 1. Create an article
    art = repo.articles.upsert(
        external_id="test-art-1",
        title="AI Breakthrough in Reasoning",
        url="https://openai.com/index/reasoning-breakthrough",
        published_at=datetime.now(timezone.utc),
        source_id=sample_sources["openai"].id,
        raw_content="Full content about reasoning...",
        word_count=500,
    )[0]
    db_session.commit()

    # 2. Create digest entry
    digest_entry = repo.article_digests.create(
        article_id=art.id,
        title=art.title,
        url=art.url,
        source_name=sample_sources["openai"].name,
        source_type="openai",
        summary="This half-page summary details the reasoning paradigm...",
        problem_statement="LLMs struggle with complex multi-step reasoning.",
        solution_statement="Reinforcement learning over chain-of-thought tokens.",
        key_insights=["Test-time compute scaling is effective."],
        model_used="openai:responses/gpt-4o-mini",
        tokens_used=450,
    )
    db_session.commit()

    assert digest_entry.id is not None
    assert digest_entry.__tablename__ == "digest"
    assert digest_entry.url == art.url

    # 3. Retrieve
    retrieved = repo.article_digests.get_by_article_id(art.id)
    assert retrieved is not None
    assert retrieved.title == art.title
    assert "Test-time compute" in retrieved.key_insights[0]

    # 4. Upsert (update)
    updated, created = repo.article_digests.upsert(
        article_id=art.id,
        title="Updated Reasoning Title",
        url=art.url,
        source_name="OpenAI",
        source_type="openai",
        summary="Updated summary...",
        model_used="gemini:interactions/gemini-2.5-flash",
    )
    db_session.commit()
    assert not created
    assert updated.title == "Updated Reasoning Title"
    assert updated.model_used == "gemini:interactions/gemini-2.5-flash"

    # 5. Verify relationship
    assert art.digest_entry is not None
    assert art.digest_entry.id == updated.id

    # 6. Delete
    deleted = repo.article_digests.delete_by_article_id(art.id)
    db_session.commit()
    assert deleted
    assert repo.article_digests.get_by_article_id(art.id) is None


# =========================================================================
# 4. DigestAgent End-to-End Processing Tests
# =========================================================================


def test_digest_agent_process_all_sources(db_session: Session, sample_sources, monkeypatch):
    """Test DigestAgent processing articles from YouTube, OpenAI, and Anthropic."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    repo = NewsRepository(session=db_session)

    # Ingest 1 article from each of the 3 sources
    art_yt = repo.articles.upsert(
        external_id="yt-video-1",
        title="Sam Altman On Superintelligence",
        url="https://www.youtube.com/watch?v=TjPPIf-M6KE",
        published_at=datetime.now(timezone.utc),
        source_id=sample_sources["youtube"].id,
        raw_content="Transcript of the podcast discussing compute, energy, and robotics...",
        content_type="transcript",
        word_count=12000,
    )[0]

    art_oa = repo.articles.upsert(
        external_id="oa-art-1",
        title="OpenAI Acquires Pierre Computer Company",
        url="https://openai.com/index/openai-to-acquire-pierre",
        published_at=datetime.now(timezone.utc),
        source_id=sample_sources["openai"].id,
        raw_content="OpenAI today announced the acquisition of Pierre...",
        content_type="article",
        word_count=495,
    )[0]

    art_an = repo.articles.upsert(
        external_id="an-art-1",
        title="Claude for Scientific Discovery",
        url="https://www.anthropic.com/research/how-researchers-use-claude",
        published_at=datetime.now(timezone.utc),
        source_id=sample_sources["anthropic"].id,
        raw_content="Researchers across biology and chemistry utilize Claude...",
        content_type="article",
        word_count=7600,
    )[0]
    db_session.commit()

    # Run agent with deterministic mock LLM
    agent = DigestAgent(session=db_session, preferred_provider="auto")
    report = agent.run(limit=10)

    assert report.total_articles == 3
    assert report.successful == 3
    assert report.failed == 0

    # Verify all articles marked as summarized
    assert art_yt.is_summarized is True
    assert art_oa.is_summarized is True
    assert art_an.is_summarized is True

    # Verify records stored in table 'digest'
    all_digests = repo.article_digests.get_all()
    assert len(all_digests) == 3

    # Check that URLs match original sources
    urls = {d.url for d in all_digests}
    assert art_yt.url in urls
    assert art_oa.url in urls
    assert art_an.url in urls

    # Check markdown formatting of an entry
    md = all_digests[0].to_markdown()
    assert "## Summary" in md
    assert "### Problem Highlighted" in md
    assert "### Solution & Breakthrough" in md
    assert "### Key Insights & Actionable Takeaways" in md

