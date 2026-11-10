"""Unit tests for EmailDeliveryAgent end-to-end integration."""

from datetime import date, datetime, timezone
import os
from typing import Generator
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.email_agent import EmailDeliveryAgent
from app.database.models import Article, ArticleDigest, Base, Source, Subscriber, UserDigestRanking
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
def populated_rankings_db(db_session: Session) -> dict:
    """Populate database with subscriber, articles, digests, and user rankings."""
    repo = NewsRepository(session=db_session)
    now = datetime.now(timezone.utc)
    b_date = now.date()

    # 1. Subscriber
    sub = repo.subscribers.create(
        email="subscriber@example.com",
        name="Research Practitioner",
        bio_or_role="AI Software Architect & Full-Stack Engineer",
        expertise_level="advanced",
        interests=["Autonomous Agents", "LLMs", "Reasoning"],
        is_active=True,
    )

    # 2. Source & Articles
    src = repo.sources.create("Anthropic", "anthropic", "https://anthropic.com")
    art1 = repo.articles.create(source_id=src.id, external_id="art-1", title="Claude for Science", url="https://anthropic.com/1", published_at=now)
    art2 = repo.articles.create(source_id=src.id, external_id="art-2", title="Daybreak Red Cyber", url="https://openai.com/2", published_at=now)

    # 3. Digest summaries
    d1 = repo.article_digests.create(
        article_id=art1.id,
        title="Claude for Science: Task Level Acceleration",
        url="https://anthropic.com/1",
        source_name="Anthropic",
        source_type="anthropic",
        summary="A half-page analysis of reinforcement learning task acceleration.",
        problem_statement="Scientific discovery bottleneck.",
        solution_statement="Autonomous agent simulation loops.",
        key_insights=["RL enables task scaling.", "Cognitive labor acceleration."],
    )
    d2 = repo.article_digests.create(
        article_id=art2.id,
        title="Daybreak Red Cyber Security",
        url="https://openai.com/2",
        source_name="OpenAI",
        source_type="openai",
        summary="GovCloud cyber security deployment.",
        problem_statement="Vulnerability analysis.",
        solution_statement="GPT-5.6 Cyber guardrails.",
        key_insights=["Specialized alignment reduces false refusals."],
    )

    # 4. User rankings (1 featured, 1 quick bite)
    repo.user_rankings.create(
        subscriber_id=sub.id,
        digest_id=d1.id,
        rank_position=1,
        rank_score=96.0,
        relevance_reasoning="Direct alignment with autonomous agents & reasoning.",
        key_takeaways_for_user=["Takeaway 1", "Takeaway 2"],
        batch_date=b_date,
        tier="featured",
    )
    repo.user_rankings.create(
        subscriber_id=sub.id,
        digest_id=d2.id,
        rank_position=2,
        rank_score=72.0,
        relevance_reasoning="Ecosystem awareness for cloud security.",
        key_takeaways_for_user=["Takeaway 3"],
        batch_date=b_date,
        tier="quick_bite",
        quick_summary="A concise 3-line quick summary of GovCloud cyber security deployment.",
    )

    db_session.commit()
    return {"subscriber": sub, "batch_date": b_date}


def test_email_agent_run_daily_delivery(db_session: Session, populated_rankings_db, tmp_path):
    """Verify EmailDeliveryAgent successfully generates PDF & HTML and reports batch success."""
    b_date = populated_rankings_db["batch_date"]
    export_dir = str(tmp_path / "briefings")

    with EmailDeliveryAgent(session=db_session) as agent:
        report = agent.run_daily_delivery(
            batch_date=b_date,
            send_live=False,
            export_dir=export_dir,
        )

        assert report.subscribers_targeted == 1
        assert report.briefings_generated == 1
        assert report.exports_saved == 1
        assert len(report.errors) == 0

        res = report.results[0]
        assert res["email"] == "subscriber@example.com"
        assert res["success"] is True
        assert os.path.exists(res["pdf_path"])
        assert os.path.exists(res["html_path"])
        assert os.path.getsize(res["pdf_path"]) > 3000

