"""CRUD Repositories for AI News Aggregator Database.

Implements the Repository Pattern for:
- SourceRepository: Source management (YouTube channels, RSS feeds)
- ArticleRepository: Article & transcript storage, deduplication, and retrieval
- DigestRepository: Daily summary storage and retrieval
- SubscriberRepository: Recipient email management
- NewsRepository: Unified Unit-of-Work aggregator with context manager support
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.database.models import (
    Article,
    ArticleDigest,
    Digest,
    Source,
    Subscriber,
    UserDigestRanking,
    utc_now,
)
from app.database.session import SessionLocal, get_db_session

logger = logging.getLogger("ai_news.repository")


class BaseRepository:
    """Base repository holding an active SQLAlchemy Session."""

    def __init__(self, session: Session) -> None:
        self.session = session


class SourceRepository(BaseRepository):
    """CRUD operations for content sources (YouTube channels, OpenAI, Anthropic)."""

    def create(
        self,
        name: str,
        source_type: str,
        url: str,
        feed_url: Optional[str] = None,
        channel_id: Optional[str] = None,
        handle: Optional[str] = None,
        is_active: bool = True,
    ) -> Source:
        """Create and persist a new source."""
        source = Source(
            name=name,
            source_type=source_type,
            url=url,
            feed_url=feed_url,
            channel_id=channel_id,
            handle=handle,
            is_active=is_active,
        )
        self.session.add(source)
        self.session.flush()
        logger.debug("Created source: %s (id=%d)", source.name, source.id)
        return source

    def get_by_id(self, source_id: int) -> Optional[Source]:
        """Find source by primary key ID."""
        return self.session.get(Source, source_id)

    def get_by_name(self, name: str) -> Optional[Source]:
        """Find source by unique name."""
        stmt = select(Source).where(Source.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_channel_id(self, channel_id: str) -> Optional[Source]:
        """Find source by YouTube 24-character channel ID."""
        stmt = select(Source).where(Source.channel_id == channel_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_active(self, source_type: Optional[str] = None) -> List[Source]:
        """List all active sources, optionally filtered by source_type."""
        stmt = select(Source).where(Source.is_active.is_(True))
        if source_type:
            stmt = stmt.where(Source.source_type == source_type)
        return list(self.session.execute(stmt).scalars().all())

    def get_all(self) -> List[Source]:
        """List all sources in the system."""
        stmt = select(Source).order_by(Source.name.asc())
        return list(self.session.execute(stmt).scalars().all())

    def update(self, source_id: int, **kwargs: Any) -> Optional[Source]:
        """Update source attributes by ID."""
        source = self.get_by_id(source_id)
        if not source:
            return None
        for key, value in kwargs.items():
            if hasattr(source, key):
                setattr(source, key, value)
        source.updated_at = utc_now()
        self.session.flush()
        return source

    def delete(self, source_id: int) -> bool:
        """Delete a source by ID."""
        source = self.get_by_id(source_id)
        if not source:
            return False
        self.session.delete(source)
        self.session.flush()
        return True

    def upsert(
        self,
        name: str,
        source_type: str,
        url: str,
        feed_url: Optional[str] = None,
        channel_id: Optional[str] = None,
        handle: Optional[str] = None,
        is_active: bool = True,
    ) -> Tuple[Source, bool]:
        """Upsert source by channel_id (if present) or name. Returns (source, is_created)."""
        existing: Optional[Source] = None
        if channel_id:
            existing = self.get_by_channel_id(channel_id)
        if not existing:
            existing = self.get_by_name(name)

        if existing:
            existing.source_type = source_type
            existing.url = url
            existing.feed_url = feed_url or existing.feed_url
            existing.channel_id = channel_id or existing.channel_id
            existing.handle = handle or existing.handle
            existing.is_active = is_active
            existing.updated_at = utc_now()
            self.session.flush()
            return existing, False

        new_source = self.create(
            name=name,
            source_type=source_type,
            url=url,
            feed_url=feed_url,
            channel_id=channel_id,
            handle=handle,
            is_active=is_active,
        )
        return new_source, True


class ArticleRepository(BaseRepository):
    """CRUD operations and deduplication for articles and video transcripts."""

    def create(
        self,
        external_id: str,
        title: str,
        url: str,
        published_at: datetime,
        source_id: Optional[int] = None,
        author: Optional[str] = None,
        category: Optional[str] = None,
        summary: Optional[str] = None,
        raw_content: Optional[str] = None,
        content_type: str = "article",
        word_count: Optional[int] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        is_summarized: bool = False,
    ) -> Article:
        """Create and persist a new article/transcript."""
        article = Article(
            source_id=source_id,
            external_id=external_id,
            title=title,
            url=url,
            author=author,
            category=category,
            published_at=published_at,
            summary=summary,
            raw_content=raw_content,
            content_type=content_type,
            word_count=word_count,
            metadata_json=metadata_json,
            is_summarized=is_summarized,
        )
        self.session.add(article)
        self.session.flush()
        return article

    def get_by_id(self, article_id: int) -> Optional[Article]:
        """Find article by primary key ID."""
        return self.session.get(Article, article_id)

    def get_by_external_id(self, external_id: str) -> Optional[Article]:
        """Find article by unique external ID (e.g. YouTube video ID or article slug)."""
        stmt = select(Article).where(Article.external_id == external_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def exists(self, external_id: str) -> bool:
        """Check whether an item already exists by external_id (deduplication check)."""
        return self.get_by_external_id(external_id) is not None

    def upsert(
        self,
        external_id: str,
        title: str,
        url: str,
        published_at: datetime,
        source_id: Optional[int] = None,
        author: Optional[str] = None,
        category: Optional[str] = None,
        summary: Optional[str] = None,
        raw_content: Optional[str] = None,
        content_type: str = "article",
        word_count: Optional[int] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Article, bool]:
        """Upsert article based on external_id. Returns (article, is_created)."""
        existing = self.get_by_external_id(external_id)
        if existing:
            existing.title = title
            existing.url = url
            existing.published_at = published_at
            if source_id is not None:
                existing.source_id = source_id
            if author is not None:
                existing.author = author
            if category is not None:
                existing.category = category
            if summary is not None:
                existing.summary = summary
            if raw_content is not None:
                existing.raw_content = raw_content
                existing.word_count = word_count or len(raw_content.split())
            if metadata_json is not None:
                existing.metadata_json = metadata_json
            existing.updated_at = utc_now()
            self.session.flush()
            return existing, False

        new_art = self.create(
            source_id=source_id,
            external_id=external_id,
            title=title,
            url=url,
            author=author,
            category=category,
            published_at=published_at,
            summary=summary,
            raw_content=raw_content,
            content_type=content_type,
            word_count=word_count or (len(raw_content.split()) if raw_content else None),
            metadata_json=metadata_json,
        )
        return new_art, True

    def get_recent(
        self,
        hours: Optional[int] = None,
        source_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Article]:
        """Retrieve recent articles published within the lookback window."""
        stmt = select(Article)
        if hours and hours > 0:
            cutoff = utc_now() - timedelta(hours=hours)
            stmt = stmt.where(Article.published_at >= cutoff)
        if source_type:
            stmt = stmt.join(Article.source).where(Source.source_type == source_type)

        stmt = stmt.order_by(desc(Article.published_at)).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_all(self, limit: int = 100) -> List[Article]:
        """Retrieve all articles ordered by published_at desc."""
        stmt = select(Article).order_by(desc(Article.published_at)).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def export_llm_markdown(
        self,
        hours: Optional[int] = None,
        source_type: Optional[str] = None,
        unsummarized_only: bool = False,
        limit: int = 100,
        grouped: bool = False,
    ) -> str:
        """Format matching database articles into clean Markdown for LLMs."""
        from app.services.llm_formatter import LLMMarkdownFormatter

        if unsummarized_only:
            articles = self.get_unsummarized(hours=hours, limit=limit)
        else:
            articles = self.get_recent(hours=hours, source_type=source_type, limit=limit)

        formatter = LLMMarkdownFormatter()
        return formatter.format_source_grouped(articles) if grouped else formatter.format_articles(articles)

    def get_by_source_id(self, source_id: int, limit: int = 50) -> List[Article]:
        """Retrieve articles published by a specific source."""
        stmt = (
            select(Article)
            .where(Article.source_id == source_id)
            .order_by(desc(Article.published_at))
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_unsummarized(self, hours: Optional[int] = None, limit: int = 50) -> List[Article]:
        """Retrieve articles that have not yet been summarized by the LLM."""
        stmt = select(Article).where(Article.is_summarized.is_(False))
        if hours and hours > 0:
            cutoff = utc_now() - timedelta(hours=hours)
            stmt = stmt.where(Article.published_at >= cutoff)
        stmt = stmt.order_by(desc(Article.published_at)).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def mark_summarized(self, article_id: int, summary: Optional[str] = None) -> Optional[Article]:
        """Mark an article as summarized, optionally storing the LLM-generated summary."""
        article = self.get_by_id(article_id)
        if not article:
            return None
        article.is_summarized = True
        if summary:
            article.summary = summary
        article.updated_at = utc_now()
        self.session.flush()
        return article

    def delete(self, article_id: int) -> bool:
        """Delete an article by ID."""
        article = self.get_by_id(article_id)
        if not article:
            return False
        self.session.delete(article)
        self.session.flush()
        return True

    def count(self) -> int:
        """Count total articles in database."""
        from sqlalchemy import func

        stmt = select(func.count(Article.id))
        return self.session.execute(stmt).scalar() or 0

    def stats(self) -> Dict[str, Any]:
        """Return statistical breakdown of articles."""
        from sqlalchemy import func

        total = self.count()
        unsummarized = self.session.execute(
            select(func.count(Article.id)).where(Article.is_summarized.is_(False))
        ).scalar() or 0
        return {
            "total": total,
            "summarized": total - unsummarized,
            "unsummarized": unsummarized,
        }

    def remove_shorts(self) -> int:
        """Scan and delete all YouTube Shorts stored in the database.

        Checks:
        1. URL containing /shorts/
        2. Title containing #shorts or #short
        3. Summary containing #shorts or #short
        4. metadata_json containing is_short=True

        Returns:
            Number of removed short records.
        """
        stmt = select(Article).where(
            (Article.category == "YouTube") | (Article.content_type.in_(["video", "transcript"]))
        )
        candidates = list(self.session.execute(stmt).scalars().all())
        deleted_count = 0

        for art in candidates:
            is_short = False
            if art.url and "/shorts/" in art.url:
                is_short = True
            elif art.title and ("#shorts" in art.title.lower() or "#short" in art.title.lower()):
                is_short = True
            elif art.summary and ("#shorts" in art.summary.lower() or "#short" in art.summary.lower()):
                is_short = True
            elif art.metadata_json and art.metadata_json.get("is_short") is True:
                is_short = True

            if is_short:
                logger.info("Removing YouTube Short from database: '%s' (%s)", art.title, art.external_id)
                self.session.delete(art)
                deleted_count += 1

        if deleted_count > 0:
            self.session.flush()

        return deleted_count


class DigestRepository(BaseRepository):
    """CRUD operations for generated daily news digests."""

    def create(
        self,
        digest_date: date,
        title: str,
        content_markdown: str,
        content_html: Optional[str] = None,
        article_count: int = 0,
    ) -> Digest:
        """Create and store a synthesized daily digest."""
        digest = Digest(
            digest_date=digest_date,
            title=title,
            content_markdown=content_markdown,
            content_html=content_html,
            article_count=article_count,
        )
        self.session.add(digest)
        self.session.flush()
        return digest

    def get_by_id(self, digest_id: int) -> Optional[Digest]:
        """Get digest by ID."""
        return self.session.get(Digest, digest_id)

    def get_by_date(self, digest_date: date) -> Optional[Digest]:
        """Get digest for a specific calendar date."""
        stmt = select(Digest).where(Digest.digest_date == digest_date).order_by(desc(Digest.id))
        return self.session.execute(stmt).scalars().first()

    def get_latest(self) -> Optional[Digest]:
        """Get the most recently created digest."""
        stmt = select(Digest).order_by(desc(Digest.created_at)).limit(1)
        return self.session.execute(stmt).scalar_one_or_none()

    def mark_sent(self, digest_id: int, sent_at: Optional[datetime] = None) -> Optional[Digest]:
        """Mark digest as sent to subscribers."""
        digest = self.get_by_id(digest_id)
        if not digest:
            return None
        digest.sent_at = sent_at or utc_now()
        self.session.flush()
        return digest

    def list_recent(self, limit: int = 10) -> List[Digest]:
        """List most recent digests."""
        stmt = select(Digest).order_by(desc(Digest.created_at)).limit(limit)
        return list(self.session.execute(stmt).scalars().all())


class ArticleDigestRepository(BaseRepository):
    """CRUD operations for individual article/video summaries in table 'digest'."""

    def create(
        self,
        article_id: int,
        title: str,
        url: str,
        source_name: str,
        source_type: str,
        summary: str,
        problem_statement: Optional[str] = None,
        solution_statement: Optional[str] = None,
        key_insights: Optional[List[str]] = None,
        model_used: str = "openai:responses",
        category: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> ArticleDigest:
        """Create and store an article digest entry in table 'digest'."""
        digest_entry = ArticleDigest(
            article_id=article_id,
            title=title,
            url=url,
            source_name=source_name,
            source_type=source_type,
            summary=summary,
            problem_statement=problem_statement,
            solution_statement=solution_statement,
            key_insights=key_insights,
            model_used=model_used,
            category=category,
            tokens_used=tokens_used,
        )
        self.session.add(digest_entry)
        self.session.flush()
        return digest_entry

    def get_by_id(self, digest_id: int) -> Optional[ArticleDigest]:
        """Find digest entry by primary key ID."""
        return self.session.get(ArticleDigest, digest_id)

    def get_by_article_id(self, article_id: int) -> Optional[ArticleDigest]:
        """Find digest entry by foreign key article_id."""
        stmt = select(ArticleDigest).where(ArticleDigest.article_id == article_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def upsert(
        self,
        article_id: int,
        title: str,
        url: str,
        source_name: str,
        source_type: str,
        summary: str,
        problem_statement: Optional[str] = None,
        solution_statement: Optional[str] = None,
        key_insights: Optional[List[str]] = None,
        model_used: str = "openai:responses",
        category: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> Tuple[ArticleDigest, bool]:
        """Upsert a digest entry for an article. Returns (entry, created)."""
        existing = self.get_by_article_id(article_id)
        if existing:
            existing.title = title
            existing.url = url
            existing.source_name = source_name
            existing.source_type = source_type
            existing.summary = summary
            existing.problem_statement = problem_statement
            existing.solution_statement = solution_statement
            existing.key_insights = key_insights
            existing.model_used = model_used
            existing.category = category
            existing.tokens_used = tokens_used
            existing.updated_at = utc_now()
            self.session.flush()
            return existing, False

        new_entry = self.create(
            article_id=article_id,
            title=title,
            url=url,
            source_name=source_name,
            source_type=source_type,
            summary=summary,
            problem_statement=problem_statement,
            solution_statement=solution_statement,
            key_insights=key_insights,
            model_used=model_used,
            category=category,
            tokens_used=tokens_used,
        )
        return new_entry, True

    def get_all(
        self,
        limit: int = 100,
        source_type: Optional[str] = None,
    ) -> List[ArticleDigest]:
        """List all article digest entries, optionally filtered by source_type."""
        stmt = select(ArticleDigest).order_by(desc(ArticleDigest.created_at))
        if source_type:
            stmt = stmt.where(ArticleDigest.source_type == source_type)
        stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def delete_by_article_id(self, article_id: int) -> bool:
        """Delete an article digest entry by article_id."""
        existing = self.get_by_article_id(article_id)
        if not existing:
            return False
        self.session.delete(existing)
        self.session.flush()
        return True


class SubscriberRepository(BaseRepository):
    """CRUD operations for email subscribers and user interest profiles."""

    def create(
        self,
        email: str,
        name: Optional[str] = None,
        is_active: bool = True,
        bio_or_role: Optional[str] = None,
        expertise_level: str = "intermediate",
        interests: Optional[List[str]] = None,
        preferred_sources: Optional[List[str]] = None,
        ranking_instructions: Optional[str] = None,
    ) -> Subscriber:
        """Register a new email subscriber with profile interests."""
        subscriber = Subscriber(
            email=email.strip().lower(),
            name=name,
            is_active=is_active,
            bio_or_role=bio_or_role,
            expertise_level=expertise_level,
            interests=interests or [],
            preferred_sources=preferred_sources or [],
            ranking_instructions=ranking_instructions,
        )
        self.session.add(subscriber)
        self.session.flush()
        return subscriber

    def update_profile(
        self,
        email: str,
        name: Optional[str] = None,
        bio_or_role: Optional[str] = None,
        expertise_level: Optional[str] = None,
        interests: Optional[List[str]] = None,
        preferred_sources: Optional[List[str]] = None,
        ranking_instructions: Optional[str] = None,
    ) -> Optional[Subscriber]:
        """Update user profile preferences and ranking interests."""
        sub = self.get_by_email(email)
        if not sub:
            return None
        if name is not None:
            sub.name = name
        if bio_or_role is not None:
            sub.bio_or_role = bio_or_role
        if expertise_level is not None:
            sub.expertise_level = expertise_level
        if interests is not None:
            sub.interests = interests
        if preferred_sources is not None:
            sub.preferred_sources = preferred_sources
        if ranking_instructions is not None:
            sub.ranking_instructions = ranking_instructions
        sub.updated_at = utc_now()
        self.session.flush()
        return sub

    def get_by_id(self, subscriber_id: int) -> Optional[Subscriber]:
        """Get subscriber by ID."""
        return self.session.get(Subscriber, subscriber_id)

    def get_by_email(self, email: str) -> Optional[Subscriber]:
        """Find subscriber by email address."""
        stmt = select(Subscriber).where(Subscriber.email == email.strip().lower())
        return self.session.execute(stmt).scalar_one_or_none()

    def get_active(self) -> List[Subscriber]:
        """List all active subscribers."""
        stmt = select(Subscriber).where(Subscriber.is_active.is_(True)).order_by(Subscriber.email.asc())
        return list(self.session.execute(stmt).scalars().all())

    def get_all(self) -> List[Subscriber]:
        """List all subscribers."""
        stmt = select(Subscriber).order_by(Subscriber.email.asc())
        return list(self.session.execute(stmt).scalars().all())

    def toggle_active(self, email: str, is_active: bool) -> Optional[Subscriber]:
        """Enable or disable subscriber by email."""
        sub = self.get_by_email(email)
        if not sub:
            return None
        sub.is_active = is_active
        self.session.flush()
        return sub

    def delete(self, email: str) -> bool:
        """Remove a subscriber by email."""
        sub = self.get_by_email(email)
        if not sub:
            return False
        self.session.delete(sub)
        self.session.flush()
        return True


class UserDigestRankingRepository(BaseRepository):
    """CRUD operations for personalized subscriber rankings in table 'user_digest_rankings'."""

    def create(
        self,
        subscriber_id: int,
        digest_id: int,
        rank_position: int,
        rank_score: float,
        relevance_reasoning: str,
        key_takeaways_for_user: Optional[List[str]] = None,
        batch_date: Optional[date] = None,
        tier: str = "featured",
        quick_summary: Optional[str] = None,
    ) -> UserDigestRanking:
        """Create and persist a personalized ranking entry."""
        entry = UserDigestRanking(
            subscriber_id=subscriber_id,
            digest_id=digest_id,
            rank_position=rank_position,
            rank_score=rank_score,
            relevance_reasoning=relevance_reasoning,
            key_takeaways_for_user=key_takeaways_for_user or [],
            batch_date=batch_date or utc_now().date(),
            tier=tier,
            quick_summary=quick_summary,
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def get_rankings_for_subscriber(
        self,
        subscriber_id: int,
        batch_date: Optional[date] = None,
        limit: int = 50,
    ) -> List[UserDigestRanking]:
        """Fetch ranked summaries for a subscriber ordered by rank_position asc."""
        stmt = (
            select(UserDigestRanking)
            .where(UserDigestRanking.subscriber_id == subscriber_id)
            .order_by(UserDigestRanking.rank_position.asc())
        )
        if batch_date:
            stmt = stmt.where(UserDigestRanking.batch_date == batch_date)
        stmt = stmt.limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_latest_batch_date(self, subscriber_id: int) -> Optional[date]:
        """Return the most recent batch_date available for a subscriber."""
        stmt = (
            select(UserDigestRanking.batch_date)
            .where(UserDigestRanking.subscriber_id == subscriber_id)
            .order_by(UserDigestRanking.batch_date.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def clear_subscriber_batch(self, subscriber_id: int, batch_date: date) -> int:
        """Clear existing rankings for a subscriber on a specific date before re-ranking."""
        stmt = select(UserDigestRanking).where(
            UserDigestRanking.subscriber_id == subscriber_id,
            UserDigestRanking.batch_date == batch_date,
        )
        records = list(self.session.execute(stmt).scalars().all())
        count = len(records)
        for r in records:
            self.session.delete(r)
        if count > 0:
            self.session.flush()
        return count


class NewsRepository:
    """Master Unit-of-Work aggregator coordinating all repository operations.

    Usage:
        # Context manager (auto-commits on clean exit):
        with NewsRepository() as repo:
            repo.sources.create(...)
            repo.articles.upsert(...)

        # Or with an explicit injected session:
        repo = NewsRepository(session=my_session)
    """

    def __init__(self, session: Optional[Session] = None) -> None:
        self._owns_session = session is None
        self.session = session or SessionLocal()
        self.sources = SourceRepository(self.session)
        self.articles = ArticleRepository(self.session)
        self.digests = DigestRepository(self.session)
        self.article_digests = ArticleDigestRepository(self.session)
        self.subscribers = SubscriberRepository(self.session)
        self.user_rankings = UserDigestRankingRepository(self.session)

    def __enter__(self) -> NewsRepository:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._owns_session:
            if exc_type is not None:
                self.session.rollback()
                logger.error("NewsRepository transaction rolled back: %s", exc_val)
            else:
                self.session.commit()
            self.session.close()

    def commit(self) -> None:
        """Explicitly commit pending session transactions."""
        self.session.commit()

    def rollback(self) -> None:
        """Explicitly rollback pending session transactions."""
        self.session.rollback()

    def get_digest_candidates(self, hours: int = 24, limit: int = 50) -> List[Article]:
        """Retrieve unsummarized articles in lookback window ready for LLM synthesis."""
        return self.articles.get_unsummarized(hours=hours, limit=limit)

    def stats(self) -> Dict[str, Any]:
        """Retrieve high-level system counts across sources, articles, digests, and subscribers."""
        from sqlalchemy import func

        sources_total = self.session.execute(select(func.count(Source.id))).scalar() or 0
        sources_active = self.session.execute(
            select(func.count(Source.id)).where(Source.is_active.is_(True))
        ).scalar() or 0
        subscribers_active = self.session.execute(
            select(func.count(Subscriber.id)).where(Subscriber.is_active.is_(True))
        ).scalar() or 0
        digests_total = self.session.execute(select(func.count(Digest.id))).scalar() or 0

        return {
            "sources": {"total": sources_total, "active": sources_active},
            "articles": self.articles.stats(),
            "subscribers": subscribers_active,
            "digests": digests_total,
        }

    def remove_shorts(self) -> int:
        """Remove any YouTube shorts stored in the database."""
        deleted = self.articles.remove_shorts()
        if deleted > 0:
            self.session.commit()
        return deleted

    # High-level bridge methods for Pydantic models
    def save_youtube_video(
        self,
        video: Any,
        source_id: Optional[int] = None,
    ) -> Tuple[Optional[Article], bool]:
        """Save or update a YouTube video Pydantic model into the database.

        Automatically skips YouTube Shorts.
        Deduplicates by YouTube video_id.
        """
        # Exclude Shorts from database persistence
        if (
            getattr(video, "is_short", False)
            or "/shorts/" in str(getattr(video, "url", ""))
            or (getattr(video, "title", None) and "#shorts" in str(video.title).lower())
            or (getattr(video, "description", None) and "#shorts" in str(video.description).lower())
        ):
            logger.info("Skipping YouTube Short from database persistence: %s (%s)", getattr(video, "title", ""), getattr(video, "video_id", ""))
            return None, False

        # Auto-resolve or find source if source_id not provided
        if source_id is None and getattr(video, "channel_id", None):
            src = self.sources.get_by_channel_id(video.channel_id)
            if src:
                source_id = src.id

        # Safely extract transcript text and metadata
        raw_text: Optional[str] = None
        has_transcript = False
        transcript_lang: Optional[str] = None

        if hasattr(video, "transcript") and video.transcript:
            if isinstance(video.transcript, str):
                raw_text = video.transcript
                has_transcript = bool(video.transcript.strip())
            else:
                raw_text = getattr(video.transcript, "text", None)
                has_transcript = bool(getattr(video.transcript, "has_content", False))
                transcript_lang = getattr(video.transcript, "language", None)

        return self.articles.upsert(
            external_id=video.video_id,
            title=video.title,
            url=str(video.url),
            published_at=video.published_at,
            source_id=source_id,
            author=video.channel_name,
            category="YouTube",
            summary=video.description,
            raw_content=raw_text,
            content_type="transcript" if has_transcript else "video",
            word_count=len(raw_text.split()) if raw_text else None,
            metadata_json={
                "channel_id": getattr(video, "channel_id", None),
                "has_transcript": has_transcript,
                "transcript_language": transcript_lang,
                "thumbnail_url": getattr(video, "thumbnail_url", None),
            },
        )

    def save_openai_article(
        self,
        article: Any,
        source_id: Optional[int] = None,
    ) -> Tuple[Article, bool]:
        """Save or update an OpenAI article Pydantic model into the database."""
        if source_id is None:
            src = self.sources.get_by_name("OpenAI")
            if src:
                source_id = src.id

        cats = getattr(article, "categories", [])
        cat_str = ", ".join(cats) if isinstance(cats, list) and cats else getattr(article, "category", None)

        return self.articles.upsert(
            external_id=article.article_id,
            title=article.title,
            url=str(article.url),
            published_at=article.published_at,
            source_id=source_id,
            author=getattr(article, "author", "OpenAI"),
            category=cat_str,
            summary=article.summary,
            raw_content=article.content,
            content_type="article",
            word_count=article.word_count,
            metadata_json={"categories": cats},
        )

    def save_anthropic_article(
        self,
        article: Any,
        source_id: Optional[int] = None,
    ) -> Tuple[Article, bool]:
        """Save or update an Anthropic article Pydantic model into the database."""
        if source_id is None:
            src = self.sources.get_by_name("Anthropic")
            if src:
                source_id = src.id

        cats = getattr(article, "categories", [])
        cat_str = (
            getattr(article, "feed_source", None)
            or (", ".join(cats) if isinstance(cats, list) and cats else getattr(article, "category", None))
        )

        return self.articles.upsert(
            external_id=article.article_id,
            title=article.title,
            url=str(article.url),
            published_at=article.published_at,
            source_id=source_id,
            author=getattr(article, "author", "Anthropic"),
            category=cat_str,
            summary=article.summary,
            raw_content=article.content,
            content_type="article",
            word_count=article.word_count,
            metadata_json={
                "categories": cats,
                "feed_source": getattr(article, "feed_source", None),
            },
        )

    def export_llm_markdown(
        self,
        hours: Optional[int] = None,
        source_type: Optional[str] = None,
        unsummarized_only: bool = False,
        limit: int = 100,
        grouped: bool = False,
    ) -> str:
        """Format matching database articles into clean Markdown for LLMs."""
        return self.articles.export_llm_markdown(
            hours=hours,
            source_type=source_type,
            unsummarized_only=unsummarized_only,
            limit=limit,
            grouped=grouped,
        )

