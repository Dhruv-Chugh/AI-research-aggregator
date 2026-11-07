"""SQLAlchemy 2.0 ORM Relational Schema for Axiom Dispatch.

Relational Architecture (Normalized & High-Performance):
┌──────────┐ 1      N ┌──────────┐ 1      1 ┌──────────────┐ 1      N ┌──────────────────────┐
│  Source  │─────────<│ Article  │──────────│ ArticleDigest│─────────<│  UserDigestRanking   │
└──────────┘          └──────────┘          └──────────────┘          └──────────────────────┘
                                                                                 ^
                                                                                 │ N
                                                                      ┌──────────────────────┐
                                                                      │      Subscriber      │
                                                                      └──────────────────────┘

Core Tables:
1. `sources`: Monitored technical intelligence channels (YouTube, OpenAI, Anthropic).
2. `articles`: Ingested raw papers, articles, and spoken video transcripts.
3. `digest`: Synthesized LLM analytical briefings (problem, solution, takeaways).
4. `subscribers`: User profiles (role, technical interests, custom directives).
5. `user_digest_rankings`: Calibrated daily personalized relevance rankings (tiers, scores).
6. `digests`: (Legacy / Archive) Historical compiled briefings.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarative class for all database models."""
    pass


class Source(Base):
    """Tracks content sources (YouTube channels, company RSS feeds, blogs)."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'youtube', 'openai', 'anthropic', etc.
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    feed_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    channel_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    handle: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    last_scraped_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    articles: Mapped[List["Article"]] = relationship(
        "Article", back_populates="source", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize source model to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "url": self.url,
            "feed_url": self.feed_url,
            "channel_id": self.channel_id,
            "handle": self.handle,
            "is_active": self.is_active,
            "last_scraped_at": self.last_scraped_at.isoformat() if self.last_scraped_at else None,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"<Source(id={self.id}, name='{self.name}', type='{self.source_type}', active={self.is_active})>"


class Article(Base):
    """Stores ingested articles, blog posts, and YouTube video transcripts."""

    __tablename__ = "articles"
    __table_args__ = (
        Index("ix_articles_pub_summarized", "published_at", "is_summarized"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=True, index=True
    )
    external_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )  # YouTube video ID, OpenAI article slug, or Anthropic URL hash
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Full web content or transcript
    content_type: Mapped[str] = mapped_column(
        String(50), default="article", nullable=False
    )  # 'article', 'transcript', 'video'
    word_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    is_summarized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    source: Mapped[Optional["Source"]] = relationship("Source", back_populates="articles")
    digest_entry: Mapped[Optional["ArticleDigest"]] = relationship(
        "ArticleDigest", back_populates="article", uselist=False, cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize article model to dictionary representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "external_id": self.external_id,
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "category": self.category,
            "published_at": self.published_at.isoformat(),
            "summary": self.summary,
            "content_type": self.content_type,
            "word_count": self.word_count,
            "metadata": self.metadata_json or {},
            "is_summarized": self.is_summarized,
            "created_at": self.created_at.isoformat(),
        }

    def to_llm_markdown(self, index: Optional[int] = None) -> str:
        """Format article metadata and full content into simple, clean Markdown for LLMs."""
        source_name = self.source.name if self.source else (self.category or "Unknown Source")
        source_type = self.source.source_type if self.source else "article"
        prefix = f"{index}. " if index is not None else ""
        pub_str = (
            self.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if self.published_at
            else "Unknown Date"
        )
        type_label = "Spoken Transcript" if self.content_type == "transcript" else "Full Web Article"
        word_count_str = f"{self.word_count:,} words" if self.word_count else "Not counted"

        lines = [
            f"## {prefix}[{source_name}] {self.title}",
            f"- **Source**: {source_name} (`{source_type}`)",
            f"- **Original URL**: {self.url}",
            f"- **Published**: {pub_str}",
            f"- **Author / Channel**: {self.author or 'Unknown'}",
            f"- **Category / Focus**: {self.category or 'General'}",
            f"- **Content Type**: {type_label}",
            f"- **Word Count**: {word_count_str}",
            f"- **External ID**: `{self.external_id}`",
        ]

        meta = self.metadata_json or {}
        if meta.get("transcript_language"):
            lines.append(f"- **Language**: {meta['transcript_language']}")
        if meta.get("categories") and isinstance(meta["categories"], list):
            cats_joined = ", ".join(f"`{c}`" for c in meta["categories"] if c)
            if cats_joined:
                lines.append(f"- **Tags**: {cats_joined}")

        lines.append("")

        if self.raw_content and self.raw_content.strip():
            header = "### Spoken Transcript:" if self.content_type == "transcript" else "### Full Content:"
            lines.append(header)
            lines.append(self.raw_content.strip())
        elif self.summary and self.summary.strip():
            lines.append("### Summary (Full Content Unavailable):")
            lines.append(f"> {self.summary.strip()}")
        else:
            lines.append("### Content:")
            lines.append("_No content or summary available._")

        return "\n".join(lines)

    def __repr__(self) -> str:
        safe_title = self.title[:30] if self.title else "Untitled"
        return f"<Article(id={self.id}, external_id='{self.external_id}', title='{safe_title}...')>"


class ArticleDigest(Base):
    """Stores individual article & video summaries and insights generated by the LLM agent.
    
    Mapped to table 'digest' per user specification.
    """

    __tablename__ = "digest"
    __table_args__ = (
        Index("ix_digest_article_created", "article_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)  # Original source URL
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'youtube', 'openai', 'anthropic'
    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)  # Half-page knowledge-rich summary
    problem_statement: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    solution_statement: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_insights: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    article: Mapped["Article"] = relationship("Article", back_populates="digest_entry")
    user_rankings: Mapped[List["UserDigestRanking"]] = relationship(
        "UserDigestRanking", back_populates="digest", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize digest entry to dictionary representation."""
        return {
            "id": self.id,
            "article_id": self.article_id,
            "title": self.title,
            "url": self.url,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "category": self.category,
            "summary": self.summary,
            "problem_statement": self.problem_statement,
            "solution_statement": self.solution_statement,
            "key_insights": self.key_insights or [],
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Format digest entry as clean Markdown."""
        lines = [
            f"# {self.title}",
            f"- **Source**: {self.source_name} (`{self.source_type}`)",
            f"- **Original Source URL**: [{self.url}]({self.url})",
            f"- **Model Used**: `{self.model_used}`",
        ]
        if self.category:
            lines.append(f"- **Category**: {self.category}")
        if self.tokens_used:
            lines.append(f"- **Tokens Used**: {self.tokens_used:,}")

        lines.extend([
            "",
            "## Summary",
            self.summary.strip(),
            "",
        ])

        if self.problem_statement:
            lines.extend([
                "### Problem Highlighted",
                self.problem_statement.strip(),
                "",
            ])

        if self.solution_statement:
            lines.extend([
                "### Solution & Breakthrough",
                self.solution_statement.strip(),
                "",
            ])

        if self.key_insights:
            lines.append("### Key Insights & Actionable Takeaways")
            for insight in self.key_insights:
                lines.append(f"- {insight}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        safe_title = self.title[:30] if self.title else "Untitled"
        return f"<ArticleDigest(id={self.id}, article_id={self.article_id}, title='{safe_title}...', model='{self.model_used}')>"


class Digest(Base):
    """Synthesized daily news digest sent to subscribers."""

    __tablename__ = "digests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    digest_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    content_html: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    article_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize digest model to dictionary representation."""
        return {
            "id": self.id,
            "digest_date": self.digest_date.isoformat(),
            "title": self.title,
            "article_count": self.article_count,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"<Digest(id={self.id}, date={self.digest_date}, title='{self.title}', count={self.article_count})>"


class Subscriber(Base):
    """Email recipient and personalized profile for AI news digests."""

    __tablename__ = "subscribers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    bio_or_role: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    expertise_level: Mapped[str] = mapped_column(String(50), default="intermediate", nullable=False)
    interests: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    preferred_sources: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    ranking_instructions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    rankings: Mapped[List["UserDigestRanking"]] = relationship(
        "UserDigestRanking", back_populates="subscriber", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize subscriber model to dictionary representation."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "is_active": self.is_active,
            "bio_or_role": self.bio_or_role,
            "expertise_level": self.expertise_level,
            "interests": self.interests or [],
            "preferred_sources": self.preferred_sources or [],
            "ranking_instructions": self.ranking_instructions,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"<Subscriber(id={self.id}, email='{self.email}', active={self.is_active})>"


class UserDigestRanking(Base):
    """Stores personalized ranking and relevance evaluations for a subscriber."""

    __tablename__ = "user_digest_rankings"
    __table_args__ = (
        Index("ix_user_digest_subscriber_date", "subscriber_id", "batch_date"),
        Index("ix_user_digest_subscriber_tier_date", "subscriber_id", "tier", "batch_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subscriber_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("subscribers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    digest_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("digest.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 = top recommendation
    rank_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0.0 to 100.0 score
    relevance_reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    key_takeaways_for_user: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    tier: Mapped[str] = mapped_column(String(50), default="featured", nullable=False)  # 'featured' or 'quick_bite'
    quick_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 3-4 line concise summary
    batch_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    subscriber: Mapped["Subscriber"] = relationship("Subscriber", back_populates="rankings")
    digest: Mapped["ArticleDigest"] = relationship("ArticleDigest", back_populates="user_rankings")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ranking entry to dictionary representation."""
        return {
            "id": self.id,
            "subscriber_id": self.subscriber_id,
            "digest_id": self.digest_id,
            "rank_position": self.rank_position,
            "rank_score": self.rank_score,
            "relevance_reasoning": self.relevance_reasoning,
            "key_takeaways_for_user": self.key_takeaways_for_user or [],
            "tier": self.tier,
            "quick_summary": self.quick_summary,
            "batch_date": self.batch_date.isoformat(),
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"<UserDigestRanking(subscriber_id={self.subscriber_id}, "
            f"digest_id={self.digest_id}, rank={self.rank_position}, score={self.rank_score})>"
        )


# Aliases for convenience
DigestEntry = ArticleDigest
DailyDigest = Digest


