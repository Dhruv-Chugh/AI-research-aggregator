"""Database Table Initialization and Seeding Services.

Handles creating SQLAlchemy schema tables and seeding initial sources/subscribers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from sqlalchemy import Engine

from app.database.models import Base
from app.database.session import check_connection, engine as default_engine, get_session_factory

logger = logging.getLogger("ai_news.database.init_db")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def seed_initial_data(engine: Engine) -> None:
    """Seed default sources from config/sources.json and default subscriber."""
    from app.config import get_sources_config
    from app.database.repository import NewsRepository

    session_maker = get_session_factory(engine)
    with session_maker() as session:
        repo = NewsRepository(session=session)

        # 1. Load config/sources.json
        try:
            cfg = get_sources_config()
            logger.info("Seeding sources from config/sources.json...")

            # YouTube channels
            for ch in cfg.youtube_channels:
                src, created = repo.sources.upsert(
                    name=ch.name,
                    source_type="youtube",
                    url=str(ch.url),
                    feed_url=ch.feed_url,
                    channel_id=ch.channel_id,
                    handle=ch.handle,
                    is_active=ch.is_active,
                )
                action = "Added" if created else "Updated"
                logger.info("%s YouTube source: %s (%s)", action, src.name, src.channel_id)

            # OpenAI
            if cfg.openai:
                src, created = repo.sources.upsert(
                    name="OpenAI",
                    source_type="openai",
                    url="https://openai.com/news/",
                    feed_url=str(cfg.openai.rss_url),
                    is_active=cfg.openai.is_active,
                )
                action = "Added" if created else "Updated"
                logger.info("%s OpenAI source (%s)", action, src.feed_url)

            # Anthropic
            if cfg.anthropic:
                src, created = repo.sources.upsert(
                    name="Anthropic",
                    source_type="anthropic",
                    url="https://www.anthropic.com/news",
                    feed_url=cfg.anthropic.feeds.get("news"),
                    is_active=cfg.anthropic.is_active,
                )
                action = "Added" if created else "Updated"
                logger.info("%s Anthropic source", action)

        except Exception as e:
            logger.warning("Could not seed sources from config: %s", e)

        # 2. Seed default subscriber
        import os
        default_email = os.getenv("RECIPIENT_EMAIL", "").strip().lower()
        sub_name = os.getenv("SUBSCRIBER_NAME", "Research Practitioner").strip()
        if default_email:
            sub = repo.subscribers.get_by_email(default_email)
            if not sub:
                repo.subscribers.create(
                    email=default_email,
                    name=sub_name,
                    is_active=True,
                )
                logger.info("Added default subscriber: %s", default_email)
            else:
                sub.name = sub_name
                sub.is_active = True
                session.flush()
                logger.info("Verified active subscriber: %s", default_email)

        session.commit()
        logger.info("✅ Database seeding completed successfully.")


def create_tables(
    target_engine: Optional[Engine] = None,
    drop_first: bool = False,
    seed_defaults: bool = False,
) -> None:
    """Create all defined tables in the database."""
    eng = target_engine or default_engine

    db_display = str(eng.url).split("@")[-1] if "@" in str(eng.url) else str(eng.url)
    logger.info("Connecting to database: %s", db_display)

    if drop_first:
        logger.warning("Dropping all existing database tables...")
        Base.metadata.drop_all(bind=eng)
        logger.info("Dropped existing tables.")

    logger.info("Creating tables: %s", ", ".join(Base.metadata.tables.keys()))
    Base.metadata.create_all(bind=eng)

    # Ensure any newly added columns exist in existing tables (e.g. subscribers)
    try:
        from sqlalchemy import text

        with eng.connect() as conn:
            conn.execute(text("ALTER TABLE subscribers ADD COLUMN IF NOT EXISTS bio_or_role VARCHAR(255);"))
            conn.execute(text("ALTER TABLE subscribers ADD COLUMN IF NOT EXISTS expertise_level VARCHAR(50) DEFAULT 'intermediate';"))
            conn.execute(text("ALTER TABLE subscribers ADD COLUMN IF NOT EXISTS interests JSON DEFAULT '[]';"))
            conn.execute(text("ALTER TABLE subscribers ADD COLUMN IF NOT EXISTS preferred_sources JSON DEFAULT '[]';"))
            conn.execute(text("ALTER TABLE subscribers ADD COLUMN IF NOT EXISTS ranking_instructions TEXT;"))
            conn.execute(text("ALTER TABLE subscribers ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();"))
            conn.execute(text("ALTER TABLE user_digest_rankings ADD COLUMN IF NOT EXISTS tier VARCHAR(50) DEFAULT 'featured';"))
            conn.execute(text("ALTER TABLE user_digest_rankings ADD COLUMN IF NOT EXISTS quick_summary TEXT;"))
            conn.commit()
    except Exception as e:
        logger.debug("Column migration note: %s", e)

    logger.info("✅ Successfully created all database tables!")

    if seed_defaults:
        seed_initial_data(eng)


# Clean alias
init_db = create_tables


def main() -> None:
    """CLI entry point for database table initialization."""
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Initialize database tables for AI News Aggregator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all existing tables before recreating them (destructive)",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed initial sources and default subscriber after table creation",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check database connection reachability without creating tables",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🛠️  DATABASE INITIALIZATION")
    print("=" * 60)

    is_online = check_connection(default_engine)
    if not is_online:
        print(f"⚠️  Database at '{default_engine.url.render_as_string(hide_password=True)}' is not reachable.")
        print("💡 Hint: If using Docker, ensure PostgreSQL container is running:")
        print("   docker compose up -d\n")
        if args.check_only:
            sys.exit(1)
    else:
        print(f"🟢 Connected to database: {default_engine.url.render_as_string(hide_password=True)}")

    if args.check_only:
        print("✅ Database connection check passed.\n")
        return

    try:
        create_tables(
            target_engine=default_engine,
            drop_first=args.drop,
            seed_defaults=args.seed,
        )
        print("\n🎉 Table initialization completed successfully!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize tables: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

