"""Manual briefing dispatch script using the configured PostgreSQL database.

Reads digests and personalized rankings from PostgreSQL (or optionally runs a
fresh scrape/summarize/rank cycle with --full) and delivers the Visual IEEE PDF
and responsive HTML briefing live via Gmail SMTP.

Usage:
    uv run python scripts/manual_send.py                # Send latest Postgres briefing via SMTP
    uv run python scripts/manual_send.py --full         # Scrape, summarize, rank & send via SMTP
    uv run python scripts/manual_send.py --dry-run      # Generate HTML/PDF locally without SMTP
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("manual_send")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually generate and send AI intelligence briefing from PostgreSQL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--email",
        type=str,
        default=os.getenv("RECIPIENT_EMAIL", "dhruvchugh2007@gmail.com"),
        help="Target subscriber email address.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Send to all active subscribers in PostgreSQL instead of filtering to --email.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full live scraping + LLM summarization + ranking before sending.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours when running with --full.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export PDF and HTML locally without sending via SMTP.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="exports/briefings",
        help="Directory to write generated PDF and HTML previews.",
    )
    return parser.parse_args()


def _ensure_postgres_ready(recipient_email: str) -> bool:
    """Verify PostgreSQL connectivity and ensure target subscriber exists."""
    from app.database.init_db import create_tables, seed_initial_data
    from app.database.models import Article, ArticleDigest, UserDigestRanking
    from app.database.repository import NewsRepository
    from app.database.session import SessionLocal, check_connection, engine

    if not check_connection(engine):
        logger.error(
            "Cannot connect to database (%s). Ensure PostgreSQL is running (e.g. 'docker compose up -d').",
            engine.url.render_as_string(hide_password=True),
        )
        return False

    logger.info("Connected to database: %s", engine.url.render_as_string(hide_password=True))
    create_tables(engine)
    seed_initial_data(engine)

    with SessionLocal() as session:
        repo = NewsRepository(session)
        sub = repo.subscribers.get_by_email(recipient_email)
        if not sub:
            logger.info("Seeding missing subscriber '%s' in PostgreSQL...", recipient_email)
            sub = repo.subscribers.create(
                email=recipient_email,
                name=os.getenv("SUBSCRIBER_NAME", "Dhruv Chugh"),
                bio_or_role="AI Software Architect & Researcher",
                expertise_level="advanced",
                interests=[
                    "Autonomous Agents",
                    "Large Language Models & Reasoning",
                    "Frontier AI Research & Architecture",
                ],
                preferred_sources=["openai", "anthropic", "youtube"],
            )
            session.commit()

        articles_cnt = session.query(Article).count()
        digests_cnt = session.query(ArticleDigest).count()
        rankings_cnt = (
            session.query(UserDigestRanking)
            .filter(UserDigestRanking.subscriber_id == sub.id)
            .count()
        )
        logger.info(
            "PostgreSQL state -> Articles: %d | Digests: %d | Rankings for %s: %d",
            articles_cnt,
            digests_cnt,
            sub.email,
            rankings_cnt,
        )
    return True


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    send_live = not args.dry_run
    target_email = args.email.strip().lower()
    email_filter = None if args.all else target_email

    if not _ensure_postgres_ready(target_email):
        return 1

    if args.full:
        from app.scrapers.runner import AggregationRunner

        logger.info("Running full AggregationRunner pipeline (hours=%d, send_live=%s)...", args.hours, send_live)
        runner = AggregationRunner()
        res = runner.run(
            hours_back=args.hours,
            include_youtube=True,
            include_openai=True,
            include_anthropic=True,
            fetch_transcripts=True,
            fetch_full_content=True,
            sync_to_db=True,
            summarize=True,
            rank=True,
            email_briefing=False,  # Handle targeted dispatch cleanly below
            provider="gemini",
        )
        logger.info(
            "Full scrape/summarize/rank done -> Articles synced: %d | Digests: %d | Rankings: %d",
            res.db_items_saved,
            res.summaries_created,
            res.rankings_created,
        )
    else:
        # Ensure any un-summarized digests / missing rankings in Postgres are filled before sending
        from app.database.models import ArticleDigest, UserDigestRanking
        from app.database.repository import NewsRepository
        from app.database.session import SessionLocal

        with SessionLocal() as session:
            repo = NewsRepository(session)
            sub = repo.subscribers.get_by_email(target_email)
            digest_count = session.query(ArticleDigest).count()
            sub_rankings = (
                session.query(UserDigestRanking)
                .filter(UserDigestRanking.subscriber_id == sub.id)
                .count()
                if sub
                else 0
            )

        if digest_count == 0:
            from app.agent.summarizer import DigestAgent

            logger.info("No digests found in PostgreSQL. Summarizing existing articles...")
            with DigestAgent(preferred_provider="gemini") as digest_agent:
                rep = digest_agent.run(force=False)
                logger.info("Summarized %d articles into table 'digest'.", rep.successful)

        if sub_rankings == 0:
            from app.agent.ranking_agent import UserRankingAgent

            logger.info("No rankings found for %s. Generating personalized rankings...", target_email)
            with UserRankingAgent(preferred_provider="gemini") as ranking_agent:
                rank_rep = ranking_agent.run_daily_ranking(email_filter=email_filter)
                logger.info("Generated %d personalized rankings.", rank_rep.rankings_stored)

    # Dispatch email from PostgreSQL rankings
    from app.agent.email_agent import EmailDeliveryAgent

    with EmailDeliveryAgent() as email_agent:
        report = email_agent.run_daily_delivery(
            email_filter=email_filter,
            send_live=send_live,
            export_dir=args.export_dir,
        )

    print("\n" + "=" * 64)
    print("MANUAL POSTGRESQL BRIEFING DISPATCH REPORT")
    print("=" * 64)
    print(f"Batch Date:            {report.batch_date}")
    print(f"Subscribers Targeted:  {report.subscribers_targeted}")
    print(f"Briefings Generated:   {report.briefings_generated}")
    print(f"Live Emails Sent:      {report.emails_sent_live}")
    print(f"Local Exports Only:    {report.exports_saved}")
    for r in report.results:
        status = "SENT LIVE" if r["sent_live"] else ("FAILED" if not r["success"] else "EXPORTED")
        print(f"  [{status}] {r['email']} | PDF: {r.get('pdf_path')} ({r.get('pdf_size', 0):,} bytes)")
    if report.errors:
        print("Errors:")
        for err in report.errors:
            print(f"  - {err}")
    print("=" * 64)

    if send_live:
        return 0 if report.emails_sent_live > 0 and not report.errors else 1
    return 0 if report.briefings_generated > 0 and not report.errors else 1


if __name__ == "__main__":
    sys.exit(main())
