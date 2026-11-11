"""Manual briefing dispatch script using the configured PostgreSQL database.

Reads digests and personalized rankings from PostgreSQL (or optionally runs a
fresh scrape/summarize/rank cycle with --full) and delivers the Visual IEEE PDF
and responsive HTML briefing live via Gmail SMTP (587 -> 465) with automatic
Resend HTTPS (443) fallback when SMTP ports are blocked.

Usage:
    uv run python scripts/manual_send.py                # Send latest Postgres briefing live
    uv run python scripts/manual_send.py --full         # Scrape, summarize, rank & send live
    uv run python scripts/manual_send.py --preflight    # Verify DB, API keys & email ports
    uv run python scripts/manual_send.py --dry-run      # Generate HTML/PDF locally only
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict

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
        default=os.getenv("RECIPIENT_EMAIL", "subscriber@example.com"),
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
        help="Export PDF and HTML locally without sending live email.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check PostgreSQL, LLM keys, and outbound email ports without scraping or sending.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="exports/briefings",
        help="Directory to write generated PDF and HTML previews.",
    )
    return parser.parse_args()


def _probe_tcp(host: str, port: int, timeout: float = 2.5) -> bool:
    """Return True if outbound TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run_preflight() -> Dict[str, Any]:
    """Inspect local PostgreSQL, LLM API keys, and email transport ports."""
    from app.database.session import check_connection, engine
    from app.services.email_service import EmailService

    db_ok = check_connection(engine)
    email_svc = EmailService()
    smtp_587_ok = _probe_tcp(email_svc.smtp_host, 587) if email_svc.has_smtp_credentials else False
    smtp_465_ok = _probe_tcp(email_svc.smtp_host, 465) if email_svc.has_smtp_credentials else False
    gemini_ok = bool(os.getenv("GEMINI_API_KEY", "").strip())
    openai_ok = bool(os.getenv("OPENAI_API_KEY", "").strip())

    return {
        "db_ok": db_ok,
        "db_url": engine.url.render_as_string(hide_password=True),
        "gemini_ok": gemini_ok,
        "openai_ok": openai_ok,
        "smtp_configured": email_svc.has_smtp_credentials,
        "smtp_host": email_svc.smtp_host,
        "smtp_587_reachable": smtp_587_ok,
        "smtp_465_reachable": smtp_465_ok,
        "resend_configured": email_svc.has_resend,
        "email_ready": smtp_587_ok or smtp_465_ok or email_svc.has_resend or email_svc.has_smtp_credentials,
    }


def _ensure_postgres_ready(recipient_email: str) -> bool:
    """Verify PostgreSQL connectivity and ensure target subscriber exists."""
    from app.database.init_db import create_tables, seed_initial_data
    from app.database.models import Article, ArticleDigest, UserDigestRanking
    from app.database.repository import NewsRepository
    from app.database.session import SessionLocal, check_connection, engine

    if not check_connection(engine):
        logger.error(
            "Cannot connect to database (%s). Start PostgreSQL with 'docker start ai_news_postgres' or 'docker compose up -d postgres'.",
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
                name=os.getenv("SUBSCRIBER_NAME", "AI Research Subscriber"),
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

    if args.preflight:
        checks = _run_preflight()
        print("\n" + "=" * 64)
        print("LOCAL PIPELINE PREFLIGHT CHECK")
        print("=" * 64)
        print(f"PostgreSQL:   {'OK' if checks['db_ok'] else 'UNREACHABLE'} ({checks['db_url']})")
        print(f"LLM Keys:     Gemini={'OK' if checks['gemini_ok'] else 'MISSING'} | OpenAI={'OK' if checks['openai_ok'] else 'MISSING'}")
        print(
            f"Gmail SMTP:   Creds={'OK' if checks['smtp_configured'] else 'MISSING'} | "
            f"Port 587={'OPEN' if checks['smtp_587_reachable'] else 'BLOCKED/UNCHECKED'} | "
            f"Port 465={'OPEN' if checks['smtp_465_reachable'] else 'BLOCKED/UNCHECKED'}"
        )
        print(f"Resend 443:   {'READY (HTTPS fallback available)' if checks['resend_configured'] else 'NOT SET'}")
        print("=" * 64)
        return 0 if checks["db_ok"] and (checks["gemini_ok"] or checks["openai_ok"]) and checks["email_ready"] else 1

    if not _ensure_postgres_ready(target_email):
        return 1

    if args.full:
        from app.scrapers.runner import AggregationRunner

        logger.info("Running full AggregationRunner pipeline (hours=%d, send_live=%s)...", args.hours, send_live)
        with AggregationRunner() as runner:
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
                expand_if_empty=(args.hours <= 48),
            )
        logger.info(
            "Full scrape/summarize/rank done (window=%sh) -> Scraped: %d | Articles synced: %d | Digests: %d | Rankings: %d",
            res.hours_back,
            res.total_items,
            res.db_items_saved,
            res.summaries_created,
            res.rankings_created,
        )
        if res.warnings:
            for warn in res.warnings:
                logger.warning("Pipeline warning: %s", warn)
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
