"""Email Delivery Agent for Automated Daily AI Intelligence Briefings.

Orchestrates the synthesis of visual PDF intelligence briefings and responsive HTML
emails for subscribers based on personalized interest rankings in table 'user_digest_rankings'.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Ensure environment variables from .env are loaded
load_dotenv()

from app.database.models import Subscriber, UserDigestRanking, utc_now
from app.database.repository import NewsRepository
from app.database.session import SessionLocal
from app.services.email_service import EmailSendResult, EmailService
from app.services.pdf_generator import PDFReportGenerator

logger = logging.getLogger("ai_news.agent.email")


class EmailBatchDeliveryReport(BaseModel):
    """Execution summary for a daily email briefing dispatch cycle."""

    batch_date: str
    executed_at: str
    subscribers_targeted: int = 0
    briefings_generated: int = 0
    emails_sent_live: int = 0
    exports_saved: int = 0
    results: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class EmailDeliveryAgent:
    """Agent that generates publication-grade PDF intelligence reports and emails them to subscribers."""

    def __init__(
        self,
        session: Optional[Session] = None,
        email_service: Optional[EmailService] = None,
        pdf_generator: Optional[PDFReportGenerator] = None,
    ) -> None:
        self._owns_session = session is None
        self.session = session or SessionLocal()
        self.repo = NewsRepository(session=self.session)
        self.email_service = email_service or EmailService()
        self.pdf_generator = pdf_generator or PDFReportGenerator()

    def close(self) -> None:
        """Cleanly close owned session."""
        if self._owns_session and self.session:
            self.session.close()

    def __enter__(self) -> EmailDeliveryAgent:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def get_subscribers(self, email_filter: Optional[str] = None) -> List[Subscriber]:
        """Fetch targeted active subscribers."""
        if email_filter:
            sub = self.repo.subscribers.get_by_email(email_filter)
            return [sub] if sub else []
        return self.repo.subscribers.get_active()

    def get_subscriber_rankings(
        self,
        subscriber_id: int,
        batch_date: Optional[date] = None,
    ) -> List[UserDigestRanking]:
        """Fetch ranked digests for subscriber on date (or latest available batch)."""
        b_date = batch_date or utc_now().date()
        rankings = self.repo.user_rankings.get_rankings_for_subscriber(subscriber_id, batch_date=b_date)
        if not rankings:
            # Fallback to any recent rankings if specific date had no run
            rankings = self.repo.user_rankings.get_rankings_for_subscriber(subscriber_id, batch_date=None, limit=50)
        return rankings

    def deliver_for_subscriber(
        self,
        subscriber: Subscriber,
        batch_date: date,
        send_live: bool = False,
        export_dir: str = "exports/briefings",
    ) -> EmailSendResult:
        """Generate PDF and deliver/export email briefing for a single subscriber."""
        rankings = self.get_subscriber_rankings(subscriber.id, batch_date=batch_date)
        if not rankings:
            raise ValueError(f"No ranked digests found for subscriber '{subscriber.email}' on {batch_date}.")

        sub_name = subscriber.name or subscriber.email.split("@")[0]
        sub_role = subscriber.bio_or_role or "AI Practitioner & Software Architect"
        expertise = subscriber.expertise_level or "advanced"
        interests = subscriber.interests or ["Autonomous Agents", "LLMs", "Reasoning"]

        quick_bites = [r for r in rankings if getattr(r, "tier", "featured") == "quick_bite"]
        featured = [r for r in rankings if getattr(r, "tier", "featured") == "featured"]

        if not quick_bites and not featured:
            featured = rankings

        top_item = min(featured, key=lambda x: x.rank_position) if featured else rankings[0]
        top_title = top_item.digest.title

        # Synthesize Executive Summary text
        exec_summary = (
            f"Today's intelligence dispatch for {sub_name} ({sub_role}) highlights empirical breakthroughs "
            f"and strategic infrastructure moves, culminating in our #1 recommendation: '{top_title}'. "
            f"Auxiliary and ecosystem news has been routed to the Ecosystem Radar to provide complete scannability."
        )

        # 1. Generate Publication-Grade Visual PDF (Ordered: least rank/radar first -> #1 top pick last)
        os.makedirs(export_dir, exist_ok=True)
        pdf_filename = f"Daily_AI_Briefing_{batch_date.isoformat()}.pdf"
        pdf_path = os.path.join(export_dir, pdf_filename)

        pdf_bytes = self.pdf_generator.generate_pdf(
            subscriber_name=sub_name,
            subscriber_role=sub_role,
            expertise_level=expertise,
            interests=interests,
            executive_summary=exec_summary,
            rankings=rankings,
            batch_date=batch_date,
            output_path=pdf_path,
        )

        # 2. Dispatch Email (MIME multipart with HTML body and PDF attached)
        result = self.email_service.dispatch(
            recipient_email=subscriber.email,
            subscriber_name=sub_name,
            subscriber_role=sub_role,
            executive_summary=exec_summary,
            batch_date=batch_date,
            radar_items=quick_bites,
            featured_items=featured,
            pdf_bytes=pdf_bytes,
            top_title=top_title,
            send_live=send_live,
            export_dir=export_dir,
        )

        return result

    def run_daily_delivery(
        self,
        batch_date: Optional[date] = None,
        email_filter: Optional[str] = None,
        send_live: bool = False,
        export_dir: str = "exports/briefings",
    ) -> EmailBatchDeliveryReport:
        """Execute daily email dispatch across targeted subscribers."""
        b_date = batch_date or utc_now().date()
        report = EmailBatchDeliveryReport(
            batch_date=b_date.isoformat(),
            executed_at=utc_now().isoformat(),
        )

        subscribers = self.get_subscribers(email_filter=email_filter)
        report.subscribers_targeted = len(subscribers)
        logger.info("Found %d targeted subscriber(s) for email delivery.", len(subscribers))

        if not subscribers:
            logger.warning("No active subscribers found.")
            return report

        for sub in subscribers:
            try:
                res = self.deliver_for_subscriber(
                    subscriber=sub,
                    batch_date=b_date,
                    send_live=send_live,
                    export_dir=export_dir,
                )
                report.briefings_generated += 1
                if res.sent_live:
                    report.emails_sent_live += 1
                else:
                    report.exports_saved += 1

                report.results.append({
                    "email": sub.email,
                    "name": sub.name,
                    "success": res.success,
                    "sent_live": res.sent_live,
                    "pdf_path": res.pdf_export_path,
                    "html_path": res.html_export_path,
                    "pdf_size": res.pdf_bytes_count,
                })
            except Exception as e:
                err = f"Failed to deliver briefing for subscriber {sub.email}: {e}"
                logger.error(err, exc_info=True)
                report.errors.append(err)

        return report


def main() -> None:
    """CLI runner for EmailDeliveryAgent."""
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Run Daily Email & Visual PDF Delivery Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        help="Target a specific subscriber email",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send live email via SMTP using configured SMTP_USER/SMTP_PASSWORD in .env",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Force dry run mode (save PDF and HTML locally without sending)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="exports/briefings",
        help="Directory to save generated PDF and HTML reports",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific date for rankings batch (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging",
    )

    args = parser.parse_args()

    log_lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_lvl, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    batch_date = None
    if args.date:
        batch_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    send_live = args.send and not args.dry_run

    print("\n" + "=" * 70)
    print("📬 DAILY AI INTELLIGENCE BRIEFING DELIVERY AGENT")
    print(f"Mode: {'LIVE SMTP DISPATCH' if send_live else 'LOCAL PDF & HTML EXPORT (DRY RUN)'}")
    print(f"Export Directory: {args.export_dir}")
    print("=" * 70 + "\n")

    with EmailDeliveryAgent() as agent:
        report = agent.run_daily_delivery(
            batch_date=batch_date,
            email_filter=args.email,
            send_live=send_live,
            export_dir=args.export_dir,
        )

        print("\n" + "=" * 70)
        print("📊 DELIVERY EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Batch Date:            {report.batch_date}")
        print(f"Subscribers Targeted:  {report.subscribers_targeted}")
        print(f"Briefings Generated:   {report.briefings_generated}")
        print(f"Live SMTP Dispatches:  {report.emails_sent_live}")
        print(f"Local Exports Saved:   {report.exports_saved}")

        if report.errors:
            print("\n❌ Errors Encountered:")
            for err in report.errors:
                print(f"  • {err}")

        for res in report.results:
            print("\n" + "-" * 70)
            print(f"👤 Recipient: {res['name']} ({res['email']})")
            print(f"Status:       {'✅ Sent via SMTP' if res['sent_live'] else '📄 Exported locally'}")
            print(f"PDF Report:   {res['pdf_path']} ({res['pdf_size']:,} bytes)")
            print(f"HTML Preview: {res['html_path']}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

