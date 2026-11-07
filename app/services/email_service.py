"""Email Dispatcher Service for Daily AI Intelligence Briefings.

Constructs publication-grade responsive HTML & plain-text emails,
attaches the generated PDF intelligence report, and delivers via SMTP
(or exports locally to disk when running in test / dry-run mode).
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import date, datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure environment variables from .env are loaded
load_dotenv()

logger = logging.getLogger("ai_news.services.email")


class EmailSendResult(BaseModel):
    """Result container for an email dispatch operation."""

    success: bool
    recipient: str
    subject: str
    pdf_filename: str
    pdf_bytes_count: int
    sent_live: bool = False
    html_export_path: Optional[str] = None
    pdf_export_path: Optional[str] = None
    error_message: Optional[str] = None


class EmailService:
    """Service responsible for composing and delivering personalized daily briefings."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        sender_name: str = "The AI Dispatch",
        sender_email: Optional[str] = None,
    ) -> None:
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(smtp_port or os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER", "")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.sender_name = sender_name
        self.sender_email = sender_email or self.smtp_user or "ai-news@digest.internal"

    @property
    def has_credentials(self) -> bool:
        """Check if live SMTP credentials are configured."""
        return bool(self.smtp_user and self.smtp_password and len(self.smtp_password.strip()) > 3)

    def render_html_body(
        self,
        subscriber_name: str,
        subscriber_role: str,
        executive_summary: str,
        batch_date: date,
        radar_items: List[Any],
        featured_items: List[Any],
        top_title: str,
    ) -> str:
        """Render a clean, responsive HTML email template previewing the attached PDF."""
        date_str = batch_date.strftime("%A, %B %d, %Y")

        # Top item info
        top_url = "#"
        top_score = 95.0
        if featured_items:
            best = min(featured_items, key=lambda x: x.rank_position)
            top_url = best.digest.url
            top_score = float(best.rank_score)

        radar_count = len(radar_items)
        featured_count = len(featured_items)

        # Build preview list of deep dives (Executive styling without rank numbers)
        featured_rows = ""
        for item in sorted(featured_items, key=lambda x: -x.rank_score):
            d = item.digest
            is_top = (item.rank_position == 1)
            badge = "&#9733; <b>TOP STRATEGIC PICK</b>" if is_top else "<b>FEATURED STORY</b>"
            badge_bg = "#ECFDF5" if is_top else "#EFF6FF"
            badge_color = "#059669" if is_top else "#2563EB"
            featured_rows += f"""
            <tr style="border-bottom: 1px solid #E2E8F0;">
                <td style="padding: 10px 8px; font-size: 13px; color: #0F172A;">
                    <span style="background-color: {badge_bg}; color: {badge_color}; font-weight: bold; font-size: 11px; padding: 3px 8px; border-radius: 4px;">{badge}</span>
                    <span style="font-weight: bold; margin-left: 8px;"><a href="{d.url}" style="color: #0F172A; text-decoration: none;">{d.title}</a></span>
                    <div style="font-size: 11px; color: #64748B; margin-top: 4px;">Source: <b>{d.source_name}</b> &bull; Relevance Score: <b>{item.rank_score:.1f}/100</b></div>
                </td>
            </tr>
            """

        radar_rows = ""
        for item in sorted(radar_items, key=lambda x: item.rank_position):
            d = item.digest
            radar_rows += f"""
            <tr style="border-bottom: 1px solid #F1F5F9;">
                <td style="padding: 6px 8px; font-size: 12px; color: #334155;">
                    <span style="color: #64748B; font-size: 10px;">[{d.source_name}]</span>
                    <a href="{d.url}" style="color: #2563EB; text-decoration: none;"><b>{d.title}</b></a>
                    <div style="color: #64748B; font-size: 11px;">{item.quick_summary or d.summary[:150] + '...'}</div>
                </td>
            </tr>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily AI Intelligence Briefing - {date_str}</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #F8FAFC; margin: 0; padding: 24px; color: #0F172A;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width: 640px; margin: 0 auto; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
    
    <!-- Top Header -->
    <tr>
      <td style="background-color: #0F172A; padding: 24px 28px; text-align: left;">
        <div style="font-size: 11px; font-weight: bold; letter-spacing: 1.5px; color: #38BDF8; text-transform: uppercase;">The AI Dispatch &bull; Daily Briefing</div>
        <div style="font-size: 22px; font-weight: bold; color: #FFFFFF; margin-top: 4px;">Daily AI Intelligence Briefing</div>
        <div style="font-size: 13px; color: #94A3B8; margin-top: 4px;">{date_str}</div>
      </td>
    </tr>

    <!-- Subscriber Greeting & Executive Overview -->
    <tr>
      <td style="padding: 24px 28px;">
        <p style="font-size: 15px; line-height: 1.5; color: #1E293B; margin-top: 0;">
          Good morning <b>{subscriber_name}</b>,
        </p>
        <p style="font-size: 14px; line-height: 1.6; color: #334155;">
          Here is your personalized intelligence briefing for today, tailored specifically to your work as an <b>{subscriber_role}</b>.
        </p>

        <!-- Executive Summary Callout -->
        <div style="background-color: #F1F5F9; border-left: 4px solid #2563EB; padding: 14px 16px; border-radius: 4px; margin: 18px 0;">
          <div style="font-size: 11px; font-weight: bold; color: #2563EB; text-transform: uppercase; margin-bottom: 4px;">Executive Overview</div>
          <div style="font-size: 13px; line-height: 1.6; color: #0F172A;">{executive_summary}</div>
        </div>

        <!-- Attachment Notice Banner -->
        <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; padding: 14px 18px; border-radius: 6px; margin: 20px 0;">
          <div style="display: flex; align-items: center;">
            <span style="font-size: 22px; margin-right: 12px;">&#128206;</span>
            <div>
              <div style="font-size: 14px; font-weight: bold; color: #065F46;">Full Visual PDF Report Attached</div>
              <div style="font-size: 12px; color: #047857; margin-top: 2px;">
                Open <b>Daily_AI_Briefing_{batch_date.isoformat()}.pdf</b> for in-depth executive analyses, strategic relevance metrics, problem/solution comparisons, and role takeaways.
              </div>
            </div>
          </div>
        </div>

        <!-- How It's Structured Explanation -->
        <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; padding: 12px 16px; border-radius: 6px; font-size: 12px; color: #92400E; line-height: 1.5; margin-bottom: 20px;">
          <b>&#9889; Reading Order:</b> The attached PDF is structured progressively: It begins with the <b>Ecosystem Radar</b> (short 3-4 line briefs of broader market signals), and leads into your <b>Featured Strategic Deep Dives</b>, culminating in the <b>Top Strategic Recommendation</b>.
        </div>

        <!-- Section 1 Preview: Featured Deep Dives -->
        <div style="font-size: 14px; font-weight: bold; color: #0F172A; margin: 24px 0 10px 0; border-bottom: 2px solid #0F172A; padding-bottom: 4px;">
          &#127775; Top Featured Strategic Stories ({featured_count})
        </div>
        <table width="100%" cellspacing="0" cellpadding="0" style="margin-bottom: 20px;">
          {featured_rows}
        </table>

        <!-- Section 2 Preview: Radar Quick Takes -->
        <div style="font-size: 14px; font-weight: bold; color: #0F172A; margin: 24px 0 10px 0; border-bottom: 2px solid #0F172A; padding-bottom: 4px;">
          &#9889; Ecosystem Radar & Quick Takes ({radar_count})
        </div>
        <table width="100%" cellspacing="0" cellpadding="0" style="margin-bottom: 20px;">
          {radar_rows}
        </table>

        <p style="font-size: 12px; color: #64748B; margin-top: 28px; line-height: 1.5;">
          This briefing was automatically aggregated, synthesized by Gemini LLM, and ranked for your profile.<br/>
          Check the attached PDF for full graphics, metrics, and detailed problem-solution breakdowns.
        </p>
      </td>
    </tr>

    <!-- Footer -->
    <tr>
      <td style="background-color: #F8FAFC; border-top: 1px solid #E2E8F0; padding: 16px 28px; text-align: center; font-size: 11px; color: #94A3B8;">
        The AI Dispatch &bull; Automated Daily Briefing Service &bull; {date_str}
      </td>
    </tr>

  </table>
</body>
</html>
"""
        return html

    def render_plain_text(
        self,
        subscriber_name: str,
        subscriber_role: str,
        executive_summary: str,
        batch_date: date,
        radar_items: List[Any],
        featured_items: List[Any],
    ) -> str:
        """Render fallback plain-text email."""
        date_str = batch_date.strftime("%B %d, %Y")
        lines = [
            f"DAILY AI INTELLIGENCE BRIEFING - {date_str}",
            "=" * 60,
            f"Personalized for: {subscriber_name} ({subscriber_role})",
            "",
            "EXECUTIVE OVERVIEW:",
            executive_summary,
            "",
            "NOTE: The complete publication-grade intelligence briefing is attached",
            f"as a high-resolution PDF: Daily_AI_Briefing_{batch_date.isoformat()}.pdf",
            "",
            "STRUCTURE: Organized from Ecosystem Radar (3-4 line quick bites) to",
            "Top Featured Deep Dives, culminating in the Top Strategic Recommendation.",
            "",
            "=" * 60,
            f"FEATURED DEEP DIVES ({len(featured_items)}):",
        ]
        for it in sorted(featured_items, key=lambda x: -x.rank_score):
            d = it.digest
            is_top = (it.rank_position == 1)
            prefix = "[TOP RECOMMENDATION]" if is_top else "[FEATURED]"
            lines.append(f"  * {prefix} {d.title} (Relevance Score: {it.rank_score:.1f}/100)")
            lines.append(f"      Source: {d.source_name} | URL: {d.url}")
            lines.append(f"      Strategic Context: {it.relevance_reasoning}")
            lines.append("")

        lines.extend([
            "=" * 60,
            f"ECOSYSTEM RADAR & QUICK TAKES ({len(radar_items)}):",
        ])
        for it in sorted(radar_items, key=lambda x: it.rank_position):
            d = it.digest
            brief = it.quick_summary or d.summary[:150] + "..."
            lines.append(f"  - [{d.source_name}] {d.title} (Score: {it.rank_score:.1f}/100)")
            lines.append(f"    URL: {d.url}")
            lines.append(f"    Brief: {brief}")
            lines.append("")

        return "\n".join(lines)

    def dispatch(
        self,
        recipient_email: str,
        subscriber_name: str,
        subscriber_role: str,
        executive_summary: str,
        batch_date: date,
        radar_items: List[Any],
        featured_items: List[Any],
        pdf_bytes: bytes,
        top_title: str,
        send_live: bool = False,
        export_dir: str = "exports/briefings",
    ) -> EmailSendResult:
        """Compose email, attach PDF, and either send via SMTP or export locally."""
        date_str = batch_date.strftime("%A, %B %d, %Y")
        subject = f"🎯 AI Daily Intelligence Briefing — {date_str}"
        pdf_filename = f"Daily_AI_Briefing_{batch_date.isoformat()}.pdf"

        html_body = self.render_html_body(
            subscriber_name=subscriber_name,
            subscriber_role=subscriber_role,
            executive_summary=executive_summary,
            batch_date=batch_date,
            radar_items=radar_items,
            featured_items=featured_items,
            top_title=top_title,
        )
        plain_text = self.render_plain_text(
            subscriber_name=subscriber_name,
            subscriber_role=subscriber_role,
            executive_summary=executive_summary,
            batch_date=batch_date,
            radar_items=radar_items,
            featured_items=featured_items,
        )

        # Build MIME Message
        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = f"{self.sender_name} <{self.sender_email}>"
        msg["To"] = recipient_email
        msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")

        # Body alternative (plain text + HTML)
        alt_part = MIMEMultipart("alternative")
        alt_part.attach(MIMEText(plain_text, "plain", "utf-8"))
        alt_part.attach(MIMEText(html_body, "html", "utf-8"))
        msg.attach(alt_part)

        # Attach PDF
        pdf_attachment = MIMEApplication(pdf_bytes, _subtype="pdf")
        pdf_attachment.add_header("Content-Disposition", "attachment", filename=pdf_filename)
        msg.attach(pdf_attachment)

        # Check export directory
        os.makedirs(export_dir, exist_ok=True)
        pdf_export_path = os.path.join(export_dir, pdf_filename)
        html_export_path = os.path.join(export_dir, f"Daily_AI_Briefing_{batch_date.isoformat()}.html")

        # Always save local export
        with open(pdf_export_path, "wb") as f:
            f.write(pdf_bytes)
        with open(html_export_path, "w", encoding="utf-8") as f:
            f.write(html_body)

        # Live SMTP delivery if requested and configured
        if send_live and self.has_credentials:
            try:
                # Strip spaces if user pasted 16-character app password formatted as 'xxxx xxxx xxxx xxxx'
                clean_password = self.smtp_password.replace(" ", "")
                logger.info("Connecting to SMTP server: %s:%d...", self.smtp_host, self.smtp_port)

                if self.smtp_port == 465:
                    server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=20)
                else:
                    server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20)
                    server.ehlo()
                    server.starttls()
                    server.ehlo()

                try:
                    server.login(self.smtp_user, clean_password)
                    server.send_message(msg)
                    server.quit()
                finally:
                    try:
                        server.close()
                    except Exception:
                        pass

                logger.info("✅ Live email successfully sent to: %s with attachment %s", recipient_email, pdf_filename)
                return EmailSendResult(
                    success=True,
                    recipient=recipient_email,
                    subject=subject,
                    pdf_filename=pdf_filename,
                    pdf_bytes_count=len(pdf_bytes),
                    sent_live=True,
                    html_export_path=html_export_path,
                    pdf_export_path=pdf_export_path,
                )
            except Exception as e:
                err_str = str(e)
                if "535" in err_str or "BadCredentials" in err_str or "Connection unexpectedly closed" in err_str:
                    hint = (
                        "\n💡 Note: Gmail requires a 16-character 'App Password' instead of your standard account password.\n"
                        "To generate one in 30 seconds:\n"
                        "1. Open https://myaccount.google.com/apppasswords\n"
                        "2. Name it 'AI News Aggregator' and click Create.\n"
                        "3. Copy the 16-character code (e.g. 'abcd efgh ijkl mnop') and update SMTP_PASSWORD in .env."
                    )
                    err_msg = f"SMTP authentication failed (Gmail BadCredentials): {e}.{hint}"
                else:
                    err_msg = f"SMTP dispatch failed: {e}"

                logger.error(err_msg)
                return EmailSendResult(
                    success=False,
                    recipient=recipient_email,
                    subject=subject,
                    pdf_filename=pdf_filename,
                    pdf_bytes_count=len(pdf_bytes),
                    sent_live=False,
                    html_export_path=html_export_path,
                    pdf_export_path=pdf_export_path,
                    error_message=err_msg,
                )

        # Safe local export (dry run or credentials unconfigured)
        logger.info(
            "📄 Briefing PDF and HTML preview exported to %s. (Set SMTP_USER/SMTP_PASSWORD in .env for live sending)",
            export_dir,
        )
        return EmailSendResult(
            success=True,
            recipient=recipient_email,
            subject=subject,
            pdf_filename=pdf_filename,
            pdf_bytes_count=len(pdf_bytes),
            sent_live=False,
            html_export_path=html_export_path,
            pdf_export_path=pdf_export_path,
        )

