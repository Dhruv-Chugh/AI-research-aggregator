"""Unit tests for Email Dispatcher Service."""

from datetime import date
from types import SimpleNamespace
import os
import pytest

from app.services.email_service import EmailService


def make_mock_item(title: str, url: str, source: str, rank: int, score: float, tier: str):
    d = SimpleNamespace(title=title, url=url, source_name=source, summary="A concise summary.")
    return SimpleNamespace(
        digest=d,
        rank_position=rank,
        rank_score=score,
        tier=tier,
        quick_summary="A 3-line quick summary.",
        relevance_reasoning="Strategic fit.",
    )


def test_render_html_body_includes_metadata_and_structure_guide():
    """Verify HTML body renders date, greeting, executive overview, reading guide, and items."""
    svc = EmailService()
    it1 = make_mock_item("Claude 3.7", "https://anthropic.com", "Anthropic", 1, 95.0, "featured")
    it2 = make_mock_item("Daybreak Red", "https://openai.com", "OpenAI", 5, 70.0, "quick_bite")

    batch_date = date(2026, 11, 7)
    html = svc.render_html_body(
        subscriber_name="Alex Mercer",
        subscriber_role="AI Architect",
        executive_summary="Empirical RL breakthroughs.",
        batch_date=batch_date,
        radar_items=[it2],
        featured_items=[it1],
        top_title="Claude 3.7",
    )

    assert "Daily AI Intelligence Briefing" in html
    assert "November 07, 2026" in html
    assert "Alex Mercer" in html
    assert "AI Architect" in html
    assert "Empirical RL breakthroughs." in html
    assert "Full Visual PDF Report Attached" in html
    assert "Daily_AI_Briefing_2026-11-07.pdf" in html
    assert "Reading Order:" in html
    assert "Top Featured Strategic Stories" in html
    assert "Ecosystem Radar &amp; Quick Takes" in html or "Ecosystem Radar & Quick Takes" in html
    assert "Rank #" not in html
    assert "video thumbnails" not in html
    assert "TOP STRATEGIC PICK" in html
    assert "IEEE Computational Intelligence" in html
    assert "Abstract" in html
    assert "Index Terms" in html
    assert "[A1]" in html
    assert "[B1]" in html
    assert "synthesized by Gemini LLM" not in html
    assert "&#128206;" not in html  # no paperclip emoji
    assert "&#9889;" not in html    # no lightning emoji
    assert "&#127775;" not in html  # no star emoji


def test_render_plain_text():
    """Verify plain text fallback content."""
    svc = EmailService()
    it1 = make_mock_item("Claude 3.7", "https://anthropic.com", "Anthropic", 1, 95.0, "featured")
    it2 = make_mock_item("Daybreak Red", "https://openai.com", "OpenAI", 5, 70.0, "quick_bite")

    text = svc.render_plain_text(
        subscriber_name="Alex Mercer",
        subscriber_role="AI Architect",
        executive_summary="Summary text.",
        batch_date=date(2026, 11, 7),
        radar_items=[it2],
        featured_items=[it1],
    )

    assert "DAILY AI INTELLIGENCE BRIEFING" in text
    assert "Alex Mercer" in text
    assert "Summary text." in text
    assert "Daily_AI_Briefing_2026-11-07.pdf" in text
    assert "Rank #" not in text
    assert "[1]" not in text
    assert "[TOP RECOMMENDATION]" in text
    assert "IEEE COMPUTATIONAL INTELLIGENCE" in text
    assert "ABSTRACT" in text
    assert "INDEX TERMS" in text
    assert "[A1]" in text
    assert "[B1]" in text


def test_dispatch_safe_local_export(tmp_path):
    """Verify dispatch in dry-run/local mode saves both PDF and HTML to disk."""
    svc = EmailService()
    it1 = make_mock_item("Claude 3.7", "https://anthropic.com", "Anthropic", 1, 95.0, "featured")
    fake_pdf = b"%PDF-1.4 mock content %%EOF"

    export_dir = str(tmp_path / "briefings")
    res = svc.dispatch(
        recipient_email="test@example.com",
        subscriber_name="Alex Test",
        subscriber_role="ML Engineer",
        executive_summary="Summary text.",
        batch_date=date(2026, 11, 7),
        radar_items=[],
        featured_items=[it1],
        pdf_bytes=fake_pdf,
        top_title="Claude 3.7",
        send_live=False,
        export_dir=export_dir,
    )

    assert res.success is True
    assert res.sent_live is False
    assert os.path.exists(res.pdf_export_path)
    assert os.path.exists(res.html_export_path)
    assert os.path.getsize(res.pdf_export_path) == len(fake_pdf)


def test_dispatch_smtp_587_falls_back_to_465(tmp_path, monkeypatch):
    """Verify SMTP port 587 timeout automatically fails over to port 465 SSL."""
    import smtplib

    attempted = []

    class FakeSMTP587:
        def __init__(self, host, port, timeout=20):
            attempted.append(("STARTTLS", port))
            raise TimeoutError("Port 587 blocked")

    class FakeSMTPSSL465:
        def __init__(self, host, port, timeout=20, context=None):
            attempted.append(("SSL", port))

        def login(self, user, pwd):
            pass

        def send_message(self, msg):
            pass

        def quit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP587)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTPSSL465)

    svc = EmailService(smtp_user="bot@gmail.com", smtp_password="abcd efgh ijkl mnop", smtp_port=587, resend_api_key="")
    it1 = make_mock_item("Claude 3.7", "https://anthropic.com", "Anthropic", 1, 95.0, "featured")
    res = svc.dispatch(
        recipient_email="test@example.com",
        subscriber_name="Alex Test",
        subscriber_role="ML Engineer",
        executive_summary="Summary text.",
        batch_date=date(2026, 11, 7),
        radar_items=[],
        featured_items=[it1],
        pdf_bytes=b"%PDF-1.4 %%EOF",
        top_title="Claude 3.7",
        send_live=True,
        export_dir=str(tmp_path / "briefings"),
    )

    assert res.success is True
    assert res.sent_live is True
    assert attempted == [("STARTTLS", 587), ("SSL", 465)]


def test_dispatch_smtp_blocked_falls_back_to_resend(tmp_path, monkeypatch):
    """Verify Resend HTTPS (443) takes over cleanly when both SMTP 587 and 465 are blocked."""
    import resend
    import smtplib

    def fail_smtp(*args, **kwargs):
        raise OSError("Network is unreachable")

    sent_payloads = []
    monkeypatch.setattr(smtplib, "SMTP", fail_smtp)
    monkeypatch.setattr(smtplib, "SMTP_SSL", fail_smtp)
    monkeypatch.setattr(resend.Emails, "send", lambda params: sent_payloads.append(params) or {"id": "re_fallback_443"})

    svc = EmailService(
        smtp_user="bot@gmail.com",
        smtp_password="abcd efgh ijkl mnop",
        smtp_port=587,
        resend_api_key="re_mock_test_key",
        resend_from_email="The AI Dispatch <onboarding@resend.dev>",
    )
    it1 = make_mock_item("Claude 3.7", "https://anthropic.com", "Anthropic", 1, 95.0, "featured")
    res = svc.dispatch(
        recipient_email="test@example.com",
        subscriber_name="Alex Test",
        subscriber_role="ML Engineer",
        executive_summary="Summary text.",
        batch_date=date(2026, 11, 7),
        radar_items=[],
        featured_items=[it1],
        pdf_bytes=b"%PDF-1.4 %%EOF",
        top_title="Claude 3.7",
        send_live=True,
        export_dir=str(tmp_path / "briefings"),
    )

    assert res.success is True
    assert res.sent_live is True
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["to"] == ["test@example.com"]


def test_dispatch_smtp_auth_error_surfaces_hint_without_port_retry(tmp_path, monkeypatch):
    """Verify Gmail 535 auth failures stop cleanly without retrying port 465 or masking credentials."""
    import smtplib

    attempts = []

    class BadAuthSMTP:
        def __init__(self, host, port, timeout=20):
            attempts.append(port)

        def ehlo(self):
            pass

        def starttls(self, context=None):
            pass

        def login(self, user, pwd):
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")

        def close(self):
            pass

    monkeypatch.setattr(smtplib, "SMTP", BadAuthSMTP)

    svc = EmailService(
        smtp_user="bot@gmail.com",
        smtp_password="wrongpassword",
        smtp_port=587,
        resend_api_key="re_mock_test_key",
    )
    it1 = make_mock_item("Claude 3.7", "https://anthropic.com", "Anthropic", 1, 95.0, "featured")
    res = svc.dispatch(
        recipient_email="test@example.com",
        subscriber_name="Alex Test",
        subscriber_role="ML Engineer",
        executive_summary="Summary text.",
        batch_date=date(2026, 11, 7),
        radar_items=[],
        featured_items=[it1],
        pdf_bytes=b"%PDF-1.4 %%EOF",
        top_title="Claude 3.7",
        send_live=True,
        export_dir=str(tmp_path / "briefings"),
    )

    assert res.success is False
    assert res.sent_live is False
    assert attempts == [587]
    assert "App Password" in res.error_message


def test_dispatch_resend_only_when_no_smtp_credentials(tmp_path, monkeypatch):
    """Verify Resend works directly when SMTP credentials are absent."""
    import resend

    sent_payloads = []
    monkeypatch.setattr(resend.Emails, "send", lambda params: sent_payloads.append(params) or {"id": "re_only_1"})

    svc = EmailService(
        smtp_user="",
        smtp_password="",
        resend_api_key="re_mock_test_key",
        resend_from_email="The AI Dispatch <onboarding@resend.dev>",
    )
    it1 = make_mock_item("Claude 3.7", "https://anthropic.com", "Anthropic", 1, 95.0, "featured")
    res = svc.dispatch(
        recipient_email="test@example.com",
        subscriber_name="Alex Test",
        subscriber_role="ML Engineer",
        executive_summary="Summary text.",
        batch_date=date(2026, 11, 7),
        radar_items=[],
        featured_items=[it1],
        pdf_bytes=b"%PDF-1.4 %%EOF",
        top_title="Claude 3.7",
        send_live=True,
        export_dir=str(tmp_path / "briefings"),
    )

    assert res.success is True
    assert res.sent_live is True
    assert len(sent_payloads) == 1


