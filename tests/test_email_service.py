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
        subscriber_name="Dhruv Chugh",
        subscriber_role="AI Architect",
        executive_summary="Empirical RL breakthroughs.",
        batch_date=batch_date,
        radar_items=[it2],
        featured_items=[it1],
        top_title="Claude 3.7",
    )

    assert "Daily AI Intelligence Briefing" in html
    assert "November 07, 2026" in html
    assert "Dhruv Chugh" in html
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
        subscriber_name="Dhruv Chugh",
        subscriber_role="AI Architect",
        executive_summary="Summary text.",
        batch_date=date(2026, 11, 7),
        radar_items=[it2],
        featured_items=[it1],
    )

    assert "DAILY AI INTELLIGENCE BRIEFING" in text
    assert "Dhruv Chugh" in text
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


def test_dispatch_via_resend_success(tmp_path, monkeypatch):
    """Verify Resend HTTP API primary delivery sends HTML, plain text, and PDF bytes."""
    import resend

    sent_payloads = []

    def fake_send(params):
        sent_payloads.append(params)
        return {"id": "re_mock_email_123"}

    monkeypatch.setattr(resend.Emails, "send", fake_send)

    svc = EmailService(
        resend_api_key="re_mock_test_key",
        resend_from_email="The AI Dispatch <onboarding@resend.dev>",
    )
    it1 = make_mock_item("Claude 3.7", "https://anthropic.com", "Anthropic", 1, 95.0, "featured")
    fake_pdf = b"%PDF-1.4 mock content %%EOF"

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
        send_live=True,
        export_dir=str(tmp_path / "briefings"),
    )

    assert res.success is True
    assert res.sent_live is True
    assert res.error_message is None
    assert len(sent_payloads) == 1
    payload = sent_payloads[0]
    assert payload["from"] == "The AI Dispatch <onboarding@resend.dev>"
    assert payload["to"] == ["test@example.com"]
    assert payload["attachments"][0]["filename"] == "Daily_AI_Briefing_2026-11-07.pdf"
    assert payload["attachments"][0]["content"] == list(fake_pdf)
    assert payload["attachments"][0]["content_type"] == "application/pdf"


def test_dispatch_via_resend_failure_propagates_error(tmp_path, monkeypatch):
    """Verify Resend errors propagate cleanly without falling back to blocked SMTP."""
    import resend
    import smtplib

    def failing_send(_params):
        raise RuntimeError("403 invalid API key")

    smtp_called = False

    def fake_smtp(*_args, **_kwargs):
        nonlocal smtp_called
        smtp_called = True
        raise AssertionError("SMTP fallback should not be attempted when Resend is configured")

    monkeypatch.setattr(resend.Emails, "send", failing_send)
    monkeypatch.setattr(smtplib, "SMTP", fake_smtp)

    svc = EmailService(
        resend_api_key="re_mock_test_key",
        smtp_user="user@gmail.com",
        smtp_password="app_password",
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
        pdf_bytes=b"%PDF-1.4 mock %%EOF",
        top_title="Claude 3.7",
        send_live=True,
        export_dir=str(tmp_path / "briefings"),
    )

    assert res.success is False
    assert res.sent_live is False
    assert "Resend API error" in (res.error_message or "")
    assert smtp_called is False


def test_dispatch_falls_back_to_smtp_when_resend_absent(tmp_path, monkeypatch):
    """Verify SMTP fallback remains active when RESEND_API_KEY is not configured."""
    import smtplib

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def ehlo(self):
            pass

        def starttls(self):
            pass

        def login(self, _user, _password):
            pass

        def send_message(self, _msg):
            self.sent = True

        def quit(self):
            pass

    fake_instances = []

    def smtp_factory(host, port, timeout=None):
        inst = FakeSMTP(host, port, timeout=timeout)
        fake_instances.append(inst)
        return inst

    monkeypatch.setattr(smtplib, "SMTP", smtp_factory)

    svc = EmailService(
        resend_api_key="",
        smtp_user="user@gmail.com",
        smtp_password="app_password",
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
        pdf_bytes=b"%PDF-1.4 mock %%EOF",
        top_title="Claude 3.7",
        send_live=True,
        export_dir=str(tmp_path / "briefings"),
    )

    assert res.success is True
    assert res.sent_live is True
    assert len(fake_instances) == 1
    assert fake_instances[0].sent is True


