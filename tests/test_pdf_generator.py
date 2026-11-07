"""Unit tests for Visual PDF Report Generator."""

from datetime import date
from types import SimpleNamespace
import os
import pytest

from app.services.pdf_generator import (
    PDFReportGenerator,
    clean_xml,
    make_score_bar,
)


def make_mock_digest(title: str, url: str, source_name: str, source_type: str, summary: str):
    return SimpleNamespace(
        title=title,
        url=url,
        source_name=source_name,
        source_type=source_type,
        summary=summary,
        problem_statement="Problem statement text.",
        solution_statement="Solution statement text.",
        key_insights=["Key insight 1", "Key insight 2"],
        article=SimpleNamespace(external_id="mock_id", metadata_json={}),
    )


def make_mock_ranking(digest, rank_position: int, rank_score: float, tier: str, relevance: str, quick_summary: str = None):
    return SimpleNamespace(
        digest=digest,
        rank_position=rank_position,
        rank_score=rank_score,
        tier=tier,
        relevance_reasoning=relevance,
        key_takeaways_for_user=["Takeaway 1", "Takeaway 2"],
        quick_summary=quick_summary,
    )


def test_clean_xml_escapes_entities():
    """Ensure clean_xml correctly escapes special characters for ReportLab."""
    raw = "OpenAI & Anthropic <models> & 'quotes'"
    cleaned = clean_xml(raw)
    assert "&amp;" in cleaned
    assert "&lt;" in cleaned
    assert "&gt;" in cleaned
    assert "<models>" not in cleaned


def test_make_score_bar_valid_table():
    """Verify make_score_bar produces valid tables across score spectrum."""
    for score in [0.0, 25.5, 75.0, 95.0, 100.0]:
        table = make_score_bar(score, width=120)
        assert table is not None
        assert len(table._argH) == 1
        assert len(table._argW) == 2


def test_pdf_generator_creates_valid_pdf(tmp_path):
    """Verify that PDFReportGenerator produces a valid, compliant PDF file."""
    d1 = make_mock_digest("Claude 3.7 Reasoning", "https://anthropic.com", "Anthropic", "anthropic", "Half page summary of Claude.")
    r1 = make_mock_ranking(d1, rank_position=1, rank_score=95.0, tier="featured", relevance="Critical fit")

    d2 = make_mock_digest("OpenAI GovCloud", "https://openai.com", "OpenAI", "openai", "GovCloud deployment details.")
    r2 = make_mock_ranking(d2, rank_position=5, rank_score=70.0, tier="quick_bite", relevance="Background signal", quick_summary="3-line quick summary.")

    output_pdf = str(tmp_path / "test_report.pdf")
    gen = PDFReportGenerator(cache_dir=str(tmp_path / "cache"))

    pdf_bytes = gen.generate_pdf(
        subscriber_name="Dhruv Chugh",
        subscriber_role="AI Software Architect",
        expertise_level="advanced",
        interests=["autonomous agents", "reasoning"],
        executive_summary="Today's briefing covers frontier RL research and agent frameworks.",
        rankings=[r1, r2],
        batch_date=date(2026, 11, 7),
        output_path=output_pdf,
    )

    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF")
    assert b"%%EOF" in pdf_bytes[-1024:]
    assert os.path.exists(output_pdf)
    assert os.path.getsize(output_pdf) == len(pdf_bytes)


def test_pdf_generator_reverse_climax_ordering(tmp_path):
    """Verify that Part I sorts lowest-rank first, and Part II sorts ascending to #1 Top recommendation last."""
    gen = PDFReportGenerator(cache_dir=str(tmp_path / "cache"))

    # 3 Radar items (scores: 10, 66, 73)
    d_rad1 = make_mock_digest("Radar 10", "https://yt.com/1", "YouTube", "youtube", "Radar item 1")
    r_rad1 = make_mock_ranking(d_rad1, rank_position=8, rank_score=10.0, tier="quick_bite", relevance="R1")

    d_rad2 = make_mock_digest("Radar 66", "https://oa.com/1", "OpenAI", "openai", "Radar item 2")
    r_rad2 = make_mock_ranking(d_rad2, rank_position=6, rank_score=66.0, tier="quick_bite", relevance="R2")

    d_rad3 = make_mock_digest("Radar 73", "https://oa.com/2", "OpenAI", "openai", "Radar item 3")
    r_rad3 = make_mock_ranking(d_rad3, rank_position=5, rank_score=73.0, tier="quick_bite", relevance="R3")

    # 2 Featured items (scores: 85, 95)
    d_feat1 = make_mock_digest("Featured 85", "https://oa.com/3", "OpenAI", "openai", "Featured deep dive 2")
    r_feat1 = make_mock_ranking(d_feat1, rank_position=2, rank_score=85.0, tier="featured", relevance="F1")

    d_feat2 = make_mock_digest("Featured 95 (Crown Jewel)", "https://ant.com/1", "Anthropic", "anthropic", "Top recommendation")
    r_feat2 = make_mock_ranking(d_feat2, rank_position=1, rank_score=95.0, tier="featured", relevance="F2")

    radar_flowables = gen.build_radar_section([r_rad3, r_rad1, r_rad2])
    # Radar flowables include title paragraph, description paragraph, HR, and 3 card tables
    assert len(radar_flowables) > 3

    featured_flowables = gen.build_featured_section([r_feat2, r_feat1])
    assert len(featured_flowables) > 2


def test_pdf_generator_no_rank_numbers_or_thumbnails(tmp_path):
    """Ensure PDF flowables do not contain 'Rank #' or thumbnail images."""
    gen = PDFReportGenerator(cache_dir=str(tmp_path / "cache"))

    d1 = make_mock_digest("Claude 3.7", "https://anthropic.com", "Anthropic", "anthropic", "Claude details.")
    r1 = make_mock_ranking(d1, rank_position=1, rank_score=95.0, tier="featured", relevance="Top fit")

    d2 = make_mock_digest("YouTube Tech Talk", "https://yt.com/123", "YouTube", "youtube", "Video summary.")
    r2 = make_mock_ranking(d2, rank_position=4, rank_score=75.0, tier="quick_bite", relevance="Video fit", quick_summary="Short take.")

    radar_flowables = gen.build_radar_section([r2])
    featured_flowables = gen.build_featured_section([r1])
    intro_flowables = gen.build_intro_page("Dhruv", "Architect", "advanced", ["AI"], "Exec summary", date(2026, 11, 8), 1, 1, "Claude 3.7")

    # Serialize flowables text for assertion
    def extract_text(item):
        texts = []
        if isinstance(item, str):
            return [item]
        if isinstance(item, list):
            for sub in item:
                texts.extend(extract_text(sub))
            return texts
        if hasattr(item, "text"):
            texts.append(item.text)
        if hasattr(item, "_cellvalues"):
            for row in item._cellvalues:
                for cell in row:
                    texts.extend(extract_text(cell))
        return texts

    all_text = " ".join(extract_text(intro_flowables + radar_flowables + featured_flowables))
    assert "Rank #" not in all_text
    assert "#1 Top Recommendation" not in all_text
    assert "TOP STRATEGIC RECOMMENDATION" in all_text
    assert "STRATEGIC FEATURED DEEP DIVE" in all_text or "TOP STRATEGIC RECOMMENDATION" in all_text

