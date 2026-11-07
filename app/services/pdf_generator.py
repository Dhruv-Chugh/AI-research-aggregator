"""Visually Appealing PDF Intelligence Briefing Generator using ReportLab.

Renders an executive, publication-grade AI intelligence briefing:
- Page 1: Masthead Cover & Intro page with subscriber profile, executive overview,
  and reading guide explaining the reverse-climax structure (Radar/Short summaries first,
  progressing to the Top Strategic Recommendation).
- Part I: Ecosystem Radar & Quick Bites (3-4 line concise summaries).
- Part II: Featured Strategic Deep Dives (building up to the Top Strategic Recommendation,
  with visual score bars, executive metadata cards, problem/solution matrices, and actionable takeaways).
- Dynamic two-pass NumberedCanvas for 'Page X of Y' and running headers.
"""

from __future__ import annotations

import html
import io
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("ai_news.services.pdf")

# Color Palette: Modern Editorial Deep Navy & Slate
NAVY_PRIMARY = colors.HexColor("#0F172A")      # Slate 900
BRAND_BLUE = colors.HexColor("#2563EB")        # Blue 600
BRAND_LIGHT_BG = colors.HexColor("#F1F5F9")    # Slate 100
CARD_BG = colors.HexColor("#F8FAFC")           # Slate 50
CARD_BORDER = colors.HexColor("#CBD5E1")       # Slate 300
TEXT_MAIN = colors.HexColor("#0F172A")         # Slate 900
TEXT_MUTED = colors.HexColor("#475569")        # Slate 600
TEXT_LIGHT = colors.HexColor("#94A3B8")        # Slate 400
ACCENT_GREEN = colors.HexColor("#059669")      # Emerald 600
ACCENT_GREEN_BG = colors.HexColor("#ECFDF5")   # Emerald 50
ACCENT_AMBER = colors.HexColor("#D97706")      # Amber 600
ACCENT_AMBER_BG = colors.HexColor("#FFFBEB")   # Amber 50
ACCENT_RED = colors.HexColor("#DC2626")        # Red 600
ACCENT_RED_BG = colors.HexColor("#FEF2F2")     # Red 50
DIVIDER_COLOR = colors.HexColor("#E2E8F0")     # Slate 200

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_HORIZONTAL = 40
MARGIN_VERTICAL = 40
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN_HORIZONTAL)


def clean_xml(text: Optional[str]) -> str:
    """Escape text safely for ReportLab Paragraph XML parser."""
    if not text:
        return ""
    # Normalize whitespace, preserve content
    clean = " ".join(str(text).strip().split())
    return html.escape(clean)


def make_score_bar(score: float, width: float = 120, height: float = 7) -> Table:
    """Create a colored two-cell horizontal progress bar representing a score out of 100."""
    clamped = max(0.0, min(100.0, float(score)))
    pct = clamped / 100.0
    w_fill = max(4.0, width * pct)
    w_empty = max(0.0, width - w_fill)

    if clamped >= 85:
        bar_color = ACCENT_GREEN
    elif clamped >= 70:
        bar_color = BRAND_BLUE
    elif clamped >= 45:
        bar_color = ACCENT_AMBER
    else:
        bar_color = TEXT_LIGHT

    t = Table([["", ""]], colWidths=[w_fill, w_empty], rowHeights=[height])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), bar_color),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    return t


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas that computes total pages dynamically and draws running headers/footers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []
        self.briefing_date_str = ""
        self.subscriber_name = ""

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        # Suppress running header/footer on Cover / Intro page (Page 1)
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 7)
        self.setFillColor(TEXT_MUTED)

        # Running Header
        header_y = PAGE_HEIGHT - 28
        self.drawString(MARGIN_HORIZONTAL, header_y, "THE AI DISPATCH | DAILY INTELLIGENCE BRIEFING")
        self.setFont("Helvetica", 7)
        self.drawRightString(PAGE_WIDTH - MARGIN_HORIZONTAL, header_y, self.briefing_date_str)

        self.setStrokeColor(DIVIDER_COLOR)
        self.setLineWidth(0.6)
        self.line(MARGIN_HORIZONTAL, header_y - 4, PAGE_WIDTH - MARGIN_HORIZONTAL, header_y - 4)

        # Running Footer
        footer_y = 28
        self.line(MARGIN_HORIZONTAL, footer_y + 10, PAGE_WIDTH - MARGIN_HORIZONTAL, footer_y + 10)
        self.setFont("Helvetica", 7)
        user_label = f"Personalized for {self.subscriber_name}" if self.subscriber_name else "Daily Briefing"
        self.drawString(MARGIN_HORIZONTAL, footer_y, user_label)
        self.drawRightString(
            PAGE_WIDTH - MARGIN_HORIZONTAL,
            footer_y,
            f"Page {self._pageNumber} of {page_count}",
        )
        self.restoreState()


class PDFReportGenerator:
    """Generates publication-quality, visually appealing intelligence briefing PDFs."""

    def __init__(self, cache_dir: str = "exports/thumbnails") -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_styles()

    def _init_styles(self) -> None:
        """Initialize polished typographic styles."""
        self.base_styles = getSampleStyleSheet()

        # Masthead & Main Titles
        self.style_masthead = ParagraphStyle(
            "Masthead",
            parent=self.base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY_PRIMARY,
            spaceAfter=4,
        )
        self.style_subtitle = ParagraphStyle(
            "Subtitle",
            parent=self.base_styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=BRAND_BLUE,
            spaceAfter=12,
        )

        # Section Headers
        self.style_section_h1 = ParagraphStyle(
            "SectionH1",
            parent=self.base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY_PRIMARY,
            spaceBefore=12,
            spaceAfter=4,
        )
        self.style_section_desc = ParagraphStyle(
            "SectionDesc",
            parent=self.base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=TEXT_MUTED,
            spaceAfter=10,
        )

        # Card & Item Titles
        self.style_item_title = ParagraphStyle(
            "ItemTitle",
            parent=self.base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=NAVY_PRIMARY,
            spaceAfter=4,
        )
        self.style_item_meta = ParagraphStyle(
            "ItemMeta",
            parent=self.base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=TEXT_MUTED,
            spaceAfter=6,
        )

        # Body text & Summaries
        self.style_body = ParagraphStyle(
            "BodyTextCustom",
            parent=self.base_styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13.5,
            textColor=TEXT_MAIN,
            spaceAfter=6,
        )
        self.style_quick_brief = ParagraphStyle(
            "QuickBriefText",
            parent=self.base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12.5,
            textColor=TEXT_MAIN,
            spaceAfter=3,
        )
        self.style_relevance = ParagraphStyle(
            "RelevanceText",
            parent=self.base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11.5,
            textColor=TEXT_MUTED,
        )

        # Callouts & Boxes
        self.style_box_header = ParagraphStyle(
            "BoxHeader",
            parent=self.base_styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=NAVY_PRIMARY,
            spaceAfter=2,
        )
        self.style_box_body = ParagraphStyle(
            "BoxBody",
            parent=self.base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11.5,
            textColor=TEXT_MAIN,
        )
        self.style_bullet = ParagraphStyle(
            "BulletCustom",
            parent=self.base_styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=TEXT_MAIN,
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=3,
        )

    def _fetch_thumbnail(self, url: Optional[str], external_id: str) -> Optional[RLImage]:
        """Download or retrieve cached thumbnail image, scaling gracefully to 16:9 ratio."""
        if not url:
            return None

        # Sanitize cache filename
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", external_id)
        cache_path = os.path.join(self.cache_dir, f"{safe_id}.jpg")

        image_bytes: Optional[bytes] = None

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    image_bytes = f.read()
            except Exception as e:
                logger.debug("Failed to read cached thumbnail %s: %s", cache_path, e)

        if not image_bytes:
            try:
                with httpx.Client(timeout=4.0, follow_redirects=True) as client:
                    resp = client.get(url)
                    if resp.status_code == 200 and len(resp.content) > 500:
                        image_bytes = resp.content
                        with open(cache_path, "wb") as f:
                            f.write(image_bytes)
            except Exception as e:
                logger.debug("Could not download thumbnail from %s: %s", url, e)
                return None

        if not image_bytes:
            return None

        try:
            pil_img = PILImage.open(io.BytesIO(image_bytes))
            # Convert to RGB if needed
            if pil_img.mode in ("RGBA", "P"):
                pil_img = pil_img.convert("RGB")

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            buf.seek(0)

            # Target width 180, height 101 (16:9)
            target_w = 175
            target_h = 98
            return RLImage(buf, width=target_w, height=target_h)
        except Exception as e:
            logger.debug("Error processing thumbnail image: %s", e)
            return None

    def build_intro_page(
        self,
        subscriber_name: str,
        subscriber_role: str,
        expertise_level: str,
        interests: List[str],
        executive_summary: str,
        batch_date: date,
        radar_count: int,
        featured_count: int,
        top_title: str,
    ) -> List[Any]:
        """Construct Page 1: Masthead Cover & Intro page with structure guide."""
        flowables: List[Any] = []

        date_str = batch_date.strftime("%A, %B %d, %Y")

        # 1. Masthead Header Table
        masthead_left = [
            Paragraph("<b>THE AI DISPATCH</b>", self.style_masthead),
            Paragraph(f"<b>DAILY INTELLIGENCE BRIEFING</b> &bull; {clean_xml(date_str).upper()}", self.style_subtitle),
        ]
        badge_p = Paragraph(
            f"<para align='right'><font size='7' color='#2563EB'><b>VOLUME II &bull; ISSUE 24</b></font><br/>"
            f"<font size='8' color='#0F172A'><b>PERSONALIZED EDITION</b></font></para>",
            self.style_item_meta,
        )
        t_mast = Table([[masthead_left, badge_p]], colWidths=[CONTENT_WIDTH * 0.75, CONTENT_WIDTH * 0.25])
        t_mast.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        flowables.append(t_mast)

        flowables.append(HRFlowable(width="100%", thickness=2, color=NAVY_PRIMARY, spaceBefore=2, spaceAfter=10))

        # 2. Subscriber Profile Card
        interests_str = ", ".join(interests[:4]) if interests else "General AI & Engineering"
        profile_content = [
            Paragraph(
                f"<b>READER PROFILE:</b> {clean_xml(subscriber_name)} &bull; "
                f"<font color='#2563EB'><b>{clean_xml(subscriber_role)}</b></font>",
                self.style_box_header,
            ),
            Paragraph(
                f"<b>Expertise:</b> {clean_xml(expertise_level.capitalize())} &nbsp;|&nbsp; "
                f"<b>Active Focus Areas:</b> {clean_xml(interests_str)}",
                self.style_box_body,
            ),
        ]
        t_prof = Table([[profile_content]], colWidths=[CONTENT_WIDTH])
        t_prof.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 1, CARD_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        flowables.append(t_prof)
        flowables.append(Spacer(1, 10))

        # 3. Executive Overview Box
        exec_text = clean_xml(executive_summary) if executive_summary else (
            "Today's intelligence cycle synthesizes key developments across Anthropic research, "
            "OpenAI enterprise and agentic infrastructure, and strategic founder interviews."
        )
        overview_content = [
            Paragraph("<b>EXECUTIVE INTELLIGENCE SYNTHESIS</b>", self.style_box_header),
            Spacer(1, 3),
            Paragraph(exec_text, self.style_body),
        ]
        t_over = Table([[overview_content]], colWidths=[CONTENT_WIDTH])
        t_over.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT_BG),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#93C5FD")),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        flowables.append(t_over)
        flowables.append(Spacer(1, 12))

        # 4. Briefing Navigation & Reading Guide
        guide_text = (
            f"<b>HOW THIS BRIEFING IS ORGANIZED (PROGRESSIVE STRUCTURE):</b><br/>"
            f"To maximize efficiency and provide natural cognitive progression, this report is divided into two distinct sections:<br/>"
            f"&bull; <b>Part I &mdash; Ecosystem Radar & Quick Bites ({radar_count} items):</b> Opens the briefing with compact 3&ndash;4 line "
            f"operational summaries covering broader market movements, ecosystem announcements, and technical signals to provide fast, high-level scanning.<br/>"
            f"&bull; <b>Part II &mdash; Featured Strategic Deep Dives ({featured_count} items):</b> Expands into comprehensive technical deep dives, "
            f"architectural problem/solution analyses, and actionable takeaways, culminating in today's "
            f"<b>Lead Strategic Analysis:</b> <i>'{clean_xml(top_title[:65])}...'</i>."
        )
        t_guide = Table([[Paragraph(guide_text, self.style_body)]], colWidths=[CONTENT_WIDTH])
        t_guide.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ACCENT_AMBER_BG),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FCD34D")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        flowables.append(t_guide)
        flowables.append(Spacer(1, 12))

        # 5. Fast Metric Callout Strip
        metrics = [
            [
                Paragraph("<para align='center'><font size='14' color='#0F172A'><b>8</b></font><br/><font size='7' color='#64748B'>ITEMS ANALYZED</font></para>", self.style_item_meta),
                Paragraph("<para align='center'><font size='14' color='#2563EB'><b>3</b></font><br/><font size='7' color='#64748B'>SOURCES COVERED</font></para>", self.style_item_meta),
                Paragraph(f"<para align='center'><font size='14' color='#059669'><b>{featured_count}</b></font><br/><font size='7' color='#64748B'>DEEP DIVES</font></para>", self.style_item_meta),
                Paragraph(f"<para align='center'><font size='14' color='#D97706'><b>{radar_count}</b></font><br/><font size='7' color='#64748B'>RADAR BRIEFS</font></para>", self.style_item_meta),
            ]
        ]
        t_met = Table(metrics, colWidths=[CONTENT_WIDTH / 4] * 4)
        t_met.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, CARD_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, CARD_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        flowables.append(t_met)

        # Break to Page 2 for Part I
        flowables.append(PageBreak())
        return flowables

    def build_radar_section(self, quick_bites: List[Any]) -> List[Any]:
        """Construct Part I: Ecosystem Radar (Least rank / lowest score first)."""
        flowables: List[Any] = []
        if not quick_bites:
            return flowables

        # Section Heading
        flowables.append(Paragraph("<b>PART I: ECOSYSTEM RADAR &amp; IN BRIEF</b>", self.style_section_h1))
        flowables.append(
            Paragraph(
                "Compact tactical briefings on broader industry movements, platform updates, and background signals.",
                self.style_section_desc,
            )
        )
        flowables.append(HRFlowable(width="100%", thickness=1, color=DIVIDER_COLOR, spaceBefore=2, spaceAfter=8))

        # Ascending sort: lowest score first to build momentum
        sorted_radar = sorted(quick_bites, key=lambda x: (x.rank_score, -x.rank_position))

        for idx, item in enumerate(sorted_radar, 1):
            digest = item.digest
            score = float(item.rank_score)
            src_name = clean_xml(digest.source_name)
            src_type = clean_xml(digest.source_type)
            title = clean_xml(digest.title)
            url = digest.url
            quick_summary = clean_xml(item.quick_summary or digest.summary[:280] + "...")
            relevance = clean_xml(item.relevance_reasoning)

            # Source badge & score bar (Clean executive header without rank numbers)
            header_cell = [
                Paragraph(
                    f"<font color='#2563EB'><b>[{src_name} &bull; {src_type.upper()}]</b></font> &nbsp;&bull;&nbsp; "
                    f"Relevance Score: <b>{score:.1f}/100</b>",
                    self.style_item_meta,
                ),
                Paragraph(
                    f"<a href='{url}' color='#0F172A'><u><b>{title}</b></u></a>",
                    self.style_item_title,
                ),
            ]

            body_cell = [
                Paragraph(f"<b>Quick Brief:</b> {quick_summary}", self.style_quick_brief),
                Paragraph(f"<b>Role Context:</b> {relevance}", self.style_relevance),
            ]

            card_table = Table([[header_cell], [body_cell]], colWidths=[CONTENT_WIDTH])
            card_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, CARD_BORDER),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, DIVIDER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))

            flowables.append(card_table)
            flowables.append(Spacer(1, 7))

        flowables.append(Spacer(1, 10))
        flowables.append(PageBreak())
        return flowables

    def build_featured_section(self, featured_items: List[Any]) -> List[Any]:
        """Construct Part II: Strategic Deep Dives (Ascending climax order -> #1 Top Pick last)."""
        flowables: List[Any] = []
        if not featured_items:
            return flowables

        # Section Heading
        flowables.append(Paragraph("<b>PART II: STRATEGIC FEATURED DEEP DIVES</b>", self.style_section_h1))
        flowables.append(
            Paragraph(
                "High-conviction architectural and technical breakthroughs tailored to your focus areas, "
                "progressing toward today's <b>Lead Strategic Recommendation</b>.",
                self.style_section_desc,
            )
        )
        flowables.append(HRFlowable(width="100%", thickness=1, color=DIVIDER_COLOR, spaceBefore=2, spaceAfter=10))

        # Order ascending toward climax: lowest score first -> Lead Strategic Recommendation last
        sorted_featured = sorted(featured_items, key=lambda x: (x.rank_score, -x.rank_position))

        for item in sorted_featured:
            digest = item.digest
            score = float(item.rank_score)
            is_top = (item.rank_position == 1)

            src_name = clean_xml(digest.source_name)
            src_type = clean_xml(digest.source_type)
            title = clean_xml(digest.title)
            url = digest.url

            # Executive badge (no rank numbers)
            if is_top:
                badge_html = "<font color='#059669'><b>&#9733; TOP STRATEGIC RECOMMENDATION (LEAD ANALYSIS)</b></font>"
                border_color = ACCENT_GREEN
                border_width = 1.5
                header_bg = ACCENT_GREEN_BG
            else:
                badge_html = "<font color='#2563EB'><b>STRATEGIC FEATURED DEEP DIVE</b></font>"
                border_color = CARD_BORDER
                border_width = 0.8
                header_bg = CARD_BG

            # 1. Header Banner with Score Bar
            score_bar = make_score_bar(score, width=100)
            header_left = Paragraph(
                f"{badge_html} &nbsp;|&nbsp; <b>{src_name}</b> ({src_type.upper()})<br/>"
                f"<a href='{url}' color='#0F172A'><u><b>{title}</b></u></a>",
                self.style_item_title,
            )
            score_right = [
                Paragraph(f"<para align='right'><font size='10' color='#0F172A'><b>{score:.1f}</b></font><font size='7' color='#64748B'>/100</font></para>", self.style_item_meta),
                score_bar,
            ]
            t_head = Table([[header_left, score_right]], colWidths=[CONTENT_WIDTH * 0.72, CONTENT_WIDTH * 0.28])
            t_head.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))

            item_flowables: List[Any] = [t_head, Spacer(1, 4)]

            # 2. Executive Metadata Card (Streamlined typographic layout without video thumbnails)
            url_display = url if len(url) <= 65 else f"{url[:62]}..."
            meta_status = "<b>★ LEAD STORY</b>" if is_top else "FEATURED ANALYSIS"
            meta_rows = [
                [
                    Paragraph(f"<b>PLATFORM / SOURCE:</b> {src_name} &bull; {src_type.upper()}", self.style_box_header),
                    Paragraph(f"<b>COVERAGE STATUS:</b> <font color='#2563EB'>{meta_status}</font>", self.style_box_header),
                ],
                [
                    Paragraph(f"<b>RESOURCE LINK:</b> <a href='{url}' color='#2563EB'><u>{clean_xml(url_display)}</u></a>", self.style_box_body),
                    Paragraph(f"<b>RELEVANCE RATING:</b> <b>{score:.1f}</b> / 100", self.style_box_body),
                ],
            ]
            t_meta = Table(meta_rows, colWidths=[(CONTENT_WIDTH - 20) * 0.65, (CONTENT_WIDTH - 20) * 0.35])
            t_meta.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), BRAND_LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            item_flowables.append(t_meta)
            item_flowables.append(Spacer(1, 5))

            # Strategic Relevance Rationale Box
            rationale_bg = ACCENT_GREEN_BG if is_top else CARD_BG
            rationale_border = colors.HexColor("#A7F3D0") if is_top else colors.HexColor("#CBD5E1")
            t_rat = Table(
                [[Paragraph(f"<b>Strategic Fit for Your Role:</b> <i>{clean_xml(item.relevance_reasoning)}</i>", self.style_body)]],
                colWidths=[CONTENT_WIDTH - 20],
            )
            t_rat.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), rationale_bg),
                ("BOX", (0, 0), (-1, -1), 0.5, rationale_border),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]))
            item_flowables.append(t_rat)
            item_flowables.append(Spacer(1, 5))

            # 3. Comprehensive Half-Page Summary
            summary_clean = clean_xml(digest.summary)
            item_flowables.append(Paragraph("<b>EXECUTIVE ANALYSIS &amp; SYNTHESIS:</b>", self.style_box_header))
            item_flowables.append(Paragraph(summary_clean, self.style_body))

            # 4. Problem Statement vs Breakthrough Solution (Two-tone Box)
            prob_text = clean_xml(digest.problem_statement) if digest.problem_statement else "N/A"
            sol_text = clean_xml(digest.solution_statement) if digest.solution_statement else "N/A"

            box_table = Table(
                [
                    [
                        Paragraph(f"<b>PROBLEM HIGHLIGHTED:</b><br/>{prob_text}", self.style_box_body),
                        Paragraph(f"<b>SOLUTION &amp; BREAKTHROUGH:</b><br/>{sol_text}", self.style_box_body),
                    ]
                ],
                colWidths=[(CONTENT_WIDTH - 20) / 2, (CONTENT_WIDTH - 20) / 2],
            )
            box_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), ACCENT_RED_BG),
                ("BACKGROUND", (1, 0), (1, 0), ACCENT_GREEN_BG),
                ("BOX", (0, 0), (0, 0), 0.5, colors.HexColor("#FCA5A5")),
                ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#6EE7B7")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            item_flowables.append(box_table)
            item_flowables.append(Spacer(1, 6))

            # 5. Actionable Takeaways
            takeaways = item.key_takeaways_for_user or digest.key_insights or []
            if takeaways:
                item_flowables.append(Paragraph("<b>ACTIONABLE TAKEAWAYS FOR YOUR ROLE:</b>", self.style_box_header))
                for t in takeaways:
                    item_flowables.append(Paragraph(f"&bull; {clean_xml(t)}", self.style_bullet))

            # Wrap entire deep dive inside an outer card
            t_outer = Table([[item_flowables]], colWidths=[CONTENT_WIDTH])
            t_outer.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
                ("BOX", (0, 0), (-1, -1), border_width, border_color),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]))

            flowables.append(t_outer)
            flowables.append(Spacer(1, 12))

        return flowables

    def generate_pdf(
        self,
        subscriber_name: str,
        subscriber_role: str,
        expertise_level: str,
        interests: List[str],
        executive_summary: str,
        rankings: List[Any],
        batch_date: date,
        output_path: Optional[str] = None,
    ) -> bytes:
        """Render the complete visual PDF report and return bytes (and optionally save to disk)."""
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=MARGIN_HORIZONTAL,
            rightMargin=MARGIN_HORIZONTAL,
            topMargin=MARGIN_VERTICAL,
            bottomMargin=MARGIN_VERTICAL,
            title=f"AI Intelligence Briefing - {batch_date.isoformat()}",
            author="AI News Aggregator Agent",
        )

        quick_bites = [r for r in rankings if getattr(r, "tier", "featured") == "quick_bite"]
        featured = [r for r in rankings if getattr(r, "tier", "featured") == "featured"]

        if not quick_bites and not featured:
            featured = rankings

        top_item = min(featured, key=lambda x: x.rank_position) if featured else None
        top_title = top_item.digest.title if top_item else "Frontier AI Breakthroughs"

        flowables: List[Any] = []

        # 1. Page 1: Cover / Intro Page
        intro_flowables = self.build_intro_page(
            subscriber_name=subscriber_name,
            subscriber_role=subscriber_role,
            expertise_level=expertise_level,
            interests=interests,
            executive_summary=executive_summary,
            batch_date=batch_date,
            radar_count=len(quick_bites),
            featured_count=len(featured),
            top_title=top_title,
        )
        flowables.extend(intro_flowables)

        # 2. Part I: Ecosystem Radar & Quick Bites (Ascending from least score / least rank)
        radar_flowables = self.build_radar_section(quick_bites)
        flowables.extend(radar_flowables)

        # 3. Part II: Featured Strategic Deep Dives (Ascending to #1 Top Pick)
        featured_flowables = self.build_featured_section(featured)
        flowables.extend(featured_flowables)

        # Custom Canvas factory with metadata
        def canvas_maker(*args: Any, **kwargs: Any) -> NumberedCanvas:
            c = NumberedCanvas(*args, **kwargs)
            c.briefing_date_str = batch_date.strftime("%B %d, %Y")
            c.subscriber_name = subscriber_name
            return c

        doc.build(flowables, canvasmaker=canvas_maker)

        pdf_bytes = buffer.getvalue()
        buffer.close()

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            logger.info("Saved visual PDF briefing (%d bytes) to: %s", len(pdf_bytes), output_path)

        return pdf_bytes

