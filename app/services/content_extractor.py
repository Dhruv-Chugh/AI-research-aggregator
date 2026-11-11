"""Article Content Extractor Service (OOP & Pydantic).

Extracts and serializes full article text, structure, and metadata directly from the
original websites of OpenAI (openai.com) and Anthropic (anthropic.com).
Converts web content into clean, publication-ready GitHub-Flavored Markdown.
"""

from __future__ import annotations

import html
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import bs4
from bs4 import BeautifulSoup, NavigableString, Tag
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("ai_news.content_extractor")


class ExtractedContent(BaseModel):
    """Pydantic model representing extracted article content from a web page."""

    url: str = Field(..., description="Canonical or fetched URL of the article")
    title: Optional[str] = Field(default=None, description="Title extracted from the web page")
    author: Optional[str] = Field(default=None, description="Author or team listed on page")
    published_date_str: Optional[str] = Field(default=None, description="Raw date string from page")
    markdown: str = Field(default="", description="Clean GitHub-Flavored Markdown of the article body")
    text: str = Field(default="", description="Plain text content of article body")
    word_count: int = Field(default=0, description="Word count of extracted body text")
    success: bool = Field(default=True, description="Whether extraction succeeded")
    error_message: Optional[str] = Field(default=None, description="Error details if extraction failed")


class ArticleContentExtractor:
    """Service to fetch and extract full article content from OpenAI and Anthropic websites."""

    DEFAULT_HEADERS: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        timeout: float = 15.0,
        enable_reader_fallback: Optional[bool] = None,
    ) -> None:
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=timeout,
            headers=self.DEFAULT_HEADERS,
            follow_redirects=True,
        )
        if enable_reader_fallback is not None:
            self.enable_reader_fallback = enable_reader_fallback
        else:
            self.enable_reader_fallback = os.getenv("ENABLE_READER_FALLBACK", "true").lower() in ("true", "1", "yes")

    def __enter__(self) -> ArticleContentExtractor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close HTTP client if owned by this extractor."""
        if self._owns_client:
            self.client.close()

    def _fetch_via_reader(self, url: str) -> ExtractedContent:
        """Fallback to Jina Reader Markdown extraction when cloud IPs hit HTTP 403/429."""
        reader_url = f"https://r.jina.ai/{url}"
        try:
            resp = self.client.get(reader_url, headers={"Accept": "text/markdown"})
            if resp.status_code != 200:
                return ExtractedContent(
                    url=url,
                    success=False,
                    error_message=f"Reader fallback HTTP {resp.status_code}",
                )

            raw_text = (resp.text or "").strip()
            title: Optional[str] = None
            markdown_body = raw_text

            if "Markdown Content:" in raw_text:
                header_part, body_part = raw_text.split("Markdown Content:", 1)
                markdown_body = body_part.strip()
                for line in header_part.splitlines():
                    if line.startswith("Title:"):
                        title = line.removeprefix("Title:").strip() or None
                        break

            plain_text = " ".join(markdown_body.split())
            words = len(plain_text.split())
            return ExtractedContent(
                url=url,
                title=title,
                markdown=markdown_body,
                text=plain_text,
                word_count=words,
                success=bool(words > 0),
                error_message=None if words > 0 else "Reader fallback returned empty content",
            )
        except Exception as e:
            logger.warning("Reader fallback failed for %s: %s", url, e)
            return ExtractedContent(url=url, success=False, error_message=str(e))

    def fetch_content(self, url: str) -> ExtractedContent:
        """Fetch article web page and parse full content according to the domain."""
        logger.debug("Fetching article content from: %s", url)

        try:
            response = self.client.get(url)
            if response.status_code != 200:
                if response.status_code in (403, 429) and self.enable_reader_fallback:
                    logger.info(
                        "HTTP %d for %s; attempting reader fallback extraction...",
                        response.status_code,
                        url,
                    )
                    fallback = self._fetch_via_reader(url)
                    if fallback.success:
                        return fallback
                logger.warning("HTTP %d when fetching article from %s", response.status_code, url)
                return ExtractedContent(
                    url=url,
                    success=False,
                    error_message=f"HTTP {response.status_code}: Unable to load page",
                )

            html_text = response.text
            final_url = str(response.url)
            domain = urlparse(final_url).netloc.lower()

            if "anthropic.com" in domain:
                return self.extract_anthropic(html_text, final_url)
            elif "openai.com" in domain:
                return self.extract_openai(html_text, final_url)
            else:
                return self.extract_generic(html_text, final_url)

        except Exception as e:
            logger.error("Failed to fetch or extract content from %s: %s", url, e)
            return ExtractedContent(
                url=url,
                success=False,
                error_message=str(e),
            )

    def extract_anthropic(self, html_content: str, url: str) -> ExtractedContent:
        """Extract article content from an Anthropic webpage."""
        soup = BeautifulSoup(html_content, "lxml")

        # Locate main article tag
        article_tag = soup.find("article") or soup.find("main")
        if not article_tag:
            return ExtractedContent(
                url=url,
                success=False,
                error_message="Could not locate <article> or <main> container in Anthropic HTML",
            )

        # Clone tag to avoid mutating
        article_copy = BeautifulSoup(str(article_tag), "lxml")

        # Extract title
        title_el = article_copy.find(["h1", ".VYxBBq_Title"])
        title = title_el.get_text(strip=True) if title_el else None

        # Extract author & date from metadata header if present
        date_el = article_copy.find("time")
        published_date_str = date_el.get_text(strip=True) if date_el else None

        # Remove tags
        for tag_name in ["script", "style", "nav", "svg", "button", "header", "footer"]:
            for unwanted in article_copy.find_all(tag_name):
                unwanted.decompose()

        # Remove specific CSS classes
        for selector in [
            ".VYxBBq_Header",
            ".yApI1G_panel",
            "._74-EFW_container",
            ".EaEYjq_Footnotes",
            ".EaEYjq_FootnotesList",
        ]:
            for unwanted in article_copy.select(selector):
                unwanted.decompose()

        # Convert remaining body elements into structured Markdown
        markdown_body = self._elements_to_markdown(article_copy)
        plain_text = " ".join(markdown_body.split())
        words = len(plain_text.split())

        return ExtractedContent(
            url=url,
            title=title,
            published_date_str=published_date_str,
            markdown=markdown_body,
            text=plain_text,
            word_count=words,
            success=bool(words > 0),
        )

    def extract_openai(self, html_content: str, url: str) -> ExtractedContent:
        """Extract article content from an OpenAI webpage."""
        soup = BeautifulSoup(html_content, "lxml")

        # Locate main article tag
        article_tag = soup.find("article") or soup.find("main")
        if not article_tag:
            return ExtractedContent(
                url=url,
                success=False,
                error_message="Could not locate <article> or <main> container in OpenAI HTML",
            )

        article_copy = BeautifulSoup(str(article_tag), "lxml")

        # Extract title
        title_el = article_copy.find("h1")
        title = title_el.get_text(strip=True) if title_el else None

        # Remove scripts, styles, nav, share buttons, header redundant elements
        for unwanted in article_copy.find_all(["script", "style", "nav", "svg", "button"]):
            unwanted.decompose()

        # Convert article elements into structured Markdown
        markdown_body = self._elements_to_markdown(article_copy)
        plain_text = " ".join(markdown_body.split())
        words = len(plain_text.split())

        return ExtractedContent(
            url=url,
            title=title,
            markdown=markdown_body,
            text=plain_text,
            word_count=words,
            success=bool(words > 0),
        )

    def extract_generic(self, html_content: str, url: str) -> ExtractedContent:
        """Fallback extractor for generic HTML articles."""
        soup = BeautifulSoup(html_content, "lxml")
        container = soup.find("article") or soup.find("main") or soup.find("body")

        if not container:
            return ExtractedContent(url=url, success=False, error_message="No body container found")

        for unwanted in container.find_all(["script", "style", "nav", "header", "footer", "svg"]):
            unwanted.decompose()

        markdown_body = self._elements_to_markdown(container)
        plain_text = " ".join(markdown_body.split())
        words = len(plain_text.split())

        return ExtractedContent(
            url=url,
            markdown=markdown_body,
            text=plain_text,
            word_count=words,
            success=bool(words > 0),
        )

    def _elements_to_markdown(self, root: Tag) -> str:
        """Walk element tree and serialize headings, paragraphs, lists, blockquotes, and code to Markdown."""
        lines: List[str] = []

        # Find direct block-level children or semantic elements
        blocks = root.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "blockquote", "pre"],
            recursive=True,
        )

        seen_blocks = set()

        for el in blocks:
            # Skip if inside pre or blockquote that was already handled
            if any(parent in seen_blocks for parent in el.parents):
                continue

            seen_blocks.add(el)

            if el.name.startswith("h"):
                level = int(el.name[1])
                heading_text = self._format_inline(el).strip()
                if heading_text:
                    lines.append(f"\n{'#' * level} {heading_text}\n")

            elif el.name == "p":
                para_text = self._format_inline(el).strip()
                if para_text:
                    lines.append(f"{para_text}\n")

            elif el.name == "blockquote":
                quote_text = self._format_inline(el).strip()
                if quote_text:
                    quoted_lines = [f"> {line}" for line in quote_text.splitlines()]
                    lines.append("\n".join(quoted_lines) + "\n")

            elif el.name in ("ul", "ol"):
                items = el.find_all("li", recursive=False)
                for idx, li in enumerate(items, 1):
                    item_text = self._format_inline(li).strip()
                    if item_text:
                        bullet = f"{idx}." if el.name == "ol" else "-"
                        lines.append(f"{bullet} {item_text}")
                lines.append("")

            elif el.name == "pre":
                code_text = el.get_text()
                lines.append(f"```\n{code_text.strip()}\n```\n")

        raw_md = "\n".join(lines).strip()
        # Consolidate excessive blank lines
        while "\n\n\n" in raw_md:
            raw_md = raw_md.replace("\n\n\n", "\n\n")

        return raw_md

    def _format_inline(self, elem: Tag | NavigableString) -> str:
        """Format inline HTML (links, bold, italics, code) into Markdown syntax."""
        if isinstance(elem, NavigableString):
            return str(elem)

        children_text = "".join(self._format_inline(child) for child in elem.children)

        name = getattr(elem, "name", None)
        if name in ("strong", "b"):
            return f"**{children_text.strip()}**" if children_text.strip() else ""
        if name in ("em", "i"):
            return f"*{children_text.strip()}*" if children_text.strip() else ""
        if name == "code" and getattr(elem.parent, "name", None) != "pre":
            return f"`{children_text.strip()}`" if children_text.strip() else ""
        if name == "a" and elem.get("href"):
            href = elem["href"]
            text = children_text.strip()
            return f"[{text}]({href})" if text else ""

        return children_text
