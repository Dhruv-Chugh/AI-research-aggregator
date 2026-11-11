"""Unit tests for Article Content Extractor Service."""

from unittest.mock import MagicMock

import pytest

from app.services.content_extractor import (
    ArticleContentExtractor,
    ExtractedContent,
)

MOCK_ANTHROPIC_HTML = """
<!DOCTYPE html>
<html>
<head><title>Anthropic Post</title></head>
<body>
<article class="EaEYjq_Article">
  <header class="VYxBBq_Header">
    <h1 class="VYxBBq_Title">Redesigning source control at Anthropic</h1>
    <time datetime="2026-10-13">Oct 13, 2026</time>
    <button>Share</button>
  </header>
  <div class="EaEYjq_ContentWrapper">
    <p>At Anthropic, coding is no longer the bottleneck. <strong>Claude</strong> iterates on research loops quickly.</p>
    <h2>The Scaling Challenge</h2>
    <p>In May 2026, we averaged around 2,500 code changes merged per day. Check our <a href="https://example.com/repo">repo link</a>.</p>
    <ul>
      <li>First major constraint: merge queues</li>
      <li>Second major constraint: repo locks</li>
    </ul>
    <blockquote>Decentralized monorepos require novel server architectures.</blockquote>
    <pre><code>git checkout -b feature-scion</code></pre>
  </div>
  <footer class="EaEYjq_Footnotes">
    <p>Footnote 1</p>
  </footer>
</article>
</body>
</html>
"""

MOCK_OPENAI_HTML = """
<!DOCTYPE html>
<html>
<head><title>OpenAI Post</title></head>
<body>
<article>
  <h1>How we detect frontier model misalignment</h1>
  <p>Published October 29, 2026</p>
  <h2>Our internal monitoring system today</h2>
  <p>As we train increasingly capable AI models, we need reliable ways to detect when their behavior departs from intention.</p>
  <p>Here is an architectural breakdown:</p>
  <ol>
    <li>Layer 0: activation probes</li>
    <li>Layer 1: model-based reviewer</li>
  </ol>
  <pre><code>probe = load_layer_probe("layer_42")</code></pre>
</article>
</body>
</html>
"""


def test_extract_anthropic_content():
    """Verify clean extraction of headings, paragraphs, lists, and code from Anthropic HTML."""
    extractor = ArticleContentExtractor()
    extracted = extractor.extract_anthropic(MOCK_ANTHROPIC_HTML, "https://www.anthropic.com/engineering/redesigning-source-control")

    assert extracted.success is True
    assert extracted.title == "Redesigning source control at Anthropic"
    assert extracted.published_date_str == "Oct 13, 2026"
    assert extracted.word_count > 20

    md = extracted.markdown
    assert "## The Scaling Challenge" in md
    assert "**Claude** iterates on research loops quickly." in md
    assert "[repo link](https://example.com/repo)" in md
    assert "- First major constraint: merge queues" in md
    assert "> Decentralized monorepos require novel server architectures." in md
    assert "```\ngit checkout -b feature-scion\n```" in md
    # Ensure footnotes and buttons were cleaned out
    assert "Share" not in md
    assert "Footnote 1" not in md


def test_extract_openai_content():
    """Verify clean extraction of headings, paragraphs, lists, and code from OpenAI HTML."""
    extractor = ArticleContentExtractor()
    extracted = extractor.extract_openai(MOCK_OPENAI_HTML, "https://openai.com/index/misalignment")

    assert extracted.success is True
    assert extracted.title == "How we detect frontier model misalignment"
    assert extracted.word_count > 20

    md = extracted.markdown
    assert "## Our internal monitoring system today" in md
    assert "As we train increasingly capable AI models" in md
    assert "1. Layer 0: activation probes" in md
    assert "2. Layer 1: model-based reviewer" in md
    assert "```\nprobe = load_layer_probe(\"layer_42\")\n```" in md


def test_fetch_content_with_mocked_http():
    """Test fetch_content network abstraction with mocked httpx response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = MOCK_OPENAI_HTML
    mock_response.url = "https://openai.com/index/misalignment"
    mock_client.get.return_value = mock_response

    extractor = ArticleContentExtractor(client=mock_client)
    extracted = extractor.fetch_content("https://openai.com/index/misalignment")

    assert extracted.success is True
    assert extracted.title == "How we detect frontier model misalignment"
    assert "## Our internal monitoring system today" in extracted.markdown


def test_fetch_content_http_error():
    """Verify error handling on non-200 responses."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_client.get.return_value = mock_response

    extractor = ArticleContentExtractor(client=mock_client)
    extracted = extractor.fetch_content("https://openai.com/index/non-existent")

    assert extracted.success is False
    assert "HTTP 404" in extracted.error_message


def test_fetch_content_exception():
    """Verify graceful handling when network fails."""
    mock_client = MagicMock()
    mock_client.get.side_effect = Exception("Connection refused")

    extractor = ArticleContentExtractor(client=mock_client)
    extracted = extractor.fetch_content("https://openai.com/index/error")

    assert extracted.success is False
    assert "Connection refused" in extracted.error_message


def test_fetch_content_403_uses_reader_fallback():
    """Verify HTTP 403 triggers clean reader Markdown fallback when enabled."""
    mock_client = MagicMock()
    blocked_resp = MagicMock()
    blocked_resp.status_code = 403

    reader_resp = MagicMock()
    reader_resp.status_code = 200
    reader_resp.text = (
        "Title: OpenAI Law Case Study\n\n"
        "URL Source: https://openai.com/index/example/\n\n"
        "Markdown Content:\n"
        "OpenAI partners with global firms to deploy enterprise reasoning tools cleanly."
    )

    mock_client.get.side_effect = [blocked_resp, reader_resp]

    extractor = ArticleContentExtractor(client=mock_client, enable_reader_fallback=True)
    extracted = extractor.fetch_content("https://openai.com/index/example/")

    assert extracted.success is True
    assert extracted.title == "OpenAI Law Case Study"
    assert extracted.markdown.startswith("OpenAI partners with global firms")
    assert "Markdown Content:" not in extracted.markdown
    assert extracted.word_count >= 10
    assert mock_client.get.call_count == 2


