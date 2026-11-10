"""Unit tests for YouTube scraper service, TranscriptService, and Pydantic models."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from youtube_transcript_api import (
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

from app.models.youtube import (
    ChannelConfig,
    FilterCriteria,
    TranscriptSnippet,
    VideoTranscript,
    YouTubeVideo,
)
from app.scrapers.youtube import (
    ChannelResolver,
    FeedParserClient,
    VideoFilter,
    YouTubeService,
)
from app.services.transcript_service import (
    TranscriptCleaner,
    TranscriptIdParser,
    TranscriptService,
)


@pytest.fixture
def youtube_service():
    service = YouTubeService(channels=[])
    yield service
    service.close()


@pytest.fixture
def transcript_service():
    return TranscriptService()


# ---------------------------------------------------------------------------
# Pydantic Model Tests
# ---------------------------------------------------------------------------
def test_channel_config_validation():
    """ChannelConfig should validate that channel_id starts with UC and is 24 chars."""
    valid_channel = ChannelConfig(
        name="Think School",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        handle="@thinkschool",
    )
    assert valid_channel.name == "Think School"
    assert valid_channel.feed_url == "https://www.youtube.com/feeds/videos.xml?channel_id=UCKZozRVHRYsYHGEyNKuhhdA"

    with pytest.raises(ValueError):
        ChannelConfig(name="Invalid", channel_id="INVALID_ID")


def test_youtube_video_model():
    """YouTubeVideo model properties (is_short, has_transcript) should work cleanly."""
    now = datetime.now(timezone.utc)
    short_video = YouTubeVideo(
        video_id="short_12345",
        title="Short Video",
        url="https://www.youtube.com/shorts/short_12345",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
    )
    assert short_video.is_short is True
    assert short_video.has_transcript is False

    long_video = YouTubeVideo(
        video_id="long_123456",
        title="Long Episode",
        url="https://www.youtube.com/watch?v=long_123456",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
    )
    assert long_video.is_short is False


# ---------------------------------------------------------------------------
# ChannelResolver Tests
# ---------------------------------------------------------------------------
def test_resolve_channel_id_direct(youtube_service):
    """Direct 24-character channel IDs should be recognized instantly."""
    cid = "UCKZozRVHRYsYHGEyNKuhhdA"
    assert youtube_service.resolve_channel_id(cid) == cid


def test_resolve_channel_id_from_url(youtube_service):
    """Channel IDs in full URL paths should be extracted via regex."""
    url = "https://www.youtube.com/channel/UCKZozRVHRYsYHGEyNKuhhdA"
    assert youtube_service.resolve_channel_id(url) == "UCKZozRVHRYsYHGEyNKuhhdA"


def test_resolve_channel_id_from_handle(youtube_service):
    """Handles should be resolved via page inspection."""
    mock_html = """
    <html>
        <head>
            <link rel="canonical" href="https://www.youtube.com/channel/UCKZozRVHRYsYHGEyNKuhhdA">
        </head>
    </html>
    """
    with patch.object(youtube_service.client, "get") as mock_get:
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        resolved = youtube_service.resolve_channel_id("@CustomCreator")
        assert resolved == "UCKZozRVHRYsYHGEyNKuhhdA"
        # Second lookup should use cache
        assert youtube_service.resolve_channel_id("@CustomCreator") == "UCKZozRVHRYsYHGEyNKuhhdA"
        assert mock_get.call_count == 1


def test_add_channel_dict(youtube_service):
    """Channels can be easily added as simple name + id dictionaries or ChannelConfig."""
    config = youtube_service.add_channel({"name": "Custom Channel", "id": "UCKZozRVHRYsYHGEyNKuhhdA"})
    assert isinstance(config, ChannelConfig)
    assert len(youtube_service.channels) == 1
    assert youtube_service.channels[0].name == "Custom Channel"
    assert youtube_service.channels[0].channel_id == "UCKZozRVHRYsYHGEyNKuhhdA"


# ---------------------------------------------------------------------------
# VideoFilter & FilterCriteria Tests
# ---------------------------------------------------------------------------
def test_filter_by_time_24_hours(youtube_service):
    """Videos older than 24 hours should be excluded by VideoFilter."""
    now = datetime.now(timezone.utc)
    v_recent = YouTubeVideo(
        video_id="recent_1",
        title="Recent AI News",
        url="https://youtube.com/watch?v=recent_1",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now - timedelta(hours=3),
    )
    v_borderline = YouTubeVideo(
        video_id="borderline_2",
        title="Just under 24h",
        url="https://youtube.com/watch?v=borderline_2",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now - timedelta(hours=23, minutes=30),
    )
    v_old = YouTubeVideo(
        video_id="old_3",
        title="Old AI News",
        url="https://youtube.com/watch?v=old_3",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now - timedelta(hours=25),
    )

    all_videos = [v_recent, v_borderline, v_old]
    criteria = FilterCriteria(hours_back=24)
    filtered = VideoFilter.apply(all_videos, criteria)

    assert len(filtered) == 2
    assert [v.video_id for v in filtered] == ["recent_1", "borderline_2"]


def test_filter_exclude_shorts():
    """FilterCriteria(include_shorts=False) should filter out Shorts."""
    now = datetime.now(timezone.utc)
    v_short = YouTubeVideo(
        video_id="short_1",
        title="Short",
        url="https://youtube.com/shorts/short_1",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
    )
    v_long = YouTubeVideo(
        video_id="long_1",
        title="Long Video",
        url="https://youtube.com/watch?v=long_1",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
    )
    criteria = FilterCriteria(hours_back=24, include_shorts=False)
    filtered = VideoFilter.apply([v_short, v_long], criteria)
    assert len(filtered) == 1
    assert filtered[0].video_id == "long_1"


# ---------------------------------------------------------------------------
# TranscriptService & Parser Tests
# ---------------------------------------------------------------------------
def test_transcript_id_parser():
    """TranscriptIdParser should extract 11-char video IDs from various formats."""
    parser = TranscriptIdParser()
    assert parser.extract_video_id("DHzC4VVmolk") == "DHzC4VVmolk"
    assert parser.extract_video_id("https://www.youtube.com/watch?v=DHzC4VVmolk") == "DHzC4VVmolk"
    assert parser.extract_video_id("https://youtu.be/DHzC4VVmolk") == "DHzC4VVmolk"
    assert parser.extract_video_id("https://www.youtube.com/shorts/po0sAV_QCNM") == "po0sAV_QCNM"


def test_transcript_cleaner():
    """TranscriptCleaner should combine snippets and produce TranscriptSnippet models."""
    mock_s1 = MagicMock(text=" Healthcare costs are rising.", start=0.5, duration=2.1)
    mock_s2 = MagicMock(text=" Wall Street entered the market. ", start=2.6, duration=3.0)

    clean_text, pydantic_snippets = TranscriptCleaner.clean([mock_s1, mock_s2])
    assert clean_text == "Healthcare costs are rising. Wall Street entered the market."
    assert len(pydantic_snippets) == 2
    assert isinstance(pydantic_snippets[0], TranscriptSnippet)
    assert pydantic_snippets[0].start == 0.5


def test_transcript_service_success(transcript_service):
    """Test successful conversion of video speech into clean text."""
    mock_snippet_1 = MagicMock(text="Welcome to Think School.", start=0.0, duration=2.0)
    mock_snippet_2 = MagicMock(text="Today we analyze the healthcare market.", start=2.0, duration=3.0)

    mock_fetched = MagicMock()
    mock_fetched.snippets = [mock_snippet_1, mock_snippet_2]
    mock_fetched.language = "English"
    mock_fetched.language_code = "en"
    mock_fetched.is_generated = False

    with patch.object(transcript_service.api, "fetch", return_value=mock_fetched):
        result = transcript_service.get_transcript("DHzC4VVmolk")
        assert isinstance(result, VideoTranscript)
        assert result.available is True
        assert result.word_count == 10
        assert "healthcare market" in result.text
        assert len(result.snippets) == 2
        assert result.error_reason is None


def test_transcript_service_disabled(transcript_service):
    """When captions are disabled, it should report available=False without crashing."""
    with patch.object(
        transcript_service.api,
        "fetch",
        side_effect=TranscriptsDisabled("test_vid_12"),
    ):
        result = transcript_service.get_transcript("test_vid_12")
        assert isinstance(result, VideoTranscript)
        assert result.available is False
        assert "disabled" in result.error_reason.lower()


def test_transcript_service_ip_blocked(transcript_service):
    """When IP is rate limited by YouTube, report friendly error."""
    with patch.object(
        transcript_service.api,
        "fetch",
        side_effect=IpBlocked("test_vid_12"),
    ):
        result = transcript_service.get_transcript("test_vid_12")
        assert isinstance(result, VideoTranscript)
        assert result.available is False
        assert "rate-limited" in result.error_reason.lower() or "ipblocked" in result.error_reason.lower()


def test_service_enrich_video_transcript(youtube_service):
    """Calling youtube_service.enrich_video_with_transcript delegates to TranscriptService."""
    now = datetime.now(timezone.utc)
    video = YouTubeVideo(
        video_id="vid_test_99",
        title="Test Integration Video",
        url="https://youtube.com/watch?v=vid_test_99",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
    )

    mock_transcript = VideoTranscript(
        video_id="vid_test_99",
        available=True,
        text="This is test knowledge text.",
        word_count=5,
    )

    with patch.object(youtube_service.transcript_service, "get_transcript", return_value=mock_transcript):
        enriched = youtube_service.enrich_video_with_transcript(video)
        assert enriched.has_transcript is True
        assert enriched.transcript.text == "This is test knowledge text."


def test_youtube_shorts_detection_heuristics():
    """YouTubeVideo auto-detects shorts from #shorts hashtag in title or description."""
    now = datetime.now(timezone.utc)

    # 1. Title hashtag
    v1 = YouTubeVideo(
        video_id="short_hash_1",
        title="Quick AI Tip #shorts",
        url="https://www.youtube.com/watch?v=short_hash_1",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
    )
    assert v1.is_short is True

    # 2. Description hashtag
    v2 = YouTubeVideo(
        video_id="short_hash_2",
        title="Quick AI Tip",
        url="https://www.youtube.com/watch?v=short_hash_2",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
        description="Don't forget to like! #Shorts",
    )
    assert v2.is_short is True

    # 3. Regular video
    v3 = YouTubeVideo(
        video_id="long_regular_3",
        title="Full In-Depth Analysis of Semiconductor Industry",
        url="https://www.youtube.com/watch?v=long_regular_3",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        channel_name="Think School",
        published_at=now,
        description="A 45-minute breakdown of chips.",
    )
    assert v3.is_short is False


def test_feed_parser_probe_shorts(youtube_service):
    """FeedParserClient probes /shorts/ endpoint and accurately identifies Shorts."""
    client = youtube_service.feed_parser
    client.check_shorts = True

    # Mock HEAD response for short: status 200
    mock_resp_short = MagicMock(status_code=200)
    with patch.object(client.client, "head", return_value=mock_resp_short):
        is_short = client.is_short_video("short_probe_id")
        assert is_short is True

    # Clear cache and mock HEAD response for normal video: status 303 redirect
    client._shorts_cache.clear()
    mock_resp_long = MagicMock(status_code=303)
    with patch.object(client.client, "head", return_value=mock_resp_long):
        is_long = client.is_short_video("long_probe_id")
        assert is_long is False
