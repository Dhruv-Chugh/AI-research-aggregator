"""YouTube RSS Discovery & Scraper Service (OOP & Pydantic Architecture).

Object-oriented service to discover and filter YouTube videos via native RSS feeds.
Employs Pydantic schemas for strict validation and explicit type checking.
Includes automated YouTube Shorts detection to exclude short-form content.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import dateutil.parser
import feedparser
import httpx

from app.models.youtube import (
    ChannelConfig,
    FilterCriteria,
    VideoTranscript,
    YouTubeVideo,
)
from app.services.transcript_service import TranscriptService

logger = logging.getLogger("ai_news.youtube")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Default Curated Channels using Pydantic ChannelConfig models
# ---------------------------------------------------------------------------
DEFAULT_CHANNELS: List[ChannelConfig] = [
    ChannelConfig(
        name="Think School",
        channel_id="UCKZozRVHRYsYHGEyNKuhhdA",
        handle="@thinkschool",
    ),
    ChannelConfig(
        name="Figuring Out by Raj Shamani",
        channel_id="UCzwCEE_PchiBULMnAJqhGVg",
        handle="@rajshamani",
    ),
]


class ChannelResolver:
    """Class responsible for resolving channel identifiers (handle, URL, ID) to a 24-char Channel ID."""

    def __init__(self, client: httpx.Client) -> None:
        self.client = client
        self._cache: Dict[str, str] = {}

    def register_cache(self, identifier: str, channel_id: str) -> None:
        """Register a known mapping into the resolver cache."""
        self._cache[identifier.strip().lower()] = channel_id

    def resolve(self, identifier: str) -> str:
        """Resolve a raw channel string (ID, @handle, or URL) into a validated 24-character Channel ID."""
        cleaned = identifier.strip()

        # 1. In-memory cache check
        if cleaned.lower() in self._cache:
            return self._cache[cleaned.lower()]

        # 2. Direct 24-char channel ID starting with UC
        if re.fullmatch(r"UC[a-zA-Z0-9_-]{22}", cleaned):
            self._cache[cleaned.lower()] = cleaned
            return cleaned

        # 3. Channel URL format (youtube.com/channel/UC...)
        match = re.search(r"/channel/(UC[a-zA-Z0-9_-]{22})", cleaned)
        if match:
            cid = match.group(1)
            self._cache[cleaned.lower()] = cid
            return cid

        # 4. Handle format (@creator or URL with @creator)
        handle = cleaned if cleaned.startswith("@") or cleaned.startswith("http") else f"@{cleaned}"
        url = handle if handle.startswith("http") else f"https://www.youtube.com/{handle}"

        try:
            logger.info("Resolving channel handle via webpage: %s", url)
            response = self.client.get(url)
            response.raise_for_status()

            patterns = [
                r'<link rel="canonical" href="https://www.youtube.com/channel/(UC[a-zA-Z0-9_-]{22})"',
                r'"externalId":"(UC[a-zA-Z0-9_-]{22})"',
                r'itemprop="channelId" content="(UC[a-zA-Z0-9_-]{22})"',
                r'"browseId":"(UC[a-zA-Z0-9_-]{22})"',
            ]
            for pat in patterns:
                found = re.search(pat, response.text)
                if found:
                    cid = found.group(1)
                    self._cache[cleaned.lower()] = cid
                    logger.info("Resolved %s -> %s", cleaned, cid)
                    return cid

        except Exception as e:
            logger.warning("Failed to resolve channel handle '%s': %s", identifier, e)

        raise ValueError(f"Could not resolve a valid YouTube Channel ID from: '{identifier}'")


class FeedParserClient:
    """Class responsible for fetching raw YouTube RSS XML and parsing into Pydantic models."""

    def __init__(self, client: httpx.Client, check_shorts: bool = False) -> None:
        self.client = client
        self.check_shorts = check_shorts
        self._shorts_cache: Dict[str, bool] = {}

    def is_short_video(
        self,
        video_id: str,
        title: str = "",
        description: str = "",
        url: str = "",
    ) -> bool:
        """Accurately determine whether a video is a YouTube Short using fast, bot-safe heuristics."""
        if video_id in self._shorts_cache:
            return self._shorts_cache[video_id]

        # 1. Fast heuristics: explicit /shorts/ URL or hashtags in title/description
        if "/shorts/" in url:
            self._shorts_cache[video_id] = True
            return True

        lower_t = title.lower() if title else ""
        lower_d = description.lower() if description else ""
        short_tags = ["#shorts", "#short", "#ytshorts", "#youtubeshorts", "#shortsvideo"]
        if any(tag in lower_t or tag in lower_d for tag in short_tags):
            self._shorts_cache[video_id] = True
            return True

        # 2. Live HTTP HEAD redirect probe on https://www.youtube.com/shorts/{video_id}
        if self.check_shorts:
            try:
                head_resp = self.client.head(
                    f"https://www.youtube.com/shorts/{video_id}",
                    follow_redirects=False,
                )
                # YouTube Shorts return HTTP 200 without redirecting to /watch?v=
                if head_resp.status_code == 200:
                    self._shorts_cache[video_id] = True
                    return True
                # Regular videos return HTTP 303/302 redirecting to /watch?v=
                if head_resp.status_code in (301, 302, 303, 307, 308):
                    self._shorts_cache[video_id] = False
                    return False
            except Exception as e:
                logger.debug("Shorts head probe failed for %s: %s", video_id, e)

        self._shorts_cache[video_id] = False
        return False

    def fetch_feed(self, channel: ChannelConfig) -> List[YouTubeVideo]:
        """Fetch RSS feed for a channel and return a list of validated YouTubeVideo models."""
        feed_url = channel.feed_url
        logger.debug("Fetching feed: %s (%s)", channel.name, feed_url)

        try:
            # Use httpx with modern browser User-Agent
            response = self.client.get(feed_url)
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
            else:
                logger.warning("HTTP %d fetching feed for %s", response.status_code, channel.name)
                feed = feedparser.parse(feed_url)
        except Exception as e:
            logger.error("Error reading RSS feed for %s: %s", channel.name, e)
            return []

        if feed.bozo and not feed.entries:
            logger.warning("Feedparser error for channel %s: %s", channel.name, feed.get("bozo_exception"))
            return []

        detected_channel_title = feed.feed.get("title") or channel.name
        videos: List[YouTubeVideo] = []

        for entry in feed.entries:
            try:
                # Video ID
                vid = getattr(entry, "yt_videoid", None)
                if not vid:
                    entry_id = entry.get("id", "")
                    if "yt:video:" in entry_id:
                        vid = entry_id.split("yt:video:")[-1]
                    elif "watch?v=" in entry.get("link", ""):
                        vid = entry.link.split("watch?v=")[-1].split("&")[0]

                if not vid:
                    continue

                title = entry.get("title", "Untitled Video").strip()
                link = entry.get("link", f"https://www.youtube.com/watch?v={vid}")

                # Detect whether this item is a YouTube Short
                is_short = self.is_short_video(vid, title=title, description=entry.get("summary", ""), url=link)
                final_url = f"https://www.youtube.com/shorts/{vid}" if is_short else link

                # Publication Timestamp
                pub_raw = entry.get("published") or entry.get("updated")
                if pub_raw:
                    pub_dt = dateutil.parser.parse(pub_raw)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    else:
                        pub_dt = pub_dt.astimezone(timezone.utc)
                else:
                    pub_dt = datetime.now(timezone.utc)

                # Description
                description = entry.get("summary") or entry.get("media_description") or ""

                # Thumbnail
                thumb = None
                if "media_thumbnail" in entry and len(entry.media_thumbnail) > 0:
                    thumb = entry.media_thumbnail[0].get("url")

                video_model = YouTubeVideo(
                    video_id=vid,
                    title=title,
                    url=final_url,
                    channel_id=channel.channel_id,
                    channel_name=detected_channel_title,
                    published_at=pub_dt,
                    description=description,
                    thumbnail_url=thumb,
                    is_short=is_short,
                )
                videos.append(video_model)

            except Exception as e:
                logger.warning("Error constructing YouTubeVideo model: %s", e)
                continue

        return videos


class VideoFilter:
    """Class responsible for filtering lists of YouTubeVideo models using FilterCriteria."""

    @staticmethod
    def apply(videos: List[YouTubeVideo], criteria: FilterCriteria) -> List[YouTubeVideo]:
        """Apply time, short, and quantity filters to videos."""
        filtered = videos

        # 1. Filter out shorts if requested (defaults to False / exclude shorts)
        if not criteria.include_shorts:
            filtered = [v for v in filtered if not v.is_short]

        # 2. Time cutoff filter
        cutoff = criteria.cutoff_datetime
        if cutoff is not None:
            filtered = [v for v in filtered if v.published_at >= cutoff]

        # 3. Quantity limit
        if criteria.max_results_per_channel is not None and criteria.max_results_per_channel > 0:
            filtered = filtered[: criteria.max_results_per_channel]

        return filtered


class YouTubeService:
    """High-Level Object-Oriented Facade coordinating channel resolution, feeds, and filters."""

    def __init__(
        self,
        channels: Optional[List[Union[ChannelConfig, Dict[str, Any], str]]] = None,
        timeout: float = 15.0,
        proxy_url: Optional[str] = None,
        check_shorts: bool = False,
    ) -> None:
        """Initialize the YouTubeService.

        Args:
            channels: Optional list of ChannelConfig models, dicts, or channel strings.
            timeout: HTTP request timeout in seconds.
            proxy_url: Optional proxy URL for network calls.
            check_shorts: Whether to probe YouTube /shorts/ endpoint for accurate Shorts detection.
        """
        self.timeout = timeout
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        self.resolver = ChannelResolver(self.client)
        self.feed_parser = FeedParserClient(self.client, check_shorts=check_shorts)
        self.filter = VideoFilter()
        self.transcript_service = TranscriptService(proxy_url=proxy_url)

        self._channels: List[ChannelConfig] = []

        # Pre-seed cache with default channels
        for ch in DEFAULT_CHANNELS:
            self.resolver.register_cache(ch.channel_id, ch.channel_id)
            if ch.handle:
                self.resolver.register_cache(ch.handle, ch.channel_id)

        if channels is not None:
            for item in channels:
                self.add_channel(item)
        else:
            self._channels = list(DEFAULT_CHANNELS)

    @property
    def channels(self) -> List[ChannelConfig]:
        """Return tracked ChannelConfig models."""
        return self._channels

    def close(self) -> None:
        """Close underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> "YouTubeService":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def add_channel(self, channel_input: Union[ChannelConfig, Dict[str, Any], str]) -> ChannelConfig:
        """Add a channel using a ChannelConfig model, dictionary, or string identifier."""
        if isinstance(channel_input, ChannelConfig):
            config = channel_input
        elif isinstance(channel_input, dict):
            raw_id = channel_input.get("id") or channel_input.get("channel_id") or channel_input.get("handle")
            if not raw_id:
                raise ValueError("Channel dict must contain 'id', 'channel_id', or 'handle'")
            cid = self.resolver.resolve(raw_id)
            name = channel_input.get("name") or f"Channel {cid}"
            handle = channel_input.get("handle")
            config = ChannelConfig(name=name, channel_id=cid, handle=handle)
        elif isinstance(channel_input, str):
            cid = self.resolver.resolve(channel_input)
            handle = channel_input if channel_input.startswith("@") else None
            config = ChannelConfig(name=f"Channel {cid}", channel_id=cid, handle=handle)
        else:
            raise TypeError("channel_input must be a ChannelConfig, dict, or string")

        # Avoid duplicates
        if not any(c.channel_id == config.channel_id for c in self._channels):
            self._channels.append(config)

        return config

    def resolve_channel_id(self, identifier: str) -> str:
        """Delegate channel ID resolution to ChannelResolver."""
        return self.resolver.resolve(identifier)

    def fetch_channel_videos(self, channel_id: str, channel_name: Optional[str] = None) -> List[YouTubeVideo]:
        """Fetch all recent videos for a channel by ID."""
        cid = self.resolver.resolve(channel_id)
        name = channel_name or f"Channel {cid}"
        config = ChannelConfig(name=name, channel_id=cid)
        return self.feed_parser.fetch_feed(config)

    def get_latest_video(self, channel_id_or_handle: str) -> Optional[YouTubeVideo]:
        """Fetch the single latest video published by a channel."""
        cid = self.resolver.resolve(channel_id_or_handle)
        videos = self.fetch_channel_videos(cid)
        return videos[0] if videos else None

    def filter_by_time(
        self,
        videos: List[YouTubeVideo],
        hours_back: Optional[int] = 24,
    ) -> List[YouTubeVideo]:
        """Convenience method to filter videos by hours back."""
        criteria = FilterCriteria(hours_back=hours_back, include_shorts=False)
        return self.filter.apply(videos, criteria)

    def get_recent_videos(
        self,
        criteria: Optional[FilterCriteria] = None,
        hours_back: Optional[int] = 24,
        fetch_transcripts: bool = False,
        include_shorts: bool = False,
    ) -> List[YouTubeVideo]:
        """Retrieve recent videos across all tracked channels applying criteria (Shorts excluded by default)."""
        filter_criteria = criteria or FilterCriteria(hours_back=hours_back, include_shorts=include_shorts)
        all_videos: List[YouTubeVideo] = []

        for ch in self._channels:
            videos = self.feed_parser.fetch_feed(ch)
            filtered = self.filter.apply(videos, filter_criteria)

            if fetch_transcripts:
                for v in filtered:
                    self.enrich_video_with_transcript(v)

            all_videos.extend(filtered)

        # Sort newest first
        all_videos.sort(key=lambda v: v.published_at, reverse=True)
        return all_videos

    def enrich_video_with_transcript(self, video: YouTubeVideo) -> YouTubeVideo:
        """Attach transcript knowledge using TranscriptService."""
        video.transcript = self.transcript_service.get_transcript(video.video_id)
        return video


# ---------------------------------------------------------------------------
# Simple Top-Level Helper Functions
# ---------------------------------------------------------------------------
def get_channel_latest(channel_id_or_handle: str) -> Optional[YouTubeVideo]:
    """Get the latest video for any channel identifier."""
    with YouTubeService() as service:
        return service.get_latest_video(channel_id_or_handle)


def get_daily_ai_videos(
    hours_back: int = 24,
    fetch_transcripts: bool = False,
    include_shorts: bool = False,
) -> List[YouTubeVideo]:
    """Get all videos published across all configured channels in the last 24 hours (excluding Shorts by default)."""
    with YouTubeService() as service:
        criteria = FilterCriteria(hours_back=hours_back, include_shorts=include_shorts)
        return service.get_recent_videos(criteria=criteria, fetch_transcripts=fetch_transcripts)


def main() -> None:
    """CLI to inspect channels and latest videos using OOP & Pydantic."""
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="YouTube AI News Ingestion (OOP & Pydantic)")
    parser.add_argument(
        "--channel",
        "-c",
        type=str,
        help="Optional channel ID or handle to inspect",
    )
    parser.add_argument(
        "--hours",
        "-H",
        type=int,
        default=24,
        help="Hours lookback (default: 24). Set 0 for all latest entries.",
    )
    parser.add_argument(
        "--include-shorts",
        action="store_true",
        help="Include YouTube Shorts in results (default: excluded)",
    )
    parser.add_argument(
        "--transcripts",
        action="store_true",
        help="Fetch spoken transcripts for matching videos",
    )
    parser.add_argument(
        "--markdown",
        "-m",
        action="store_true",
        help="Print matching videos in Markdown format directly to stdout",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("AI NEWS - YOUTUBE FEED DISCOVERY (OOP & PYDANTIC)")
    print("=" * 70)

    channels = [args.channel] if args.channel else None

    with YouTubeService(channels=channels) as service:
        print(f"Tracking {len(service.channels)} Channel(s):")
        for ch in service.channels:
            print(f"  • {ch.name} (ID: {ch.channel_id}, Handle: {ch.handle})")
        print("-" * 70)

        criteria = FilterCriteria(
            hours_back=args.hours if args.hours > 0 else None,
            include_shorts=args.include_shorts,
        )

        videos = service.get_recent_videos(criteria=criteria, fetch_transcripts=args.transcripts)

        if args.markdown:
            for v in videos:
                print(v.to_markdown())
                print("\n---\n")
        else:
            print(f"Found {len(videos)} matching video(s):")
            print("=" * 70)

            for idx, v in enumerate(videos, 1):
                pub_str = v.published_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                video_type = "[SHORT]" if v.is_short else "[VIDEO]"
                print(f"\n[{idx}] {video_type} {v.title}")
                print(f"    Channel:   {v.channel_name} ({v.channel_id})")
                print(f"    Published: {pub_str}")
                print(f"    URL:       {v.url}")

                if v.transcript:
                    if v.transcript.available:
                        print(
                            f"    Transcript: Available ({v.transcript.word_count:,} words, "
                            f"language: {v.transcript.language})"
                        )
                        preview = v.transcript.text[:250].strip()
                        print(f"    Preview:    \"{preview}...\"")
                    else:
                        print(f"    Transcript: Unavailable ({v.transcript.error_reason})")

            print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
