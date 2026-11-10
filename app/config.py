"""Configuration and Source Management for AI News Aggregator.

Loads, validates, and manages source configurations (monitored YouTube channels,
OpenAI feeds, Anthropic feeds, and default pipeline parameters).
Provides dead-simple channel addition via handles, URLs, or channel IDs.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import feedparser
import httpx
from pydantic import BaseModel, Field, field_validator

from app.models.youtube import ChannelConfig

logger = logging.getLogger("ai_news.config")

DEFAULT_SOURCES_PATH = Path("config/sources.json")
DEFAULT_LOOKBACK_HOURS = 24


def resolve_channel_info(
    identifier: str,
    name: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> ChannelConfig:
    """Resolve any YouTube handle, URL, or Channel ID into a complete, validated ChannelConfig.

    Examples of accepted identifiers:
        - '@thinkschool'
        - 'https://www.youtube.com/@thinkschool'
        - 'https://www.youtube.com/channel/UCKZozRVHRYsYHGEyNKuhhdA'
        - 'UCKZozRVHRYsYHGEyNKuhhdA'
    """
    from app.scrapers.youtube import ChannelResolver

    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=15.0,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
        follow_redirects=True,
    )

    try:
        raw = identifier.strip()
        handle: Optional[str] = None

        if raw.startswith("@"):
            handle = raw
        elif "youtube.com/@" in raw:
            handle = "@" + raw.split("youtube.com/@")[1].split("/")[0].split("?")[0]

        resolver = ChannelResolver(http_client)
        cid = resolver.resolve(raw)

        # Detect channel title from RSS feed if not provided
        channel_name = name
        if not channel_name:
            feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
            try:
                resp = http_client.get(feed_url)
                if resp.status_code == 200:
                    parsed = feedparser.parse(resp.content)
                    channel_name = parsed.feed.get("title")
            except Exception:
                pass

        if not channel_name:
            channel_name = handle or f"Channel {cid}"

        channel_url = (
            f"https://www.youtube.com/{handle}"
            if handle
            else f"https://www.youtube.com/channel/{cid}"
        )

        return ChannelConfig(
            name=channel_name.strip(),
            channel_id=cid,
            handle=handle,
            url=channel_url,
            is_active=True,
        )
    finally:
        if owns_client:
            http_client.close()


class OpenAISourceConfig(BaseModel):
    """Configuration for OpenAI news & research scraping."""

    name: str = Field(default="OpenAI News & Research", description="Source name")
    rss_url: str = Field(
        default="https://openai.com/news/rss.xml",
        description="Official OpenAI RSS feed URL",
    )
    is_active: bool = Field(default=True, description="Whether OpenAI scraping is active")


class AnthropicSourceConfig(BaseModel):
    """Configuration for Anthropic news, engineering & research scraping."""

    name: str = Field(
        default="Anthropic News, Engineering & Research",
        description="Source name",
    )
    feeds: Dict[str, str] = Field(
        default_factory=lambda: {
            "news": "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml",
            "engineering": "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_engineering.xml",
            "research": "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_research.xml",
        },
        description="Named RSS feed URLs for Anthropic",
    )
    is_active: bool = Field(default=True, description="Whether Anthropic scraping is active")


class SourcesConfig(BaseModel):
    """Master configuration containing all monitored sources."""

    youtube_channels: List[ChannelConfig] = Field(
        default_factory=list,
        description="List of tracked YouTube channels",
    )
    openai: OpenAISourceConfig = Field(
        default_factory=OpenAISourceConfig,
        description="OpenAI source configuration",
    )
    anthropic: AnthropicSourceConfig = Field(
        default_factory=AnthropicSourceConfig,
        description="Anthropic source configuration",
    )

    @classmethod
    def get_default(cls) -> SourcesConfig:
        """Create default configuration with no pre-configured YouTube channels.

        Users add their own channels via the web UI or API after setup.
        OpenAI and Anthropic RSS feeds are always enabled by default.
        """
        return cls(youtube_channels=[])

    @classmethod
    def load_from_file(cls, path: Union[str, Path] = DEFAULT_SOURCES_PATH) -> SourcesConfig:
        """Load sources configuration from JSON file, falling back to defaults if not found."""
        p = Path(path)
        if not p.exists():
            logger.info("Configuration file %s not found. Creating with default sources.", p)
            default_config = cls.get_default()
            default_config.save_to_file(p)
            return default_config

        try:
            content = p.read_text(encoding="utf-8")
            data = json.loads(content)

            # Normalize youtube_channels if items are raw strings or dicts
            raw_channels = data.get("youtube_channels", [])
            normalized_channels = []
            for item in raw_channels:
                if isinstance(item, str):
                    try:
                        normalized_channels.append(resolve_channel_info(item))
                    except Exception:
                        pass
                elif isinstance(item, dict):
                    normalized_channels.append(item)

            data["youtube_channels"] = normalized_channels
            return cls.model_validate(data)
        except Exception as e:
            logger.warning("Error reading config from %s: %s. Using default sources.", p, e)
            return cls.get_default()

    def save_to_file(self, path: Union[str, Path] = DEFAULT_SOURCES_PATH) -> Path:
        """Save current configuration to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved sources configuration to %s", p.resolve())
        return p.resolve()

    def get_active_youtube_channels(self) -> List[ChannelConfig]:
        """Return only active monitored YouTube channels."""
        return [c for c in self.youtube_channels if c.is_active]

    def add_youtube_channel(
        self,
        channel_or_identifier: Union[ChannelConfig, str, Dict[str, Any]],
        name: Optional[str] = None,
    ) -> bool:
        """Add a YouTube channel using a ChannelConfig, handle string ('@creator'), URL, or dict."""
        if isinstance(channel_or_identifier, ChannelConfig):
            channel = channel_or_identifier
        elif isinstance(channel_or_identifier, str):
            channel = resolve_channel_info(channel_or_identifier, name=name)
        elif isinstance(channel_or_identifier, dict):
            cid = channel_or_identifier.get("channel_id")
            if not cid:
                ident = channel_or_identifier.get("handle") or channel_or_identifier.get("url")
                if ident:
                    channel = resolve_channel_info(ident, name=channel_or_identifier.get("name"))
                else:
                    raise ValueError("Channel dict must contain 'channel_id', 'handle', or 'url'")
            else:
                channel = ChannelConfig(**channel_or_identifier)
        else:
            raise TypeError("channel_or_identifier must be ChannelConfig, str, or dict")

        for existing in self.youtube_channels:
            if existing.channel_id == channel.channel_id:
                logger.info("Channel %s (%s) is already in configuration.", channel.name, channel.channel_id)
                return False

        self.youtube_channels.append(channel)
        return True

    def remove_youtube_channel(self, channel_id_or_handle: str) -> bool:
        """Remove a YouTube channel by ID or handle."""
        target = channel_id_or_handle.strip().lower()
        initial_len = len(self.youtube_channels)
        self.youtube_channels = [
            c
            for c in self.youtube_channels
            if c.channel_id.lower() != target and (not c.handle or c.handle.lower() != target)
        ]
        return len(self.youtube_channels) < initial_len


def get_sources_config(path: Union[str, Path] = DEFAULT_SOURCES_PATH) -> SourcesConfig:
    """Convenience accessor to load sources config."""
    return SourcesConfig.load_from_file(path)


def add_channel(
    identifier: str,
    name: Optional[str] = None,
    config_path: Union[str, Path] = DEFAULT_SOURCES_PATH,
) -> ChannelConfig:
    """Dead-simple helper: Add a channel by handle, URL, or ID and save directly to config/sources.json.

    Usage:
        add_channel("@TwoMinutePapers")
        add_channel("https://youtube.com/@mkbhd")
    """
    cfg = SourcesConfig.load_from_file(config_path)
    channel = resolve_channel_info(identifier, name=name)
    if cfg.add_youtube_channel(channel):
        cfg.save_to_file(config_path)
        logger.info("Added channel '%s' (%s) to %s", channel.name, channel.channel_id, config_path)
    return channel
