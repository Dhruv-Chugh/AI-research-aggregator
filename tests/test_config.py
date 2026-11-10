"""Unit tests for Application and Source Configuration."""

import json
from pathlib import Path

import pytest

from app.config import SourcesConfig, get_sources_config
from app.models.youtube import ChannelConfig


def test_sources_config_defaults():
    """Verify default sources configuration contains no pre-configured channels by default and OpenAI/Anthropic active."""
    cfg = SourcesConfig.get_default()
    channels = cfg.youtube_channels
    assert len(channels) == 0
    assert cfg.openai.is_active is True
    assert cfg.anthropic.is_active is True


def test_sources_config_save_and_load(tmp_path):
    """Test saving configuration to JSON and loading it back."""
    test_file = tmp_path / "test_sources.json"
    cfg = SourcesConfig.get_default()

    # Add a custom channel
    new_ch = ChannelConfig(
        name="Test Tech Channel",
        channel_id="UC1234567890123456789012",
        handle="@testtech",
        is_active=True,
    )
    assert cfg.add_youtube_channel(new_ch) is True

    # Save to disk
    cfg.save_to_file(test_file)
    assert test_file.exists()

    # Load back
    loaded_cfg = SourcesConfig.load_from_file(test_file)
    assert len(loaded_cfg.youtube_channels) == 1
    assert any(c.channel_id == "UC1234567890123456789012" for c in loaded_cfg.youtube_channels)


def test_sources_config_channel_management():
    """Test adding, duplicating, and removing channels."""
    cfg = SourcesConfig.get_default()
    initial_count = len(cfg.youtube_channels)

    ch = ChannelConfig(
        name="Channel A",
        channel_id="UCabcdefghijklmnopqrstuv",
        is_active=True,
    )

    # Add channel
    assert cfg.add_youtube_channel(ch) is True
    assert len(cfg.youtube_channels) == initial_count + 1

    # Adding duplicate should return False
    assert cfg.add_youtube_channel(ch) is False
    assert len(cfg.youtube_channels) == initial_count + 1

    # Remove channel
    assert cfg.remove_youtube_channel("UCabcdefghijklmnopqrstuv") is True
    assert len(cfg.youtube_channels) == initial_count

    # Removing non-existent returns False
    assert cfg.remove_youtube_channel("UCnonexistent1234567890") is False


def test_sources_config_load_fallback_when_corrupted(tmp_path):
    """Ensure graceful fallback to defaults when config JSON is corrupted."""
    bad_file = tmp_path / "corrupt_sources.json"
    bad_file.write_text("{invalid json: broken}", encoding="utf-8")

    cfg = SourcesConfig.load_from_file(bad_file)
    assert len(cfg.youtube_channels) == 0
    assert cfg.openai.is_active is True
    assert cfg.anthropic.is_active is True

