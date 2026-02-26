"""Tests for plugin manifest.

Phase 3 of v0.10.0 implementation.
Tests for manifest structure and capabilities.
"""

import json
from pathlib import Path
from typing import Any

import pytest


class TestManifest:
    """Tests for plugin manifest."""

    @pytest.fixture  # type: ignore[misc]
    def manifest(self) -> dict[str, Any]:
        manifest_path = Path(__file__).parent.parent / "src" / "forgesyte_yolo_tracker" / "manifest.json"
        with open(manifest_path) as f:
            data: dict[str, Any] = json.load(f)
        return data

    def test_manifest_has_streaming_capability(self, manifest: dict[str, Any]) -> None:
        """Test that manifest includes streaming_video_analysis capability."""
        assert "streaming_video_analysis" in manifest["capabilities"]

    def test_manifest_version_is_0_10_0(self, manifest: dict[str, Any]) -> None:
        """Test that version is 0.10.0."""
        assert manifest["version"] == "0.10.0"

    def test_manifest_has_required_fields(self, manifest: dict[str, Any]) -> None:
        """Test that manifest has all required fields."""
        required_fields = ["id", "name", "version", "description", "capabilities", "tools"]
        for field in required_fields:
            assert field in manifest, f"Missing required field: {field}"

    def test_video_tools_have_streaming_capability(self, manifest: dict[str, Any]) -> None:
        """Test that video tools include streaming_video_analysis capability."""
        video_tools = [t for t in manifest["tools"] if "video" in t.get("id", "")]

        for tool in video_tools:
            tool_caps = tool.get("capabilities", [])
            assert "streaming_video_analysis" in tool_caps, \
                f"Tool {tool['id']} missing streaming_video_analysis capability"

    def test_all_capabilities_have_tools(self, manifest: dict[str, Any]) -> None:
        """Test that each capability has at least one tool."""
        capabilities = set(manifest["capabilities"])
        tool_capabilities = set()

        for tool in manifest["tools"]:
            tool_capabilities.update(tool.get("capabilities", []))

        # All declared capabilities should be used by at least one tool
        for cap in capabilities:
            assert cap in tool_capabilities, f"Capability {cap} not used by any tool"

    def test_video_player_tracking_tool_exists(self, manifest: dict[str, Any]) -> None:
        """Test that video_player_tracking tool exists."""
        tool_ids = [t["id"] for t in manifest["tools"]]
        assert "video_player_tracking" in tool_ids

    def test_video_ball_detection_tool_exists(self, manifest: dict[str, Any]) -> None:
        """Test that video_ball_detection tool exists."""
        tool_ids = [t["id"] for t in manifest["tools"]]
        assert "video_ball_detection" in tool_ids

    def test_video_pitch_detection_tool_exists(self, manifest: dict[str, Any]) -> None:
        """Test that video_pitch_detection tool exists."""
        tool_ids = [t["id"] for t in manifest["tools"]]
        assert "video_pitch_detection" in tool_ids

    def test_video_radar_tool_exists(self, manifest: dict[str, Any]) -> None:
        """Test that video_radar tool exists."""
        tool_ids = [t["id"] for t in manifest["tools"]]
        assert "video_radar" in tool_ids

    def test_manifest_json_valid(self, manifest: dict[str, Any]) -> None:
        """Test that manifest.json is valid JSON."""
        assert isinstance(manifest, dict)
        assert "tools" in manifest
        assert isinstance(manifest["tools"], list)