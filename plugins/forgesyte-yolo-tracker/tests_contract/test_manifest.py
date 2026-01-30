"""Tests for manifest file and plugin registration.

Tests cover:
- Manifest file structure and required fields
- Plugin registration and ID consistency (Issue #110)
- Version format validation
- Capabilities verification
"""

import json
from pathlib import Path

import pytest

from forgesyte_yolo_tracker.plugin import Plugin


class TestManifestContent:
    """Test manifest file contents and structure."""

    @pytest.fixture
    def manifest(self) -> dict:
        """Load the manifest file."""
        manifest_path = Path(__file__).parent.parent / "forgesyte_yolo_tracker" / "manifest.json"
        with open(manifest_path) as f:
            return json.load(f)

    def test_manifest_required_fields(self, manifest: dict) -> None:
        """Test that manifest contains all required fields."""
        required_fields = [
            "id",
            "name",
            "version",
            "description",
            "author",
            "license",
            "capabilities",
        ]
        for field in required_fields:
            assert field in manifest, f"Missing required field: {field}"

    def test_manifest_id_format(self, manifest: dict) -> None:
        """Test that manifest ID follows naming convention."""
        assert isinstance(manifest["id"], str)
        assert len(manifest["id"]) > 0
        assert "-" in manifest["id"]

    def test_manifest_version_format(self, manifest: dict) -> None:
        """Test that version follows semantic versioning."""
        version = manifest["version"]
        parts = version.split(".")
        assert len(parts) == 3, "Version should be in semver format (X.Y.Z)"
        for part in parts:
            assert part.isdigit(), "Version parts should be numbers"

    def test_manifest_capabilities_list(self, manifest: dict) -> None:
        """Test that capabilities is a non-empty list."""
        assert isinstance(manifest["capabilities"], list)
        assert len(manifest["capabilities"]) > 0

    def test_manifest_capabilities_content(self, manifest: dict) -> None:
        """Test that capabilities contain expected analysis modes."""
        expected_capabilities = [
            "player_detection",
            "player_tracking",
            "ball_detection",
            "pitch_detection",
            "radar_visualization",
        ]
        for capability in expected_capabilities:
            assert capability in manifest["capabilities"], (
                f"Missing capability '{capability}' in manifest"
            )

    def test_manifest_models_structure(self, manifest: dict) -> None:
        """Test models array structure if present."""
        if "models" in manifest:
            assert isinstance(manifest["models"], list)
            for model in manifest["models"]:
                assert "name" in model
                assert "type" in model
                assert "description" in model

    def test_manifest_supported_formats(self, manifest: dict) -> None:
        """Test supported formats if specified."""
        if "supported_formats" in manifest:
            assert isinstance(manifest["supported_formats"], list)
            assert len(manifest["supported_formats"]) > 0

    def test_manifest_dependencies(self, manifest: dict) -> None:
        """Test dependencies if specified."""
        if "dependencies" in manifest:
            assert isinstance(manifest["dependencies"], list)
            expected_deps = ["ultralytics", "supervision", "numpy"]
            for dep in expected_deps:
                assert dep in manifest["dependencies"]


class TestManifestIdMatchIssue110:
    """Test manifest ID matches Plugin.name (Issue #110).

    This test catches the regression where manifest.json ID was changed from
    'yolo-tracker' to 'forgesyte-yolo-tracker' but Plugin.name was not updated,
    causing plugin lookup failures during server operation.

    Error flow that this test prevents:
    1. Frontend sends pluginId: "yolo-tracker"
    2. Backend calls registry.get("yolo-tracker") -> finds plugin
    3. Backend reads manifest.json -> finds "id": "forgesyte-yolo-tracker"
    4. ID MISMATCH DETECTED -> error returned
    5. Frontend fails: "Invalid JSON from tool"
    """

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    @pytest.fixture
    def manifest(self) -> dict:
        """Load the manifest file."""
        manifest_path = Path(__file__).parent.parent / "forgesyte_yolo_tracker" / "manifest.json"
        with open(manifest_path) as f:
            return json.load(f)

    def test_manifest_id_matches_plugin_name(
        self, plugin: Plugin, manifest: dict
    ) -> None:
        """Test that manifest ID matches Plugin.name.

        This is the core test for Issue #110 - it catches the mismatch
        between the manifest.json "id" field and the Plugin.name attribute.

        The server uses registry.get(plugin_name) to find plugins, then reads
        the manifest to validate the ID. If they don't match, the lookup fails.

        Expected: manifest["id"] == Plugin.name
        """
        manifest_id = manifest.get("id")
        plugin_name = plugin.name

        assert manifest_id == plugin_name, (
            f"Manifest ID mismatch! "
            f"manifest.json has id='{manifest_id}' but Plugin.name='{plugin_name}'. "
            f"This mismatch causes plugin lookup failures. "
            f"Please update manifest.json to have 'id': '{plugin_name}'"
        )

    def test_plugin_name_is_yolo_tracker(self, plugin: Plugin) -> None:
        """Test that Plugin.name is the expected value 'yolo-tracker'."""
        assert plugin.name == "yolo-tracker", (
            f"Plugin.name should be 'yolo-tracker' but got '{plugin.name}'. "
            f"This name is used by the frontend for API calls."
        )
