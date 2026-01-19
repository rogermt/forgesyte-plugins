"""Tests for manifest file and plugin registration."""

import json
from pathlib import Path

import pytest


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
            "team_classification",
            "pitch_detection",
            "radar_visualization",
        ]
        for capability in expected_capabilities:
            assert capability in manifest["capabilities"]

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
