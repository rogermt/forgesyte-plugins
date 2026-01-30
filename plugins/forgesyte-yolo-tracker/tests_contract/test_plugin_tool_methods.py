"""TDD Tests for Plugin Tool Method Parameter Names.

Ensures that plugin tool method parameter names match manifest.json input names.
This prevents the frame_base64 vs frame_b64 mismatch bug (Issue #56).
"""

import inspect
import json
from pathlib import Path

import pytest


class TestToolMethodParameterNames:
    """Verify tool method parameter names match manifest.json inputs."""

    @pytest.fixture
    def manifest(self) -> dict:
        """Load manifest.json."""
        manifest_path = (
            Path(__file__).parents[1]
            / "forgesyte_yolo_tracker"
            / "manifest.json"
        )
        with open(manifest_path, "r") as f:
            return json.load(f)

    @pytest.fixture
    def plugin(self):
        """Get Plugin instance."""
        from forgesyte_yolo_tracker.plugin import Plugin
        return Plugin()

    def test_player_detection_params_match_manifest(
        self, manifest: dict, plugin
    ) -> None:
        """Verify player_detection handler params match manifest inputs."""
        manifest_inputs = set(manifest["tools"]["player_detection"]["inputs"].keys())
        # Get handler callable from tools dict
        handler = plugin.tools["player_detection"]["handler"]
        sig = inspect.signature(handler)
        # No 'self' to exclude for module-level functions
        method_params = set(sig.parameters.keys())
        
        assert manifest_inputs == method_params, (
            f"Parameter mismatch for player_detection!\n"
            f"Manifest inputs: {manifest_inputs}\n"
            f"Handler params: {method_params}"
        )

    def test_player_tracking_params_match_manifest(
        self, manifest: dict, plugin
    ) -> None:
        """Verify player_tracking handler params match manifest inputs."""
        manifest_inputs = set(manifest["tools"]["player_tracking"]["inputs"].keys())
        handler = plugin.tools["player_tracking"]["handler"]
        sig = inspect.signature(handler)
        method_params = set(sig.parameters.keys())
        
        assert manifest_inputs == method_params, (
            f"Parameter mismatch for player_tracking!\n"
            f"Manifest inputs: {manifest_inputs}\n"
            f"Handler params: {method_params}"
        )

    def test_ball_detection_params_match_manifest(
        self, manifest: dict, plugin
    ) -> None:
        """Verify ball_detection handler params match manifest inputs."""
        manifest_inputs = set(manifest["tools"]["ball_detection"]["inputs"].keys())
        handler = plugin.tools["ball_detection"]["handler"]
        sig = inspect.signature(handler)
        method_params = set(sig.parameters.keys())
        
        assert manifest_inputs == method_params, (
            f"Parameter mismatch for ball_detection!\n"
            f"Manifest inputs: {manifest_inputs}\n"
            f"Handler params: {method_params}"
        )

    def test_pitch_detection_params_match_manifest(
        self, manifest: dict, plugin
    ) -> None:
        """Verify pitch_detection handler params match manifest inputs."""
        manifest_inputs = set(manifest["tools"]["pitch_detection"]["inputs"].keys())
        handler = plugin.tools["pitch_detection"]["handler"]
        sig = inspect.signature(handler)
        method_params = set(sig.parameters.keys())
        
        assert manifest_inputs == method_params, (
            f"Parameter mismatch for pitch_detection!\n"
            f"Manifest inputs: {manifest_inputs}\n"
            f"Handler params: {method_params}"
        )

    def test_radar_params_match_manifest(
        self, manifest: dict, plugin
    ) -> None:
        """Verify radar handler params match manifest inputs."""
        manifest_inputs = set(manifest["tools"]["radar"]["inputs"].keys())
        handler = plugin.tools["radar"]["handler"]
        sig = inspect.signature(handler)
        method_params = set(sig.parameters.keys())
        
        assert manifest_inputs == method_params, (
            f"Parameter mismatch for radar!\n"
            f"Manifest inputs: {manifest_inputs}\n"
            f"Handler params: {method_params}"
        )

    def test_all_tools_have_frame_base64_not_frame_b64(
        self, manifest: dict, plugin
    ) -> None:
        """Ensure no tool uses 'frame_b64' - must use 'frame_base64'."""
        tools = manifest.get("tools", {})
        
        for tool_name in tools.keys():
            if not hasattr(plugin, tool_name):
                continue
            
            method = getattr(plugin, tool_name)
            sig = inspect.signature(method)
            param_names = list(sig.parameters.keys())
            
            assert "frame_b64" not in param_names, (
                f"Tool '{tool_name}' uses 'frame_b64' - should be 'frame_base64'!"
            )
