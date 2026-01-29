"""TDD: Test OCR plugin entrypoint contract and BasePlugin compliance.

These tests MUST pass after migration to BasePlugin architecture.
Tests should run without models/GPU (CPU-safe).
"""

import importlib
import sys
from importlib.metadata import entry_points
from typing import Any

import pytest


class TestOCREntrypointContract:
    """Verify OCR plugin loads via entrypoints and passes contract."""

    def test_ocr_entrypoint_exists(self) -> None:
        """Verify OCR entrypoint is registered in forgesyte.plugins."""
        if sys.version_info >= (3, 10):
            eps = entry_points(group="forgesyte.plugins")
        else:
            eps = entry_points().get("forgesyte.plugins", [])  # type: ignore
        ocr_eps = [ep for ep in eps if ep.name == "ocr"]

        assert len(ocr_eps) > 0, (
            "OCR entrypoint not found. "
            "Check pyproject.toml [project.entry-points.'forgesyte.plugins']"
        )

    def test_ocr_entrypoint_loads_plugin_class(self) -> None:
        """Verify OCR entrypoint loads the Plugin class successfully."""
        if sys.version_info >= (3, 10):
            eps = entry_points(group="forgesyte.plugins")
        else:
            eps = entry_points().get("forgesyte.plugins", [])  # type: ignore
        ocr_eps = [ep for ep in eps if ep.name == "ocr"]
        ep = ocr_eps[0]

        plugin_class: Any = ep.load()
        assert plugin_class is not None, "Entrypoint failed to load"
        assert hasattr(plugin_class, "__name__"), "Loaded object is not a class"

    def test_plugin_has_baseplugin_methods(self) -> None:
        """Verify Plugin class has all BasePlugin contract methods."""
        from forgesyte_ocr.plugin import Plugin

        plugin = Plugin()

        # Contract methods that BasePlugin requires
        assert hasattr(plugin, "name"), "Plugin missing 'name' attribute"
        assert hasattr(plugin, "tools"), "Plugin missing 'tools' dict"
        assert hasattr(plugin, "run_tool"), "Plugin missing 'run_tool' method"
        assert hasattr(plugin, "on_load"), "Plugin missing 'on_load' method"
        assert hasattr(plugin, "on_unload"), "Plugin missing 'on_unload' method"

    def test_plugin_name_is_string(self) -> None:
        """Verify plugin name is a string."""
        from forgesyte_ocr.plugin import Plugin

        plugin = Plugin()
        assert isinstance(plugin.name, str), "Plugin name must be string"
        assert plugin.name == "ocr", f"Expected name='ocr', got '{plugin.name}'"

    def test_plugin_tools_is_dict(self) -> None:
        """Verify tools dict exists and has structure."""
        from forgesyte_ocr.plugin import Plugin

        plugin = Plugin()
        assert isinstance(plugin.tools, dict), "Plugin.tools must be dict"
        assert "analyze" in plugin.tools, "Missing 'analyze' tool in tools dict"

    def test_plugin_tools_handler_is_string(self) -> None:
        """Verify tool handlers are strings (for BasePlugin contract)."""
        from forgesyte_ocr.plugin import Plugin

        plugin = Plugin()

        for tool_name, tool_config in plugin.tools.items():
            handler = tool_config.get("handler")
            assert isinstance(
                handler, str
            ), f"Tool '{tool_name}' handler must be string, got {type(handler)}"

    def test_plugin_run_tool_callable(self) -> None:
        """Verify run_tool method is callable."""
        from forgesyte_ocr.plugin import Plugin

        plugin = Plugin()
        assert callable(plugin.run_tool), "run_tool must be callable"

    def test_plugin_on_load_callable(self) -> None:
        """Verify on_load method is callable."""
        from forgesyte_ocr.plugin import Plugin

        plugin = Plugin()
        assert callable(plugin.on_load), "on_load must be callable"

    def test_plugin_on_load_executes(self) -> None:
        """Verify on_load executes without error."""
        from forgesyte_ocr.plugin import Plugin

        plugin = Plugin()
        try:
            plugin.on_load()
        except Exception as e:
            pytest.fail(f"on_load() raised exception: {e}")

    def test_plugin_no_app_imports(self) -> None:
        """Verify plugin.py has no legacy 'app.' imports (except BasePlugin)."""
        plugin_module = importlib.import_module("forgesyte_ocr.plugin")
        plugin_source = open(plugin_module.__file__).read()  # type: ignore

        # Allow `from app.plugins.base import BasePlugin` (required)
        # Disallow other `from app.*` imports (legacy)
        import_str = "from app.plugins.base import BasePlugin"
        has_baseplugin_import = import_str in plugin_source
        other_app_imports = plugin_source.replace(import_str, "")

        assert has_baseplugin_import, "Plugin missing BasePlugin import"
        assert "from app." not in other_app_imports, (
            "Plugin contains legacy 'from app.' imports (other than BasePlugin)"
        )
        assert "import app." not in other_app_imports, (
            "Plugin contains legacy 'import app.' imports"
        )

    def test_plugin_imports_from_baseplugin(self) -> None:
        """Verify plugin imports BasePlugin correctly."""
        plugin_module = importlib.import_module("forgesyte_ocr.plugin")
        plugin_source = open(plugin_module.__file__).read()  # type: ignore

        # Should import BasePlugin (either from forgesyte or relative)
        has_baseplugin_import = (
            "from forgesyte" in plugin_source or "BasePlugin" in plugin_source
        )
        assert (
            has_baseplugin_import
        ), "Plugin missing BasePlugin import (should import from forgesyte.core)"

    def test_plugin_instance_can_be_created(self) -> None:
        """Verify plugin can be instantiated without errors."""
        from forgesyte_ocr.plugin import Plugin

        try:
            plugin = Plugin()
            assert plugin is not None
        except Exception as e:
            pytest.fail(f"Failed to instantiate Plugin: {e}")

    def test_plugin_run_tool_with_valid_args(self) -> None:
        """Verify run_tool can be called with valid arguments."""
        import io

        from PIL import Image

        from forgesyte_ocr.plugin import Plugin

        plugin = Plugin()
        plugin.on_load()

        # Create sample image bytes
        img = Image.new("RGB", (100, 100), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")

        args = {
            "image_bytes": img_bytes.getvalue(),
            "options": {"language": "eng"},
        }

        result = plugin.run_tool("analyze", args)
        assert result is not None, "run_tool returned None"
