"""Ensure plugin lifecycle hooks execute without errors."""

from forgesyte_yolo_tracker.plugin import Plugin


def test_on_load_does_not_raise():
    """Verify on_load() executes without raising."""
    plugin = Plugin()
    plugin.on_load()  # Should not raise


def test_on_unload_does_not_raise():
    """Verify on_unload() executes without raising."""
    plugin = Plugin()
    plugin.on_unload()  # Should not raise
