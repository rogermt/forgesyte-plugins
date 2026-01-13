from forgesyte.server.models import PluginMetadata
from forgesyte.server.plugin_loader import BasePlugin

class Plugin(BasePlugin):
    name = "motion_detector"

    def metadata(self):
        return PluginMetadata(
            name="motion_detector",
            description="Motion detection plugin",
            version="1.0.0",
            inputs=["image"],
            outputs=["motion"],
        ).model_dump()

    def analyze(self, image_bytes, options=None):
        return {"motion": False}
