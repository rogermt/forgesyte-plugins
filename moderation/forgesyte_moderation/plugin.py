from forgesyte.server.models import PluginMetadata
from forgesyte.server.plugin_loader import BasePlugin

class Plugin(BasePlugin):
    name = "moderation"

    def metadata(self):
        return PluginMetadata(
            name="moderation",
            description="Content moderation plugin",
            version="1.0.0",
            inputs=["image"],
            outputs=["labels"],
        ).model_dump()

    def analyze(self, image_bytes, options=None):
        return {"labels": []}
