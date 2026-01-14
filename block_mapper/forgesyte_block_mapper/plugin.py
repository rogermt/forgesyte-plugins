from forgesyte.server.models import PluginMetadata
from forgesyte.server.plugin_loader import BasePlugin


class Plugin(BasePlugin):
    name = "block_mapper"

    def metadata(self):
        return PluginMetadata(
            name="block_mapper",
            description="Block mapping plugin",
            version="1.0.0",
            inputs=["image"],
            outputs=["regions"],
        ).model_dump()

    def analyze(self, image_bytes, options=None):
        return {"regions": []}
