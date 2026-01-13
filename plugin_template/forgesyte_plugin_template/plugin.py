from forgesyte.server.models import PluginMetadata
from forgesyte.server.plugin_loader import BasePlugin

class Plugin(BasePlugin):
    name = "template_plugin"

    def metadata(self):
        return PluginMetadata(
            name="template_plugin",
            description="Template plugin for ForgeSyte",
            version="1.0.0",
            inputs=["image"],
            outputs=["json"],
        ).model_dump()

    def analyze(self, image_bytes, options=None):
        return {"result": "template output"}
