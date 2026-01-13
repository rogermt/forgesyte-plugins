from forgesyte.server.models import PluginMetadata
from forgesyte.server.plugin_loader import BasePlugin

class Plugin(BasePlugin):
    name = "ocr"

    def metadata(self):
        return PluginMetadata(
            name="ocr",
            description="OCR plugin for text extraction",
            version="1.0.0",
            inputs=["image"],
            outputs=["text"],
        ).model_dump()

    def analyze(self, image_bytes, options=None):
        return {"text": "example output"}
