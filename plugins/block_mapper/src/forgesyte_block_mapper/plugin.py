"""
Minecraft Block Mapper Plugin.
Converts real-world images into building plans using Minecraft blocks.
Optimized with Pydantic for data integrity and NumPy for color analysis.
"""

import io
import logging
from typing import Any, cast

try:
    import numpy as np
    from PIL import Image

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from app.models import AnalysisResult, PluginMetadata
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Validated Data Models (Internal)
# ---------------------------------------------------------------------------


class BlockPaletteEntry(BaseModel):  # type: ignore[misc]
    """Metadata for a single block type within the result palette."""

    count: int
    percentage: float = Field(ge=0.0, le=100.0)
    rgb: list[int] = Field(min_length=3, max_length=3)


class SchematicBlock(BaseModel):  # type: ignore[misc]
    """Individual block coordinate within the schematic."""

    x: int
    y: int
    z: int
    block: str


# ---------------------------------------------------------------------------
# 2. Plugin Implementation
# ---------------------------------------------------------------------------


class Plugin:
    """
    Block mapper plugin for converting images to Minecraft block grids.
    Utilizes weighted Euclidean distance for human perception-based color mapping.
    """

    name: str = "block_mapper"
    version: str = "1.0.0"
    description: str = "Map image colors to Minecraft blocks for building"

    def __init__(self) -> None:
        """Initialize with the simplified Minecraft block palette."""
        # Standard palette from source
        self.block_colors = {
            "stone": (128, 128, 128),
            "dirt": (134, 96, 67),
            "grass_block": (91, 135, 48),
            "oak_planks": (162, 131, 78),
            "cobblestone": (122, 122, 122),
            "sand": (219, 207, 163),
            "gravel": (136, 126, 126),
            "oak_log": (109, 85, 50),
            "oak_leaves": (54, 99, 31),
            "glass": (175, 213, 219),
            "white_wool": (234, 236, 236),
            "orange_wool": (241, 118, 20),
            "magenta_wool": (189, 68, 179),
            "light_blue_wool": (58, 175, 217),
            "yellow_wool": (248, 198, 39),
            "lime_wool": (112, 185, 26),
            "pink_wool": (237, 141, 172),
            "gray_wool": (63, 68, 72),
            "light_gray_wool": (142, 142, 135),
            "cyan_wool": (21, 137, 145),
            "purple_wool": (121, 42, 172),
            "blue_wool": (53, 57, 157),
            "brown_wool": (114, 71, 40),
            "green_wool": (84, 109, 27),
            "red_wool": (161, 39, 35),
            "black_wool": (20, 21, 26),
            "gold_block": (249, 212, 66),
            "iron_block": (220, 220, 220),
            "diamond_block": (97, 220, 225),
            "emerald_block": (42, 176, 71),
            "redstone_block": (170, 28, 28),
            "lapis_block": (39, 67, 138),
            "water": (63, 118, 228),
            "lava": (207, 92, 15),
        }

    def metadata(self) -> PluginMetadata:
        """Returns strictly validated metadata via Pydantic."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            inputs=["image"],
            outputs=["text", "blocks", "confidence"],
            config_schema={
                "width": {"type": "integer", "default": 64, "min": 8, "max": 256},
                "height": {"type": "integer", "default": 64, "min": 8, "max": 256},
                "dithering": {"type": "boolean", "default": False},
            },
        )

    def analyze(
        self, image_bytes: bytes, options: dict[str, Any] | None = None
    ) -> AnalysisResult:
        """Performs image color mapping and schematic generation."""
        if not HAS_DEPS:
            logger.error("Missing critical dependencies: PIL or numpy")
            return AnalysisResult(
                text="",
                blocks=[],
                confidence=0.0,
                language=None,
                error="PIL and numpy required",
            )

        opts = options or {}
        target_w = opts.get("width", 64)
        target_h = opts.get("height", 64)

        try:
            # 1. Image Preprocessing
            with Image.open(io.BytesIO(image_bytes)) as original_img:
                img = original_img.convert("RGB").resize(
                    (target_w, target_h), Image.Resampling.LANCZOS
                )

            pixels = np.array(img)
            schematic_blocks: list[SchematicBlock] = []

            # 2. Pixel Mapping
            for y in range(target_h):
                for x in range(target_w):
                    pixel_rgb = tuple(pixels[y, x])
                    block_name = self._find_nearest_block(
                        cast(tuple[int, int, int], pixel_rgb)
                    )
                    schematic_blocks.append(
                        SchematicBlock(x=x, y=0, z=y, block=block_name)
                    )

            # 3. Construct Universal AnalysisResult
            # text: Summary (e.g., dimensions)
            # blocks: The schematic blocks
            # confidence: 1.0 (deterministic mapping)
            summary = f"Mapped to {target_w}x{target_h} block grid."

            return AnalysisResult(
                text=summary,
                blocks=[b.model_dump() for b in schematic_blocks],
                confidence=1.0,
                language=None,
                error=None,
            )

        except Exception as e:
            logger.exception("Block mapping analysis failed", extra={"error": str(e)})
            return AnalysisResult(
                text="",
                blocks=[],
                confidence=0.0,
                language=None,
                error=str(e),
            )

    def _find_nearest_block(self, rgb: tuple[int, int, int]) -> str:
        """Weighted Euclidean distance formula for perception-accurate mapping."""
        min_dist, nearest_block = float("inf"), "stone"

        for block_name, block_rgb in self.block_colors.items():
            r_mean = (rgb[0] + block_rgb[0]) / 2
            dr = int(rgb[0]) - int(block_rgb[0])
            dg = int(rgb[1]) - int(block_rgb[1])
            db = int(rgb[2]) - int(block_rgb[2])

            # Weighted distance to account for human eye sensitivity
            distance = (
                (2 + r_mean / 256) * dr**2
                + 4 * dg**2
                + (2 + (255 - r_mean) / 256) * db**2
            )

            if distance < min_dist:
                min_dist = distance
                nearest_block = block_name

        return nearest_block

    def on_load(self) -> None:
        """Lifecycle hook: logs palette registration."""
        logger.info(
            f"Block mapper plugin loaded with {len(self.block_colors)} block types"
        )

    def on_unload(self) -> None:
        """Lifecycle hook: cleanup."""
        logger.info("Block mapper plugin unloaded")
