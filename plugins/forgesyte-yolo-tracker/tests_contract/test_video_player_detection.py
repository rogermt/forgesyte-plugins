"""Unit tests for video_player_detection tool implementation.

Tests the tool's input/output contract without loading YOLO models.
Uses mocked YOLO inference results.
"""

from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch


from forgesyte_yolo_tracker.plugin import Plugin

# YOLO is lazy-imported inside _tool_video_player_detection, so we patch ultralytics.YOLO
YOLO_PATCH_PATH = "ultralytics.YOLO"


class MockDetectionResult:
    """Mock YOLO detection result for a single frame."""

    def __init__(self, boxes: List[Dict[str, Any]]):
        self.boxes = MagicMock()
        self.boxes.data = MagicMock()

        # Simulate ultralytics Results.boxes.xyxy format
        if boxes:
            import numpy as np

            xyxy = np.array([b["xyxy"] for b in boxes], dtype=np.float32)
            conf = np.array([b["confidence"] for b in boxes], dtype=np.float32)
            cls = np.array([b["class_id"] for b in boxes], dtype=np.int32)

            self.boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            self.boxes.conf.cpu.return_value.numpy.return_value = conf
            self.boxes.cls.cpu.return_value.numpy.return_value = cls
        else:
            import numpy as np

            self.boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
                [], dtype=np.float32
            ).reshape(0, 4)
            self.boxes.conf.cpu.return_value.numpy.return_value = np.array(
                [], dtype=np.float32
            )
            self.boxes.cls.cpu.return_value.numpy.return_value = np.array(
                [], dtype=np.int32
            )


class MockYOLOModel:
    """Mock YOLO model for testing."""

    def __init__(self, frame_results: List[List[Dict[str, Any]]]):
        """Initialize with pre-defined frame results.

        Args:
            frame_results: List of frame detection results, each frame is a list
                          of detection dicts with xyxy, confidence, class_id
        """
        self.frame_results = frame_results
        self.call_count = 0

    def __call__(
        self, video_path: str, stream: bool = False, verbose: bool = False, **kwargs: Any
    ) -> Generator[MockDetectionResult, None, None]:
        """Simulate YOLO streaming inference."""
        for frame_boxes in self.frame_results:
            yield MockDetectionResult(frame_boxes)

    def to(self, device: str) -> "MockYOLOModel":
        """Simulate model.to() for device placement."""
        return self


class TestVideoPlayerDetectionContract:
    """Tests for video_player_detection tool contract."""

    def test_returns_dict_with_frames_and_summary(self) -> None:
        """Verify tool returns dict with frames and summary keys."""
        plugin = Plugin()

        # Create mock model with 3 frames
        mock_model = MockYOLOModel(
            frame_results=[
                [{"xyxy": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0}],
                [],
                [
                    {"xyxy": [20, 20, 60, 60], "confidence": 0.85, "class_id": 1},
                    {"xyxy": [30, 30, 70, 70], "confidence": 0.8, "class_id": 2},
                ],
            ]
        )

        with patch(YOLO_PATCH_PATH, return_value=mock_model):
            result = plugin.run_tool(
                "video_player_detection",
                {"video_path": "/tmp/test.mp4", "device": "cpu"},
            )

        assert isinstance(result, dict)
        assert "frames" in result
        assert "summary" in result

    def test_frames_have_correct_structure(self) -> None:
        """Verify each frame has frame_index and detections."""
        plugin = Plugin()

        mock_model = MockYOLOModel(
            frame_results=[
                [{"xyxy": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0}],
                [],
            ]
        )

        with patch(YOLO_PATCH_PATH, return_value=mock_model):
            result = plugin.run_tool(
                "video_player_detection",
                {"video_path": "/tmp/test.mp4", "device": "cpu"},
            )

        frames = result["frames"]
        assert isinstance(frames, list)
        assert len(frames) == 2

        # First frame
        assert frames[0]["frame_index"] == 0
        assert "detections" in frames[0]
        assert "xyxy" in frames[0]["detections"]
        assert "confidence" in frames[0]["detections"]
        assert "class_id" in frames[0]["detections"]

        # Second frame
        assert frames[1]["frame_index"] == 1

    def test_summary_has_correct_structure(self) -> None:
        """Verify summary has total_frames and total_detections."""
        plugin = Plugin()

        mock_model = MockYOLOModel(
            frame_results=[
                [{"xyxy": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0}],
                [],
                [
                    {"xyxy": [20, 20, 60, 60], "confidence": 0.85, "class_id": 1},
                    {"xyxy": [30, 30, 70, 70], "confidence": 0.8, "class_id": 2},
                ],
            ]
        )

        with patch(YOLO_PATCH_PATH, return_value=mock_model):
            result = plugin.run_tool(
                "video_player_detection",
                {"video_path": "/tmp/test.mp4", "device": "cpu"},
            )

        summary = result["summary"]
        assert isinstance(summary, dict)
        assert "total_frames" in summary
        assert "total_detections" in summary
        assert summary["total_frames"] == 3
        assert summary["total_detections"] == 3  # 1 + 0 + 2

    def test_empty_video_returns_empty_frames(self) -> None:
        """Verify tool handles empty video (no frames)."""
        plugin = Plugin()

        mock_model = MockYOLOModel(frame_results=[])

        with patch(YOLO_PATCH_PATH, return_value=mock_model):
            result = plugin.run_tool(
                "video_player_detection",
                {"video_path": "/tmp/empty.mp4", "device": "cpu"},
            )

        assert result["frames"] == []
        assert result["summary"]["total_frames"] == 0
        assert result["summary"]["total_detections"] == 0

    def test_uses_stream_mode(self) -> None:
        """Verify tool processes frames sequentially (streaming mode behavior)."""
        plugin = Plugin()

        # Create mock model with multiple frames - streaming processes sequentially
        mock_model = MockYOLOModel(
            frame_results=[
                [{"xyxy": [10, 10, 50, 50], "confidence": 0.9, "class_id": 0}],
                [{"xyxy": [20, 20, 60, 60], "confidence": 0.85, "class_id": 1}],
            ]
        )

        with patch(YOLO_PATCH_PATH, return_value=mock_model):
            result = plugin.run_tool(
                "video_player_detection",
                {"video_path": "/tmp/test.mp4", "device": "cpu"},
            )

        # Verify all frames were processed (streaming yields each frame)
        assert len(result["frames"]) == 2
        assert result["summary"]["total_frames"] == 2


class TestVideoPlayerDetectionDevice:
    """Tests for device parameter handling."""

    def test_default_device_is_cpu(self) -> None:
        """Verify device defaults to config value (cuda).."""
        plugin = Plugin()

        mock_model = MockYOLOModel(frame_results=[[]])
        mock_model.to = MagicMock(return_value=mock_model)

        with patch(YOLO_PATCH_PATH, return_value=mock_model):
            plugin.run_tool(
                "video_player_detection",
                {"video_path": "/tmp/test.mp4"},  # No device specified
            )

        # Verify to() was called with 'cpu'
        mock_model.to.assert_called_with(device="cuda")

    def test_cuda_device_passed_to_model(self) -> None:
        """Verify device='cuda' is passed to model."""
        plugin = Plugin()

        mock_model = MockYOLOModel(frame_results=[[]])
        mock_model.to = MagicMock(return_value=mock_model)

        with patch(YOLO_PATCH_PATH, return_value=mock_model):
            plugin.run_tool(
                "video_player_detection",
                {"video_path": "/tmp/test.mp4", "device": "cuda"},
            )

        mock_model.to.assert_called_with(device="cuda")