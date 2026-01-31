"""Base detector class for unified inference across all detectors.

This module provides a generic BaseDetector class that eliminates code duplication
across player, ball, and pitch detectors. All detectors share common functionality:
- Model loading and caching
- Frame encoding to base64
- Inference execution
- Frame annotation

Detectors inherit from BaseDetector and provide detector-specific configuration.
"""

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cv2
import numpy as np
import supervision as sv

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BaseDetector:
    """Generic detection base class for YOLO-based inference.

    This class provides shared functionality for all detectors:
    - Model loading and caching per detector instance
    - Frame encoding to base64 JPEG
    - YOLO inference execution
    - Result formatting and annotation

    Attributes:
        detector_name: Unique name for this detector (e.g., 'player', 'ball', 'pitch')
        model_name: Name of the model file (e.g., 'football-player-detection-v3.pt')
        default_confidence: Default confidence threshold for detections
        imgsz: Input image size for YOLO (default 1280)
        class_names: Dictionary mapping class_id to class_name, or None if no classes
        colors: Dictionary mapping class_id to hex color, or None
    """

    def __init__(
        self,
        detector_name: str,
        model_name: str,
        default_confidence: float,
        imgsz: int = 1280,
        class_names: Optional[Dict[int, str]] = None,
        colors: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialize BaseDetector instance.

        Args:
            detector_name: Unique identifier for this detector
            model_name: Model filename (relative to models directory)
            default_confidence: Default confidence threshold (0.0-1.0)
            imgsz: Input image size for YOLO inference (default 1280)
            class_names: Optional dict mapping class IDs to names
            colors: Optional dict mapping class IDs to hex colors

        Raises:
            ValueError: If confidence threshold not in [0.0, 1.0]
        """
        if not 0.0 <= default_confidence <= 1.0:
            raise ValueError(f"default_confidence must be in [0.0, 1.0], got {default_confidence}")

        self.detector_name: str = detector_name
        self.model_name: str = model_name
        self.default_confidence: float = default_confidence
        self.imgsz: int = imgsz
        self.class_names: Optional[Dict[int, str]] = class_names
        self.colors: Optional[Dict[int, str]] = colors
        self._model: Optional[Any] = None  # YOLO type

        # Compute model path
        model_dir = Path(__file__).parent.parent / "models"
        self.model_path: str = str(model_dir / model_name)

        logger.debug(f"🔍 Initialized {detector_name} detector")
        logger.debug(f"🔍 Model name: {model_name}")
        logger.debug(f"🔍 Model path: {self.model_path}")
        logger.debug(f"🔍 Default confidence: {default_confidence}")
        logger.debug(f"🔍 Input size (imgsz): {imgsz}")

    def get_model(self, device: str = "cpu") -> Any:
        """Get or create cached YOLO model.

        Model is cached per detector instance to avoid reloading.
        Logs model loading info and warns if model is a stub (< 1KB).

        Args:
            device: Device to run model on ('cpu' or 'cuda')

        Returns:
            YOLO model instance

        Raises:
            FileNotFoundError: If model file does not exist
        """
        from ultralytics import YOLO

        if self._model is not None:
            logger.debug(f"🎯 Using cached {self.detector_name} model")
            return self._model

        logger.info(f"📦 Loading {self.detector_name} model from: {self.model_path}")

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        model_size_kb = model_file.stat().st_size / 1024
        logger.info(f"📦 Model file size: {model_size_kb:.2f} KB")

        if model_size_kb < 1:
            logger.warning(
                f"⚠️  Model is a stub ({model_size_kb:.2f} KB)! " "Replace with real model."
            )

        self._model = YOLO(self.model_path).to(device=device)
        logger.info(f"✅ Model loaded successfully on device: {device}")

        return self._model

    def _encode_frame_to_base64(self, frame: np.ndarray[Any, np.dtype[Any]]) -> str:
        """Encode frame to base64 JPEG string.

        Encodes the frame as JPEG to reduce size, then converts to base64
        for safe transmission and storage.

        Args:
            frame: Input image frame (BGR format, numpy array)

        Returns:
            Base64 encoded JPEG string

        Raises:
            ValueError: If frame encoding fails
        """
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Failed to encode frame to JPEG")

        encoded = base64.b64encode(bytes(buffer)).decode("utf-8")
        return encoded

    def detect_json(
        self,
        frame: np.ndarray[Any, np.dtype[Any]],
        device: str = "cpu",
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run detection inference - JSON only (no annotated frame).

        Executes YOLO inference and returns structured detection results.
        If class_names provided during init, includes class names in results.

        Args:
            frame: Input image frame (BGR format, numpy array)
            device: Device to run model on ('cpu' or 'cuda')
            confidence: Detection confidence threshold (uses default if None)

        Returns:
            Dictionary with keys:
            - detections: List of detection dicts with xyxy, confidence, class_id, [class_name]
            - count: Total number of detections
            - classes: Dict of class_name->count (only if class_names provided)

        Raises:
            ValueError: If frame is empty or invalid
        """
        if confidence is None:
            confidence = self.default_confidence

        logger.info(
            f"🎬 Starting {self.detector_name} detection "
            f"(device={device}, confidence={confidence})"
        )
        logger.debug(f"🎬 Frame shape: {frame.shape}")

        model = self.get_model(device=device)

        logger.info("🔫 Running YOLO inference...")
        result = model(frame, imgsz=self.imgsz, conf=confidence, verbose=False)[0]
        logger.info("🔫 Inference complete. Processing results...")

        detections = sv.Detections.from_ultralytics(result)
        logger.info(f"📊 Found {len(detections)} detections")

        detection_list: List[Dict[str, Any]] = []
        class_counts: Dict[str, int] = {}

        # Initialize class counts if class_names provided
        if self.class_names:
            class_counts = {name: 0 for name in self.class_names.values()}

        for i in range(len(detections)):
            xyxy_arr = detections.xyxy
            conf_arr = detections.confidence
            cls_arr = detections.class_id
            if xyxy_arr is None or conf_arr is None or cls_arr is None:
                continue
            xyxy = xyxy_arr[i]
            conf = float(conf_arr[i])
            cls = int(cls_arr[i])

            detection_dict: Dict[str, Any] = {
                "xyxy": xyxy.tolist(),
                "confidence": conf,
                "class_id": cls,
            }

            # Add class name if class_names provided
            if self.class_names:
                class_name = self.class_names.get(cls, f"class_{cls}")
                detection_dict["class_name"] = class_name

                if class_name in class_counts:
                    class_counts[class_name] += 1

            detection_list.append(detection_dict)

        logger.info(f"✅ Detection complete: {len(detection_list)} objects found")
        if self.class_names:
            logger.info(f"✅ Class breakdown: {class_counts}")

        result_dict: Dict[str, Any] = {
            "detections": detection_list,
            "count": len(detection_list),
        }

        if self.class_names:
            result_dict["classes"] = class_counts

        return result_dict

    def detect_json_with_annotated_frame(
        self,
        frame: np.ndarray[Any, np.dtype[Any]],
        device: str = "cpu",
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run detection inference - JSON + base64 annotated frame.

        Executes YOLO inference, creates annotated frame with boxes and labels,
        then encodes to base64. Returns all detection data plus annotated frame.

        Args:
            frame: Input image frame (BGR format, numpy array)
            device: Device to run model on ('cpu' or 'cuda')
            confidence: Detection confidence threshold (uses default if None)

        Returns:
            Dictionary with keys:
            - detections: List of detection dicts
            - count: Total number of detections
            - classes: Dict of class_name->count (only if class_names provided)
            - annotated_frame_base64: Base64 encoded annotated frame

        Raises:
            ValueError: If frame encoding fails
        """
        if confidence is None:
            confidence = self.default_confidence

        # Get detection results
        result_dict = self.detect_json(frame, device=device, confidence=confidence)

        # Create annotated frame
        model = self.get_model(device=device)
        detection_result = model(frame, imgsz=self.imgsz, conf=confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(detection_result)

        # Build labels if class_names provided
        labels: Optional[List[str]] = None
        if self.class_names:
            cls_arr = detections.class_id
            if cls_arr is not None:
                labels = [self.class_names.get(int(cls), f"class_{cls}") for cls in cls_arr]

        # Annotate frame
        annotated = self._annotate_frame(frame, detections, labels)

        # Add annotated frame to result
        result_dict["annotated_frame_base64"] = self._encode_frame_to_base64(annotated)

        return result_dict

    def _annotate_frame(
        self,
        frame: np.ndarray[Any, np.dtype[Any]],
        detections: sv.Detections,
        labels: Optional[List[str]] = None,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Create annotated frame with boxes and labels.

        Draws bounding boxes and labels on a copy of the frame.
        Uses colors from self.colors if available.

        Args:
            frame: Input image frame (BGR format, numpy array)
            detections: Supervision Detections object
            labels: Optional list of label strings

        Returns:
            Annotated frame (copy of input with boxes/labels drawn)
        """
        annotated = frame.copy()

        if len(detections) == 0:
            return annotated

        # Create annotators
        if self.colors:
            cls_arr = detections.class_id
            if cls_arr is not None:
                color_list = [self.colors.get(int(cls), "#FFFFFF") for cls in cls_arr]
                colors = sv.ColorPalette.from_hex(color_list)
            else:
                colors = sv.ColorPalette.DEFAULT
        else:
            colors = sv.ColorPalette.DEFAULT

        box_annotator = sv.BoxAnnotator(color=colors, thickness=2)

        annotated = box_annotator.annotate(annotated, detections)

        # Add labels if provided
        if labels:
            label_annotator = sv.LabelAnnotator(
                color=colors,
                text_color=sv.Color.from_hex("#FFFFFF"),
                text_padding=5,
                text_thickness=1,
            )
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

        return annotated  # type: ignore[no-any-return]
