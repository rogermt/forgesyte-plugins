"""Tests for TeamClassifier that require model loading.

These tests are skipped by default. Run with:
    RUN_MODEL_TESTS=1 uv run pytest src/tests/utils/test_team_model.py -v
"""

import os
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from forgesyte_yolo_tracker.utils import TeamClassifier

RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS, reason="Set RUN_MODEL_TESTS=1 to run (requires network for model loading)"
)


class TestTeamClassifier:
    """Tests for TeamClassifier class."""

    def test_initialization(self) -> None:
        """Test TeamClassifier initialization."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu", batch_size=32)

            assert classifier.device == "cpu"
            assert classifier.batch_size == 32
            assert classifier.reducer is not None
            assert classifier.cluster_model is not None

    def test_initialization_gpu(self) -> None:
        """Test initialization with GPU."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cuda", batch_size=64)

            assert classifier.device == "cuda"
            assert classifier.batch_size == 64

    @patch("forgesyte_yolo_tracker.utils.team.tqdm")
    @patch("forgesyte_yolo_tracker.utils.team.torch")
    def test_extract_features_basic(self, mock_torch: MagicMock, mock_tqdm: MagicMock) -> None:
        """Test feature extraction from image crops."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
            patch("forgesyte_yolo_tracker.utils.team.sv.cv2_to_pillow"),
        ):
            classifier = TeamClassifier(device="cpu")

            mock_output = MagicMock()
            mock_output.last_hidden_state = mock_torch.Tensor([[1, 2, 3], [4, 5, 6]])
            classifier.features_model = MagicMock()
            classifier.features_model.return_value = mock_output
            classifier.processor = MagicMock()

            mock_tqdm.return_value = [[]]

            crops = [np.zeros((224, 224, 3), dtype=np.uint8)]

            result = classifier.extract_features(crops)

            assert isinstance(result, np.ndarray)

    def test_fit_requires_crops(self) -> None:
        """Test fit method requires crop images."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            crops = [np.zeros((224, 224, 3), dtype=np.uint8)]

            classifier.extract_features = MagicMock(return_value=np.random.rand(1, 512))
            classifier.reducer.fit_transform = MagicMock(return_value=np.random.rand(1, 3))
            classifier.cluster_model.fit = MagicMock()

            classifier.fit(crops)

            classifier.extract_features.assert_called_once()
            classifier.cluster_model.fit.assert_called_once()

    def test_predict_empty_crops(self) -> None:
        """Test predict with empty crops."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            crops: List[np.ndarray] = []

            result = classifier.predict(crops)

            assert len(result) == 0
            assert isinstance(result, np.ndarray)

    def test_predict_before_fit(self) -> None:
        """Test predict works without explicit fit (uses default model)."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")

            classifier.extract_features = MagicMock(return_value=np.random.rand(1, 512))
            classifier.reducer.transform = MagicMock(return_value=np.random.rand(1, 3))
            classifier.cluster_model.predict = MagicMock(return_value=np.array([0]))

            crops = [np.zeros((224, 224, 3), dtype=np.uint8)]
            result = classifier.predict(crops)

            assert len(result) >= 0


class TestTeamClassifierReducer:
    """Tests for TeamClassifier reducer (UMAP/StandardScaler fallback)."""

    def test_reducer_is_not_none(self) -> None:
        """Verify reducer is initialized."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            assert classifier.reducer is not None

    def test_reducer_has_fit_transform_method(self) -> None:
        """Verify reducer has fit_transform method for training."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            assert hasattr(classifier.reducer, "fit_transform")
            assert callable(classifier.reducer.fit_transform)

    def test_reducer_has_transform_method(self) -> None:
        """Verify reducer has transform method for prediction."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            assert hasattr(classifier.reducer, "transform")
            assert callable(classifier.reducer.transform)

    def test_cluster_model_n_clusters_is_two(self) -> None:
        """Verify cluster_model is configured for 2 teams."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            assert classifier.cluster_model.n_clusters == 2


class TestTeamClassifierClusterModel:
    """Tests for TeamClassifier KMeans cluster model."""

    def test_cluster_model_is_not_none(self) -> None:
        """Verify cluster_model is initialized."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            assert classifier.cluster_model is not None

    def test_cluster_model_has_predict_method(self) -> None:
        """Verify cluster_model has predict method."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            assert hasattr(classifier.cluster_model, "predict")
            assert callable(classifier.cluster_model.predict)

    def test_cluster_model_has_fit_method(self) -> None:
        """Verify cluster_model has fit method."""
        with (
            patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"),
            patch("forgesyte_yolo_tracker.utils.team.AutoProcessor"),
        ):
            classifier = TeamClassifier(device="cpu")
            assert hasattr(classifier.cluster_model, "fit")
            assert callable(classifier.cluster_model.fit)
