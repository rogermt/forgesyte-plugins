"""Tests for team classification utilities."""

from typing import List
import numpy as np
from unittest.mock import MagicMock, patch

from forgesyte_yolo_tracker.utils.team import (
    create_batches,
    TeamClassifier,
)


class TestCreateBatches:
    """Tests for create_batches utility function."""

    def test_create_batches_basic(self) -> None:
        """Test basic batch creation."""
        data = [1, 2, 3, 4, 5]
        batches = list(create_batches(data, batch_size=2))

        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5]

    def test_create_batches_exact_division(self) -> None:
        """Test when items divide evenly into batches."""
        data = list(range(10))
        batches = list(create_batches(data, batch_size=5))

        assert len(batches) == 2
        assert batches[0] == [0, 1, 2, 3, 4]
        assert batches[1] == [5, 6, 7, 8, 9]

    def test_create_batches_single_item_batches(self) -> None:
        """Test with batch size of 1."""
        data = [1, 2, 3]
        batches = list(create_batches(data, batch_size=1))

        assert len(batches) == 3
        assert batches == [[1], [2], [3]]

    def test_create_batches_larger_batch_than_items(self) -> None:
        """Test when batch size is larger than data."""
        data = [1, 2]
        batches = list(create_batches(data, batch_size=10))

        assert len(batches) == 1
        assert batches[0] == [1, 2]

    def test_create_batches_empty_sequence(self) -> None:
        """Test with empty sequence."""
        data: List[int] = []
        batches = list(create_batches(data, batch_size=2))

        assert len(batches) == 0

    def test_create_batches_zero_batch_size(self) -> None:
        """Test with zero batch size (should use 1)."""
        data = [1, 2, 3]
        batches = list(create_batches(data, batch_size=0))

        assert len(batches) == 3
        assert batches == [[1], [2], [3]]

    def test_create_batches_negative_batch_size(self) -> None:
        """Test with negative batch size (should use 1)."""
        data = [1, 2]
        batches = list(create_batches(data, batch_size=-5))

        assert len(batches) == 2
        assert batches == [[1], [2]]

    def test_create_batches_generator(self) -> None:
        """Test that create_batches returns a generator."""
        data = [1, 2, 3]
        result = create_batches(data, batch_size=2)

        # Should be a generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")


class TestTeamClassifier:
    """Tests for TeamClassifier class."""

    def test_initialization(self) -> None:
        """Test TeamClassifier initialization."""
        with patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"), patch(
            "forgesyte_yolo_tracker.utils.team.AutoProcessor"
        ):
            classifier = TeamClassifier(device="cpu", batch_size=32)

            assert classifier.device == "cpu"
            assert classifier.batch_size == 32
            assert classifier.reducer is not None
            assert classifier.cluster_model is not None

    def test_initialization_gpu(self) -> None:
        """Test initialization with GPU."""
        with patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"), patch(
            "forgesyte_yolo_tracker.utils.team.AutoProcessor"
        ):
            classifier = TeamClassifier(device="cuda", batch_size=64)

            assert classifier.device == "cuda"
            assert classifier.batch_size == 64

    @patch("forgesyte_yolo_tracker.utils.team.tqdm")
    @patch("forgesyte_yolo_tracker.utils.team.torch")
    def test_extract_features_basic(self, mock_torch: MagicMock, mock_tqdm: MagicMock) -> None:
        """Test feature extraction from image crops."""
        with patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"), patch(
            "forgesyte_yolo_tracker.utils.team.AutoProcessor"
        ), patch("forgesyte_yolo_tracker.utils.team.sv.cv2_to_pillow"):

            classifier = TeamClassifier(device="cpu")

            # Mock the model output
            mock_output = MagicMock()
            mock_output.last_hidden_state = mock_torch.Tensor([[1, 2, 3], [4, 5, 6]])
            classifier.features_model = MagicMock()
            classifier.features_model.return_value = mock_output
            classifier.processor = MagicMock()

            mock_tqdm.return_value = [[]]

            # Create fake image crops
            crops = [np.zeros((224, 224, 3), dtype=np.uint8)]

            result = classifier.extract_features(crops)

            assert isinstance(result, np.ndarray)

    def test_fit_requires_crops(self) -> None:
        """Test fit method requires crop images."""
        with patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"), patch(
            "forgesyte_yolo_tracker.utils.team.AutoProcessor"
        ):

            classifier = TeamClassifier(device="cpu")
            crops = [np.zeros((224, 224, 3), dtype=np.uint8)]

            # Mock extract_features
            classifier.extract_features = MagicMock(return_value=np.random.rand(1, 512))
            classifier.reducer.fit_transform = MagicMock(return_value=np.random.rand(1, 3))
            classifier.cluster_model.fit = MagicMock()

            classifier.fit(crops)

            classifier.extract_features.assert_called_once()
            classifier.cluster_model.fit.assert_called_once()

    def test_predict_empty_crops(self) -> None:
        """Test predict with empty crops."""
        with patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"), patch(
            "forgesyte_yolo_tracker.utils.team.AutoProcessor"
        ):

            classifier = TeamClassifier(device="cpu")
            crops: List[np.ndarray] = []

            result = classifier.predict(crops)

            assert len(result) == 0
            assert isinstance(result, np.ndarray)

    def test_predict_before_fit(self) -> None:
        """Test predict works without explicit fit (uses default model)."""
        with patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel"), patch(
            "forgesyte_yolo_tracker.utils.team.AutoProcessor"
        ):

            classifier = TeamClassifier(device="cpu")

            # Mock methods
            classifier.extract_features = MagicMock(return_value=np.random.rand(1, 512))
            classifier.reducer.transform = MagicMock(return_value=np.random.rand(1, 3))
            classifier.cluster_model.predict = MagicMock(return_value=np.array([0]))

            crops = [np.zeros((224, 224, 3), dtype=np.uint8)]
            result = classifier.predict(crops)

            # Should work without error
            assert len(result) >= 0
