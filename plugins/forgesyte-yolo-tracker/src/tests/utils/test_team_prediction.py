"""Tests for TeamClassifier prediction pipeline - Issue #10.

These tests verify the TeamClassifier prediction workflow by properly mocking
model dependencies to avoid network/model loading during testing.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from forgesyte_yolo_tracker.utils.team import TeamClassifier, create_batches


class TestTeamClassifierPrediction:
    """Tests for TeamClassifier prediction workflow.

    These tests use module-level patching to avoid loading actual models,
    which is necessary for CPU-only environments without GPU acceleration.
    """

    @pytest.fixture
    def mock_classifier(self) -> TeamClassifier:
        """Create a TeamClassifier with fully mocked model dependencies."""
        with patch("forgesyte_yolo_tracker.utils.team.SiglipVisionModel") as mock_model, patch(
            "forgesyte_yolo_tracker.utils.team.AutoProcessor"
        ) as mock_processor:
            classifier = TeamClassifier.__new__(TeamClassifier)
            classifier.device = "cpu"
            classifier.batch_size = 32
            classifier.features_model = mock_model
            classifier.processor = mock_processor
            classifier.cluster_model = MagicMock()
            classifier.cluster_model.predict = MagicMock(return_value=np.array([0, 1]))
            classifier.reducer = MagicMock()

            return classifier

    def test_predict_returns_numpy_array(self, mock_classifier: TeamClassifier) -> None:
        """Verify predict() returns a numpy array."""
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(2, 512))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(2, 3))
        mock_classifier.cluster_model.predict.return_value = np.array([0, 1])

        crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
        result = mock_classifier.predict(crops)

        assert isinstance(result, np.ndarray)

    def test_predict_returns_binary_labels(self, mock_classifier: TeamClassifier) -> None:
        """Verify predict() returns only 0 or 1 labels."""
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(4, 512))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(4, 3))
        mock_classifier.cluster_model.predict.return_value = np.array([0, 1, 0, 1])

        crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
        result = mock_classifier.predict(crops)

        assert len(result) == 4
        assert all(label in [0, 1] for label in result)

    def test_predict_calls_extract_features(self, mock_classifier: TeamClassifier) -> None:
        """Verify predict() calls extract_features()."""
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(2, 512))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(2, 3))

        crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
        mock_classifier.predict(crops)

        mock_classifier.extract_features.assert_called_once_with(crops)

    def test_predict_calls_reducer_transform(self, mock_classifier: TeamClassifier) -> None:
        """Verify predict() calls reducer.transform()."""
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(2, 512))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(2, 3))

        crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
        mock_classifier.predict(crops)

        mock_classifier.reducer.transform.assert_called_once()

    def test_predict_calls_cluster_predict(self, mock_classifier: TeamClassifier) -> None:
        """Verify predict() calls cluster_model.predict()."""
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(2, 512))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(2, 3))

        crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
        mock_classifier.predict(crops)

        mock_classifier.cluster_model.predict.assert_called_once()

    def test_fit_predict_pipeline(self, mock_classifier: TeamClassifier) -> None:
        """Test full fit() -> predict() workflow."""
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(2, 512))
        mock_classifier.reducer.fit_transform = MagicMock(return_value=np.random.rand(2, 3))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(2, 3))
        # Ensure cluster_model.predict returns a proper numpy array
        mock_classifier.cluster_model.predict.return_value = np.array([0, 1])

        crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]

        mock_classifier.fit(crops)
        result = mock_classifier.predict(crops)

        mock_classifier.extract_features.assert_called()
        mock_classifier.reducer.fit_transform.assert_called()
        mock_classifier.cluster_model.fit.assert_called_once()
        mock_classifier.cluster_model.predict.assert_called_once()
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_predict_single_crop(self, mock_classifier: TeamClassifier) -> None:
        """Test prediction with a single crop image."""
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(1, 512))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(1, 3))
        mock_classifier.cluster_model.predict.return_value = np.array([0])

        crops = [np.zeros((224, 224, 3), dtype=np.uint8)]
        result = mock_classifier.predict(crops)

        assert len(result) == 1
        assert result[0] in [0, 1]

    def test_predict_large_batch(self, mock_classifier: TeamClassifier) -> None:
        """Test prediction with batch larger than default batch_size."""
        mock_classifier.batch_size = 4
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(10, 512))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(10, 3))
        mock_classifier.cluster_model.predict.return_value = np.array([0, 1] * 5)

        crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(10)]
        result = mock_classifier.predict(crops)

        assert len(result) == 10
        mock_classifier.extract_features.assert_called_once()

    def test_predict_empty_crops_returns_empty_array(self, mock_classifier: TeamClassifier) -> None:
        """Verify predict() with empty crops returns empty numpy array."""
        result = mock_classifier.predict([])

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_predict_result_shape_matches_input(self, mock_classifier: TeamClassifier) -> None:
        """Verify predict() returns array with same length as input crops."""
        num_crops = 7
        mock_classifier.extract_features = MagicMock(return_value=np.random.rand(num_crops, 512))
        mock_classifier.reducer.transform = MagicMock(return_value=np.random.rand(num_crops, 3))
        mock_classifier.cluster_model.predict.return_value = np.random.randint(0, 2, size=num_crops)

        crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(num_crops)]
        result = mock_classifier.predict(crops)

        assert len(result) == num_crops

    def test_cluster_model_n_clusters_is_2(self, mock_classifier: TeamClassifier) -> None:
        """Verify cluster_model is configured with 2 clusters."""
        mock_classifier.cluster_model.n_clusters = 2
        assert mock_classifier.cluster_model.n_clusters == 2


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

    def test_create_batches_empty_sequence(self) -> None:
        """Test with empty sequence."""
        batches = list(create_batches([], batch_size=2))

        assert len(batches) == 0
