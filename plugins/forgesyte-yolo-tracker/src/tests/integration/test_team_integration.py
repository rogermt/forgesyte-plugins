"""Integration tests with real SiglipVisionModel.

Run with: RUN_INTEGRATION_TESTS=1 uv run pytest src/tests/integration/ -v
"""

import os
import numpy as np
import pytest

from forgesyte_yolo_tracker.utils import TeamClassifier

RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_INTEGRATION_TESTS,
    reason="Set RUN_INTEGRATION_TESTS=1 to run (requires network for model loading)",
)


class TestTeamClassifierIntegration:
    """Integration tests with real SiglipVisionModel."""

    def test_real_model_initialization(self) -> None:
        """Test TeamClassifier initializes with real model."""
        classifier = TeamClassifier(device="cpu")
        assert classifier.device == "cpu"
        assert classifier.features_model is not None
        assert classifier.processor is not None

    def test_real_model_embedding_output_shape(self) -> None:
        """Verify model outputs correct embedding shape (batch, 768)."""
        classifier = TeamClassifier(device="cpu")
        crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
        embeddings = classifier.extract_features(crops)
        assert embeddings.shape[1] == 768

    def test_full_fit_predict_pipeline(self) -> None:
        """Test complete fit→predict workflow with real model."""
        classifier = TeamClassifier(device="cpu")
        team_a_crops = [np.full((224, 224, 3), 255, dtype=np.uint8) for _ in range(3)]
        team_b_crops = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]
        all_crops = team_a_crops + team_b_crops

        classifier.fit(all_crops)
        predictions = classifier.predict(all_crops)

        assert len(predictions) == 6
        assert all(p in [0, 1] for p in predictions)

    def test_batch_inference_performance(self) -> None:
        """Test batch processing with real model."""
        classifier = TeamClassifier(device="cpu", batch_size=8)
        crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]

        embeddings = classifier.extract_features(crops)
        assert embeddings.shape[0] == 10

    def test_model_on_gpu(self) -> None:
        """Test model inference on GPU (requires CUDA)."""
        pytest.importorskip("torch")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        classifier = TeamClassifier(device="cuda")
        crops = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
        embeddings = classifier.extract_features(crops)
        assert embeddings.shape[1] == 768


class TestTeamClassifierModelCaching:
    """Tests for model caching behavior."""

    def test_model_reuse_on_multiple_instances(self) -> None:
        """Verify model can be reused across instances."""
        classifier1 = TeamClassifier(device="cpu")
        classifier2 = TeamClassifier(device="cpu")

        assert classifier1.features_model is not None
        assert classifier2.features_model is not None
