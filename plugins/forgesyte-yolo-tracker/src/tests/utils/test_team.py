"""Tests for team classification utilities - fast tests without model loading."""

from typing import List

from forgesyte_yolo_tracker.utils.team import create_batches


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

        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")
