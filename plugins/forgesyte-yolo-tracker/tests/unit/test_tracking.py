"""Unit tests for ByteTrackFactory - API adaptive ByteTrack construction.

Tests that ByteTrackFactory correctly handles both old and new supervision APIs
using try-except fallback strategy.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestByteTrackFactory:
    """Tests for ByteTrackFactory singleton and API adaptation."""

    def test_get_returns_bytetrack_instance(self) -> None:
        """Verify get() returns a ByteTrack instance."""
        from forgesyte_yolo_tracker.tracking import ByteTrackFactory

        # Reset singleton for clean test
        ByteTrackFactory.reset()

        tracker = ByteTrackFactory.get()

        assert tracker is not None
        ByteTrackFactory.reset()

    def test_get_returns_singleton(self) -> None:
        """Verify get() returns the same instance on multiple calls."""
        from forgesyte_yolo_tracker.tracking import ByteTrackFactory

        ByteTrackFactory.reset()

        tracker1 = ByteTrackFactory.get()
        tracker2 = ByteTrackFactory.get()

        assert tracker1 is tracker2
        ByteTrackFactory.reset()

    def test_reset_clears_instance(self) -> None:
        """Verify reset() clears the singleton."""
        from forgesyte_yolo_tracker.tracking import ByteTrackFactory

        tracker1 = ByteTrackFactory.get()
        ByteTrackFactory.reset()
        tracker2 = ByteTrackFactory.get()

        assert tracker1 is not tracker2
        ByteTrackFactory.reset()

    def test_build_with_new_api(self) -> None:
        """Verify _build() uses NEW API when it succeeds (try-except strategy)."""
        from forgesyte_yolo_tracker import tracking

        mock_bytetrack = MagicMock()
        mock_instance = MagicMock()
        mock_bytetrack.return_value = mock_instance

        with patch.object(tracking, 'sv') as mock_sv:
            mock_sv.ByteTrack = mock_bytetrack
            tracking.ByteTrackFactory.reset()
            tracking.ByteTrackFactory._build()

            # Should try NEW API first and succeed
            mock_bytetrack.assert_called_once_with(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )

        tracking.ByteTrackFactory.reset()

    def test_build_falls_back_to_old_api(self) -> None:
        """Verify _build() falls back to OLD API when NEW API fails."""
        from forgesyte_yolo_tracker import tracking

        mock_bytetrack = MagicMock()
        mock_instance = MagicMock()

        # First call (NEW API) fails, second call (OLD API) succeeds
        mock_bytetrack.side_effect = [
            TypeError("NEW API not available"),
            mock_instance,
        ]

        with patch.object(tracking, 'sv') as mock_sv:
            mock_sv.ByteTrack = mock_bytetrack
            tracking.ByteTrackFactory.reset()
            result = tracking.ByteTrackFactory._build()

            # Should return the instance from OLD API fallback
            assert result is mock_instance
            # Should have been called twice: NEW API (failed) + OLD API (succeeded)
            assert mock_bytetrack.call_count == 2

            # First call: NEW API
            first_call = mock_bytetrack.call_args_list[0]
            assert first_call.kwargs == {
                'track_activation_threshold': 0.25,
                'lost_track_buffer': 30,
                'minimum_matching_threshold': 0.8,
                'frame_rate': 30,
            }

            # Second call: OLD API
            second_call = mock_bytetrack.call_args_list[1]
            assert second_call.kwargs == {
                'track_thresh': 0.25,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'frame_rate': 30,
            }

        tracking.ByteTrackFactory.reset()

    def test_build_raises_on_both_apis_failing(self) -> None:
        """Verify RuntimeError when both NEW and OLD APIs fail."""
        from forgesyte_yolo_tracker import tracking

        mock_bytetrack = MagicMock()

        # Both calls fail
        mock_bytetrack.side_effect = [
            TypeError("NEW API not available"),
            TypeError("OLD API not available"),
        ]

        with patch.object(tracking, 'sv') as mock_sv:
            mock_sv.ByteTrack = mock_bytetrack
            tracking.ByteTrackFactory.reset()

            with pytest.raises(RuntimeError, match="ByteTrack construction failed"):
                tracking.ByteTrackFactory._build()

        tracking.ByteTrackFactory.reset()