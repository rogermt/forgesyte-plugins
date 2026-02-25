"""Unit tests for ByteTrackFactory - API adaptive ByteTrack construction.

Tests that ByteTrackFactory correctly handles both old and new supervision APIs.
"""

import inspect
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

    def test_build_with_old_api(self) -> None:
        """Verify _build() uses old API when track_thresh exists."""
        from forgesyte_yolo_tracker import tracking
        
        mock_bytetrack = MagicMock()
        mock_instance = MagicMock()
        mock_bytetrack.return_value = mock_instance
        
        # Create a mock signature with old API params
        mock_sig = MagicMock()
        mock_sig.parameters = {'track_thresh': None, 'track_buffer': None, 'match_thresh': None, 'frame_rate': None}
        
        with patch.object(tracking, 'sv') as mock_sv:
            mock_sv.ByteTrack = mock_bytetrack
            with patch.object(inspect, 'signature', return_value=mock_sig):
                tracking.ByteTrackFactory.reset()
                tracking.ByteTrackFactory._build()
                
                # Should call with old API params
                mock_bytetrack.assert_called_once_with(
                    track_thresh=0.25,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=30,
                )
        
        tracking.ByteTrackFactory.reset()

    def test_build_with_new_api(self) -> None:
        """Verify _build() uses new API when track_activation_threshold exists."""
        from forgesyte_yolo_tracker import tracking
        
        mock_bytetrack = MagicMock()
        mock_instance = MagicMock()
        mock_bytetrack.return_value = mock_instance
        
        # Create a mock signature with new API params (no track_thresh)
        mock_sig = MagicMock()
        mock_sig.parameters = {'track_activation_threshold': None, 'lost_track_buffer': None, 'minimum_matching_threshold': None, 'frame_rate': None}
        
        with patch.object(tracking, 'sv') as mock_sv:
            mock_sv.ByteTrack = mock_bytetrack
            with patch.object(inspect, 'signature', return_value=mock_sig):
                tracking.ByteTrackFactory.reset()
                tracking.ByteTrackFactory._build()
                
                # Should call with new API params
                mock_bytetrack.assert_called_once_with(
                    track_activation_threshold=0.25,
                    lost_track_buffer=30,
                    minimum_matching_threshold=0.8,
                    frame_rate=30,
                )
        
        tracking.ByteTrackFactory.reset()

    def test_build_raises_on_unsupported_signature(self) -> None:
        """Verify RuntimeError when neither API is recognized."""
        from forgesyte_yolo_tracker import tracking
        
        mock_bytetrack = MagicMock()
        
        # Create a mock signature with unknown params
        mock_sig = MagicMock()
        mock_sig.parameters = {'unknown_param': None}
        
        with patch.object(tracking, 'sv') as mock_sv:
            mock_sv.ByteTrack = mock_bytetrack
            with patch.object(inspect, 'signature', return_value=mock_sig):
                tracking.ByteTrackFactory.reset()
                with pytest.raises(RuntimeError, match="Unsupported ByteTrack"):
                    tracking.ByteTrackFactory._build()
        
        tracking.ByteTrackFactory.reset()
