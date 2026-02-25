"""Shared tracking utilities for API-adaptive ByteTrack construction.

Handles supervision version compatibility:
- Older API: track_thresh, track_buffer, match_thresh
- Newer API: track_activation_threshold, lost_track_buffer, minimum_matching_threshold
"""

import inspect

import supervision as sv


class ByteTrackFactory:
    """Lazy, class-based singleton for API-adaptive ByteTrack construction.

    This factory handles the breaking API changes between supervision versions:
    - supervision < 0.23: Uses track_thresh, track_buffer, match_thresh
    - supervision >= 0.23: Uses track_activation_threshold, lost_track_buffer,
      minimum_matching_threshold

    Usage:
        tracker = ByteTrackFactory.get()

    For testing:
        ByteTrackFactory.reset()  # Clear singleton
    """

    _instance: sv.ByteTrack | None = None

    @classmethod
    def _build(cls) -> sv.ByteTrack:
        """Build ByteTrack with correct parameters for installed supervision version.

        Returns:
            sv.ByteTrack: Configured tracker instance

        Raises:
            RuntimeError: If neither old nor new API signatures are recognized
        """
        sig = inspect.signature(sv.ByteTrack)

        if "track_thresh" in sig.parameters:
            # Older supervision API (< 0.23)
            return sv.ByteTrack(
                track_thresh=0.25,
                track_buffer=30,
                match_thresh=0.8,
                frame_rate=30,
            )

        if "track_activation_threshold" in sig.parameters:
            # Newer supervision API (>= 0.23)
            return sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )

        raise RuntimeError(
            f"Unsupported ByteTrack constructor signature. "
            f"Available params: {list(sig.parameters.keys())}"
        )

    @classmethod
    def get(cls) -> sv.ByteTrack:
        """Get or create the singleton ByteTrack instance.

        Returns:
            sv.ByteTrack: Singleton tracker instance
        """
        if cls._instance is None:
            cls._instance = cls._build()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Useful for testing or when tracker state needs to be cleared.
        """
        cls._instance = None
