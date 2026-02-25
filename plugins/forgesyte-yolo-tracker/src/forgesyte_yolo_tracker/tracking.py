"""Shared tracking utilities for API-adaptive ByteTrack construction.

Handles supervision version compatibility:
- Older API: track_thresh, track_buffer, match_thresh
- Newer API: track_activation_threshold, lost_track_buffer, minimum_matching_threshold
"""

import inspect
import logging

import supervision as sv

logger = logging.getLogger(__name__)


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

        Uses a defensive try-except approach: attempts both API signatures
        and uses whichever one succeeds.

        Returns:
            sv.ByteTrack: Configured tracker instance

        Raises:
            RuntimeError: If both API signatures fail
        """
        sv_version = getattr(sv, "__version__", "unknown")
        logger.info("ByteTrackFactory._build() called")
        logger.info(f"supervision version: {sv_version}")

        # Inspect signature for logging
        try:
            sig = inspect.signature(sv.ByteTrack)
            params = list(sig.parameters.keys())
            logger.info(f"sv.ByteTrack signature params: {params}")
        except Exception as e:
            logger.warning(f"Could not inspect ByteTrack signature: {e}")
            params = []

        # STRATEGY: Try both APIs in order, use whichever works
        # This is more reliable than signature inspection which can be fooled
        # by decorators, wrappers, or __init__ overrides

        # Try NEW API first (more recent supervision versions)
        logger.info("Attempting NEW API (track_activation_threshold)...")
        try:
            tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30,
            )
            logger.info("SUCCESS: ByteTrack created with NEW API")
            return tracker
        except TypeError as e:
            logger.info(f"NEW API failed: {e}")

        # Try OLD API (older supervision versions)
        logger.info("Attempting OLD API (track_thresh)...")
        try:
            tracker = sv.ByteTrack(
                track_thresh=0.25,
                track_buffer=30,
                match_thresh=0.8,
                frame_rate=30,
            )
            logger.info("SUCCESS: ByteTrack created with OLD API")
            return tracker
        except TypeError as e:
            logger.info(f"OLD API failed: {e}")

        # Neither worked - raise comprehensive error
        error_msg = (
            f"ByteTrack construction failed with both API signatures. "
            f"supervision version: {sv_version}, "
            f"detected params: {params}. "
            f"Please check supervision installation."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    @classmethod
    def get(cls) -> sv.ByteTrack:
        """Get or create the singleton ByteTrack instance.

        Returns:
            sv.ByteTrack: Singleton tracker instance
        """
        if cls._instance is None:
            logger.debug("Creating new ByteTrack singleton instance")
            cls._instance = cls._build()
        else:
            logger.debug("Returning existing ByteTrack singleton instance")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Useful for testing or when tracker state needs to be cleared.
        """
        logger.debug("Resetting ByteTrack singleton instance")
        cls._instance = None
