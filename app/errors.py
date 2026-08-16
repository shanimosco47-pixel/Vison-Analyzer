"""Domain errors.

Every error raised deliberately by the application carries a short,
human-readable message suitable for display in the browser.  Technical
detail belongs in the log, never in the message shown to the operator.
"""

from __future__ import annotations


class AnalyzerError(Exception):
    """Base class for all expected (non-bug) failures.

    Attributes:
        user_message: Text safe to show a non-technical user.
        http_status: Status code the web layer should use.
    """

    http_status: int = 400
    default_message = "The request could not be completed."

    def __init__(self, user_message: str | None = None, *, detail: str | None = None) -> None:
        self.user_message = user_message or self.default_message
        self.detail = detail
        super().__init__(self.user_message if detail is None else f"{self.user_message} ({detail})")


class VideoOpenError(AnalyzerError):
    """The file could not be opened as a video (bad codec, corrupt, not a video)."""

    default_message = (
        "This file could not be opened as a video. It may be corrupted or use an unsupported codec."
    )


class VideoMetadataError(AnalyzerError):
    """The video opened but essential metadata (FPS, size) is missing or nonsensical."""

    default_message = "The video is missing frame-rate information, so timings cannot be trusted."


class EmptyVideoError(AnalyzerError):
    """The video contains no decodable frames."""

    default_message = "The video contains no readable frames."


class UnsupportedFormatError(AnalyzerError):
    """Upload rejected before it was ever opened."""

    default_message = "That file type is not supported."


class UploadTooLargeError(AnalyzerError):
    http_status = 413
    default_message = "The uploaded file is larger than the configured limit."


class StorageError(AnalyzerError):
    http_status = 507
    default_message = "The file could not be stored on disk (the disk may be full)."


class InvalidROIError(AnalyzerError):
    """The requested region of interest does not fit inside the frame."""

    default_message = "The selected region is outside the video frame."


class ConfigurationError(AnalyzerError):
    """A configuration value is out of range or internally inconsistent."""

    default_message = "One of the analysis settings is invalid."


class NotFoundError(AnalyzerError):
    http_status = 404
    default_message = "The requested item no longer exists."


class ModelUnavailableError(AnalyzerError):
    """An optional neural model was requested but is not installed."""

    http_status = 501
    default_message = "The optional detection model is not installed on this machine."
