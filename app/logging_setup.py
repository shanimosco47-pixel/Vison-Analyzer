"""Standard-library logging configuration.

The application logs enough to explain *why* a detection happened: file
metadata, the resolved algorithm configuration, sampling rate, candidate
counts, timings and failures.  Nothing here writes outside the machine.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_configured = False


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure root logging once, optionally adding a rotating file log."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "vision_analyzer.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
            root.addHandler(file_handler)
        except OSError as exc:  # e.g. read-only or full disk: console still works
            root.warning("File logging disabled (%s)", exc)

    # OpenCV/urllib chatter is not useful at INFO.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
