"""Entry point: ``python -m app.main``.

Starts the local web application.  Defaults bind to 127.0.0.1 so the server is
reachable only from this machine; exposing it on the network is an explicit
decision the operator has to make with ``--host``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import AppConfig
from .errors import AnalyzerError
from .logging_setup import configure_logging, get_logger
from .web.routes import create_app

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-analyzer",
        description="Local video analysis: Zahn cup timing and factory activity detection.",
    )
    parser.add_argument("--host", help="Interface to bind (default 127.0.0.1, this machine only)")
    parser.add_argument("--port", type=int, help="Port to listen on (default 8000)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Where uploads, logs and diagnostics are stored (default ~/.vision_analyzer)",
    )
    parser.add_argument(
        "--max-upload-mb", type=int, help="Maximum accepted upload size in megabytes"
    )
    parser.add_argument(
        "--save-diagnostics",
        action="store_true",
        help="Write ROI crops and event-boundary frames for tuning and debugging",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default INFO)",
    )
    return parser


def config_from_args(argv: list[str] | None = None) -> AppConfig:
    """Environment variables provide the defaults; command-line flags win."""
    args = build_parser().parse_args(argv)
    config = AppConfig.from_env()

    if args.host:
        config = replace(config, host=args.host)
    if args.port:
        config = replace(config, port=args.port)
    if args.data_dir:
        config = replace(config, data_dir=args.data_dir.expanduser())
    if args.max_upload_mb:
        config = replace(config, max_upload_mb=args.max_upload_mb)
    if args.save_diagnostics:
        config = replace(config, save_diagnostics=True)
    if args.debug:
        config = replace(config, debug=True)
    if args.log_level:
        config = replace(config, log_level=args.log_level)

    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    try:
        config = config_from_args(argv)
    except AnalyzerError as exc:
        print(f"Configuration error: {exc.user_message}", file=sys.stderr)
        return 2

    configure_logging(config.log_level, log_dir=config.data_dir)
    app = create_app(config)

    url = f"http://{config.host}:{config.port}/"
    logger.info("Video Analyzer listening on %s", url)
    print(f"\n  Video Analyzer is running.\n  Open {url} in your browser.\n")

    # threaded=True so progress polling is served while an analysis runs.
    app.run(
        host=config.host, port=config.port, debug=config.debug, threaded=True, use_reloader=False
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
