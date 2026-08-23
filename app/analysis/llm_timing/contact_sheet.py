"""Deterministic rendering for the candidate-centred contact-sheet cascade.

Per the supervisor's exact overlay contract (see
``diagnostics/llm_spike/DESIGN.md`` for the full write-up and
``pipeline._run_end_validation_pass`` for how this is used): a fixed-size
crop around the outlet, resized into a labelled tile, several of those
tiles tiled chronologically into one composite "contact sheet" image - sent
to the provider as ONE image instead of many separate frames, with every
panel's own timestamp baked directly into its header rather than passed as
separate request metadata.

Pure image composition only - no network/model code, no knowledge of
:class:`~.provider.TimingProvider`. Everything here is a plain function of
``np.ndarray`` frames and floats, so it is unit-testable without a video
file or a provider stub.
"""

from __future__ import annotations

import cv2
import numpy as np

# --- Exact overlay contract (supervisor-specified) ------------------------ #
SOURCE_CROP_WIDTH_PX = 240
SOURCE_CROP_HEIGHT_PX = 360
OUTLET_IN_CROP_XY = (120, 50)

RENDER_TILE_WIDTH_PX = 360
RENDER_TILE_HEIGHT_PX = 540
_RENDER_SCALE = 1.5  # RENDER_TILE_WIDTH_PX / SOURCE_CROP_WIDTH_PX, both axes

OUTLET_MARKER_XY = (180, 75)
OUTLET_MARKER_RADIUS_PX = 14
OUTLET_MARKER_STROKE_PX = 3
_OUTLET_MARKER_COLOR_BGR = (0, 0, 255)  # red

HEADER_HEIGHT_PX = 35
_HEADER_COLOR_BGR = (0, 0, 0)  # black
_HEADER_TEXT_COLOR_BGR = (255, 255, 255)  # white

_SCALE_REACH_MAX = 300
_SCALE_REACH_STEP = 25
_SCALE_TICK_LENGTH_PX = 15
_SCALE_COLOR_BGR = (255, 0, 0)  # blue


def default_crop_origin(frame_width_px: int, frame_height_px: int) -> tuple[int, int]:
    """The crop's top-left corner within a raw source frame, when the
    caller has not supplied a calibrated one of its own.

    The supervisor's overlay contract fixes the crop's *size* (240x360) and
    the outlet's position *within* it (120, 50) - both were tuned against
    the experiment's own dev footage - but not where that crop sits within
    an arbitrary uploaded video's frame, which was never specified and
    cannot be inferred without exactly the kind of per-frame outlet
    tracking five classical mechanisms already failed at (see PR #4). This
    heuristic - horizontally centred, biased toward the upper part of the
    frame - follows the same "outlet near top-centre, stream flows down"
    framing convention this app's own classical-ROI defaults already
    assume (``AppConfig.roi_height_fraction``/``outlet_band_fraction`` in
    ``app/config.py``), clamped so the fixed-size crop always fits inside
    the frame. It is a documented assumption, not a calibrated value -
    flagged as such in this round's PR write-up for confirmation/tuning
    against real footage."""
    x = (frame_width_px - SOURCE_CROP_WIDTH_PX) // 2
    y = round(frame_height_px * 0.10)
    x = max(0, min(x, max(0, frame_width_px - SOURCE_CROP_WIDTH_PX)))
    y = max(0, min(y, max(0, frame_height_px - SOURCE_CROP_HEIGHT_PX)))
    return x, y


def render_panel(
    frame: np.ndarray, *, timestamp_s: float, crop_origin: tuple[int, int]
) -> np.ndarray:
    """Crop, resize and overlay one source frame into one contact-sheet
    panel tile, per the exact overlay contract above.

    Never raises on an undersized frame (a tiny synthetic test fixture, for
    instance) - the crop is clamped to what is actually available and
    zero-padded to the full 240x360 size before resizing, so the returned
    tile is always exactly ``RENDER_TILE_WIDTH_PX``x``RENDER_TILE_HEIGHT_PX``.
    """
    height, width = frame.shape[:2]
    x0, y0 = crop_origin
    x0 = max(0, min(x0, max(0, width - 1)))
    y0 = max(0, min(y0, max(0, height - 1)))
    x1 = min(width, x0 + SOURCE_CROP_WIDTH_PX)
    y1 = min(height, y0 + SOURCE_CROP_HEIGHT_PX)
    crop = frame[y0:y1, x0:x1]
    if crop.shape[0] != SOURCE_CROP_HEIGHT_PX or crop.shape[1] != SOURCE_CROP_WIDTH_PX:
        padded = np.zeros((SOURCE_CROP_HEIGHT_PX, SOURCE_CROP_WIDTH_PX, 3), dtype=frame.dtype)
        padded[: crop.shape[0], : crop.shape[1]] = crop
        crop = padded

    tile = cv2.resize(
        crop, (RENDER_TILE_WIDTH_PX, RENDER_TILE_HEIGHT_PX), interpolation=cv2.INTER_LINEAR
    )

    # Blue side scale (drawn before the header/marker so nothing else can
    # cover it, and before it could be mistaken for a full-width line - see
    # module docstring: only short 15px ticks, never a full-width line).
    for reach in range(0, _SCALE_REACH_MAX + 1, _SCALE_REACH_STEP):
        y = round(OUTLET_MARKER_XY[1] + _RENDER_SCALE * reach)
        if y >= RENDER_TILE_HEIGHT_PX:
            break
        cv2.line(tile, (0, y), (_SCALE_TICK_LENGTH_PX, y), _SCALE_COLOR_BGR, 1, cv2.LINE_AA)
        cv2.putText(
            tile,
            str(reach),
            (_SCALE_TICK_LENGTH_PX + 2, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            _SCALE_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )

    # Red outlet marker.
    cv2.circle(
        tile,
        OUTLET_MARKER_XY,
        OUTLET_MARKER_RADIUS_PX,
        _OUTLET_MARKER_COLOR_BGR,
        OUTLET_MARKER_STROKE_PX,
        cv2.LINE_AA,
    )

    # Black header band with the panel's own exact timestamp - this is what
    # lets the composite image be self-labelling, so no separate per-panel
    # request metadata is needed for the provider to cite a timestamp back.
    cv2.rectangle(tile, (0, 0), (RENDER_TILE_WIDTH_PX, HEADER_HEIGHT_PX), _HEADER_COLOR_BGR, -1)
    cv2.putText(
        tile,
        f"t={timestamp_s:.3f}s",
        (6, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        _HEADER_TEXT_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )
    return tile


def compose_grid(tiles: list[np.ndarray], *, columns: int, rows: int) -> np.ndarray:
    """Tile panels chronologically, row-major, into one composite image.

    ``len(tiles)`` may be less than ``columns * rows`` (the 7-panel refine
    sheet's own 4+3 layout leaves the 8th cell blank, per the overlay
    contract) - any unfilled cell stays solid black, since the canvas
    starts zeroed and is never read back.
    """
    if not tiles:
        raise ValueError("compose_grid needs at least one tile.")
    if len(tiles) > columns * rows:
        raise ValueError(
            f"{len(tiles)} tiles do not fit a {columns}x{rows} grid ({columns * rows} cells)."
        )
    tile_height, tile_width = tiles[0].shape[:2]
    canvas = np.zeros((tile_height * rows, tile_width * columns, 3), dtype=tiles[0].dtype)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        canvas[
            row * tile_height : (row + 1) * tile_height,
            col * tile_width : (col + 1) * tile_width,
        ] = tile
    return canvas
