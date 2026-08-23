"""Unit tests for the deterministic contact-sheet renderer (no video, no
provider, no network) - see ``app/analysis/llm_timing/contact_sheet.py``."""

from __future__ import annotations

import numpy as np

from app.analysis.llm_timing import contact_sheet as cs


def _frame(width: int = 640, height: int = 480, fill: int = 200) -> np.ndarray:
    return np.full((height, width, 3), fill, dtype=np.uint8)


def test_default_crop_origin_is_centered_horizontally_and_near_the_top() -> None:
    x, y = cs.default_crop_origin(640, 480)
    assert x == (640 - cs.SOURCE_CROP_WIDTH_PX) // 2
    assert y == round(480 * 0.10)
    # Always fits inside the frame.
    assert 0 <= x <= 640 - cs.SOURCE_CROP_WIDTH_PX
    assert 0 <= y <= 480 - cs.SOURCE_CROP_HEIGHT_PX


def test_default_crop_origin_clamps_for_a_frame_smaller_than_the_crop() -> None:
    x, y = cs.default_crop_origin(100, 80)
    assert x == 0
    assert y == 0


def test_render_panel_has_the_exact_tile_dimensions() -> None:
    tile = cs.render_panel(_frame(), timestamp_s=1.5, crop_origin=(50, 20))
    assert tile.shape == (cs.RENDER_TILE_HEIGHT_PX, cs.RENDER_TILE_WIDTH_PX, 3)


def test_render_panel_never_raises_on_an_undersized_frame() -> None:
    tiny = _frame(width=50, height=40)
    tile = cs.render_panel(tiny, timestamp_s=0.0, crop_origin=(0, 0))
    assert tile.shape == (cs.RENDER_TILE_HEIGHT_PX, cs.RENDER_TILE_WIDTH_PX, 3)


def test_render_panel_draws_a_black_header_band() -> None:
    tile = cs.render_panel(_frame(fill=255), timestamp_s=2.0, crop_origin=(0, 0))
    header_row = tile[5, :, :]
    # The header band is solid black except where the timestamp text is
    # drawn (white) - most of the row must be black, not the frame's own
    # (255,255,255) fill colour.
    black_pixels = np.all(header_row == (0, 0, 0), axis=-1)
    assert black_pixels.mean() > 0.5


def test_render_panel_draws_a_red_outlet_ring() -> None:
    tile = cs.render_panel(_frame(fill=255), timestamp_s=0.0, crop_origin=(0, 0))
    x, y = cs.OUTLET_MARKER_XY
    ring_x = x + cs.OUTLET_MARKER_RADIUS_PX
    pixel = tile[y, ring_x]
    # BGR - red is (0, 0, 255).
    assert pixel[2] > pixel[0] and pixel[2] > pixel[1]


def test_render_panel_draws_no_full_width_horizontal_line() -> None:
    """The overlay contract explicitly forbids full-width lines (only short
    15px scale ticks) - a full-width row of a single non-background colour
    anywhere below the header would violate that."""
    tile = cs.render_panel(_frame(fill=200), timestamp_s=0.0, crop_origin=(0, 0))
    # cv2.rectangle's corners are both inclusive, so the header (spec: "y=0..35")
    # occupies rows 0 through HEADER_HEIGHT_PX inclusive - start just past it.
    for row in tile[cs.HEADER_HEIGHT_PX + 1 :]:
        unique_colors = {tuple(px) for px in row}
        if len(unique_colors) == 1:
            (only_color,) = unique_colors
            assert only_color == (200, 200, 200), (
                "a full-width line was drawn where only short ticks are allowed"
            )


def test_compose_grid_3x3_places_nine_tiles_row_major() -> None:
    tiles = [
        cs.render_panel(_frame(fill=v), timestamp_s=float(v), crop_origin=(0, 0))
        for v in range(10, 100, 10)
    ]
    grid = cs.compose_grid(tiles, columns=3, rows=3)
    assert grid.shape == (cs.RENDER_TILE_HEIGHT_PX * 3, cs.RENDER_TILE_WIDTH_PX * 3, 3)


def test_compose_grid_refine_layout_leaves_one_blank_cell() -> None:
    tiles = [
        cs.render_panel(_frame(fill=v), timestamp_s=float(v), crop_origin=(0, 0))
        for v in range(10, 80, 10)
    ]
    assert len(tiles) == 7
    grid = cs.compose_grid(tiles, columns=4, rows=2)
    assert grid.shape == (cs.RENDER_TILE_HEIGHT_PX * 2, cs.RENDER_TILE_WIDTH_PX * 4, 3)
    # The 8th cell (row 1, col 3) was never written - stays solid black.
    blank_cell = grid[
        cs.RENDER_TILE_HEIGHT_PX : cs.RENDER_TILE_HEIGHT_PX * 2,
        cs.RENDER_TILE_WIDTH_PX * 3 : cs.RENDER_TILE_WIDTH_PX * 4,
    ]
    assert np.all(blank_cell == 0)


def test_compose_grid_rejects_more_tiles_than_cells() -> None:
    tiles = [cs.render_panel(_frame(), timestamp_s=0.0, crop_origin=(0, 0)) for _ in range(5)]
    try:
        cs.compose_grid(tiles, columns=2, rows=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError for too many tiles")
