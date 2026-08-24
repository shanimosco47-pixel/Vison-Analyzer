"""Unit tests for the deterministic contact-sheet renderer (no video, no
provider, no network) - see ``app/analysis/llm_timing/contact_sheet.py``."""

from __future__ import annotations

import numpy as np

from app.analysis.llm_timing import contact_sheet as cs


def _frame(width: int = 640, height: int = 480, fill: int = 200) -> np.ndarray:
    return np.full((height, width, 3), fill, dtype=np.uint8)


def test_crop_origin_from_outlet_positions_the_outlet_at_its_marker() -> None:
    outlet_x, outlet_y = 627.0, 1108.0
    x, y = cs.crop_origin_from_outlet(outlet_x, outlet_y)
    assert x == outlet_x - cs.OUTLET_IN_CROP_XY[0]
    assert y == outlet_y - cs.OUTLET_IN_CROP_XY[1]
    # The marked outlet, re-expressed inside the crop, lands exactly where
    # the overlay's own red ring is drawn.
    assert (outlet_x - x, outlet_y - y) == cs.OUTLET_IN_CROP_XY


def test_crop_origin_from_outlet_rounds_a_fractional_click() -> None:
    x, y = cs.crop_origin_from_outlet(100.6, 200.4)
    assert x == round(100.6) - cs.OUTLET_IN_CROP_XY[0]
    assert y == round(200.4) - cs.OUTLET_IN_CROP_XY[1]


def test_crop_origin_from_outlet_can_go_negative_near_a_frame_edge() -> None:
    # An outlet marked close to the frame's own top-left corner yields a
    # negative origin - render_panel's own clamping handles this, not this
    # function (see its docstring).
    x, y = cs.crop_origin_from_outlet(10.0, 5.0)
    assert x < 0
    assert y < 0


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


def test_a_real_marked_outlet_lands_on_the_rendered_ring() -> None:
    """End to end: a real (x, y) click, run through crop_origin_from_outlet
    and then render_panel, must land the marked point exactly on the
    overlay's own red ring - not merely somewhere inside the crop."""
    frame = _frame(width=1920, height=1080, fill=255)
    outlet_x, outlet_y = 627.0, 1108.0
    crop_origin = cs.crop_origin_from_outlet(outlet_x, outlet_y)
    tile = cs.render_panel(frame, timestamp_s=0.0, crop_origin=crop_origin)
    ring_x, ring_y = cs.OUTLET_MARKER_XY
    pixel = tile[ring_y, ring_x + cs.OUTLET_MARKER_RADIUS_PX]
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
