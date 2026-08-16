"""Event log formatting, wall-clock mapping and CSV export."""

from __future__ import annotations

import csv
import io
from datetime import datetime

import pytest

from app.analysis.base_detector import Event, EventStatus
from app.errors import ConfigurationError
from app.services.event_log import (
    CSV_COLUMNS,
    EventLog,
    build_event_log,
    format_duration,
    format_video_timestamp,
    parse_recording_start,
    summarise,
    to_wall_clock,
)


def make_event(
    label="Robot moving",
    start=10.0,
    end=33.0,
    confidence=0.98,
    status=EventStatus.CONFIRMED,
    notes=(),
):
    return Event(
        label=label,
        start_s=start,
        end_s=end,
        confidence=confidence,
        detector="test",
        status=status,
        notes=tuple(notes),
    )


class TestFormatting:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0:00:00.000"),
            (5.25, "0:00:05.250"),
            (61.5, "0:01:01.500"),
            (3661.25, "1:01:01.250"),
            (-3.0, "0:00:00.000"),  # never report a negative video time
        ],
    )
    def test_video_timestamp(self, seconds, expected):
        assert format_video_timestamp(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (1.9999, "0:00:02.000"),  # carry inside the minute
            (59.9999, "0:01:00.000"),  # carry across the minute
            (119.9999, "0:02:00.000"),  # carry across a later minute
            (3599.9999, "1:00:00.000"),  # carry across the hour
            (59.9994, "0:00:59.999"),  # just below: must not carry
            (0.0, "0:00:00.000"),
        ],
    )
    def test_rounding_carries_across_every_boundary(self, seconds, expected):
        """Rounding milliseconds up must propagate into minutes and hours.

        Regression: the carry used to increment the seconds field alone, so
        59.9999 s was rendered as the impossible "0:00:60.000".
        """
        assert format_video_timestamp(seconds) == expected

    def test_no_timestamp_field_ever_overflows(self):
        """No input may produce a minutes or seconds field of 60 or more."""
        for milli in range(0, 7_200_000, 999):  # two hours in ~1 s steps
            text = format_video_timestamp(milli / 1000)
            _, minutes, rest = text.split(":")
            assert int(minutes) < 60, text
            assert float(rest) < 60.0, text

    def test_without_milliseconds(self):
        assert format_video_timestamp(65.4, milliseconds=False) == "0:01:05"

    def test_without_milliseconds_truncates_rather_than_carrying(self):
        """Dropping the fraction must not round a time up into the next minute."""
        assert format_video_timestamp(59.6, milliseconds=False) == "0:00:59"

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(23.0, "23.0 s"), (95.0, "1 m 35.0 s"), (3725.0, "1 h 02 m 05.0 s")],
    )
    def test_duration(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestWallClock:
    def test_offset_applied(self):
        start = datetime(2026, 3, 4, 7, 43, 0)
        assert to_wall_clock(start, 18.0) == datetime(2026, 3, 4, 7, 43, 18)

    def test_unknown_recording_start_gives_none(self):
        """Elapsed video time is never presented as a clock time by guessing."""
        assert to_wall_clock(None, 18.0) is None

    @pytest.mark.parametrize(
        "raw",
        ["2026-03-04T07:43:00", "2026-03-04 07:43:00", "2026-03-04 07:43"],
    )
    def test_parsing_accepted_formats(self, raw):
        assert parse_recording_start(raw) == datetime(2026, 3, 4, 7, 43, 0)

    @pytest.mark.parametrize("raw", ["", None, "yesterday morning", "04/03/2026"])
    def test_unparseable_input_yields_none(self, raw):
        assert parse_recording_start(raw) is None


class TestEventLog:
    def test_rows_are_sorted_by_start(self):
        log = EventLog(events=[make_event(start=50, end=60), make_event(start=10, end=20)])
        starts = [row["start_video_time"] for row in log.rows()]
        assert starts == sorted(starts)

    def test_clock_columns_empty_without_a_recording_start(self):
        log = EventLog(events=[make_event()])
        row = log.rows()[0]
        assert row["start_wall_clock"] == ""
        assert row["start_video_time"] == "0:00:10.000"
        assert log.has_wall_clock is False

    def test_clock_columns_filled_when_supplied(self):
        log = EventLog(
            events=[make_event(start=18.0, end=41.0)],
            recording_start=datetime(2026, 3, 4, 7, 43, 0),
        )
        row = log.rows()[0]
        assert row["start_wall_clock"] == "2026-03-04 07:43:18"
        assert row["end_wall_clock"] == "2026-03-04 07:43:41"

    def test_duration_and_confidence_columns(self):
        log = EventLog(events=[make_event(start=10.0, end=33.0, confidence=0.982)])
        row = log.rows()[0]
        assert row["duration_s"] == 23.0
        assert row["confidence_pct"] == 98

    def test_review_status_is_visible(self):
        log = EventLog(events=[make_event(status=EventStatus.REVIEW, confidence=0.5)])
        assert log.rows()[0]["status"] == "review recommended"

    def test_notes_are_joined(self):
        log = EventLog(events=[make_event(notes=("first", "second"))])
        assert log.rows()[0]["notes"] == "first | second"


class TestCsvExport:
    def test_csv_has_the_expected_header(self):
        log = EventLog(events=[make_event()])
        reader = csv.reader(io.StringIO(log.to_csv()))
        assert next(reader) == list(CSV_COLUMNS)

    def test_csv_round_trips(self):
        log = EventLog(
            events=[make_event(label="Dip cycle", start=221.0, end=239.0)],
            recording_start=datetime(2026, 3, 4, 7, 40, 0),
        )
        rows = list(csv.DictReader(io.StringIO(log.to_csv())))
        assert len(rows) == 1
        assert rows[0]["event"] == "Dip cycle"
        assert rows[0]["start_video_time"] == "0:03:41.000"
        assert rows[0]["start_wall_clock"] == "2026-03-04 07:43:41"
        assert rows[0]["duration_s"] == "18.0"

    def test_empty_log_still_produces_a_header(self):
        rows = list(csv.reader(io.StringIO(EventLog().to_csv())))
        assert rows[0] == list(CSV_COLUMNS)
        assert len(rows) == 1

    def test_suggested_filename_is_safe(self):
        class FakeVideo:
            from pathlib import Path as _Path

            path = _Path("../../etc/pass wd;rm -rf.mp4")

        log = EventLog(events=[], video=FakeVideo(), mode="zahn_cup")
        name = log.suggested_filename()
        assert "/" not in name and ".." not in name and ";" not in name
        assert name.endswith("_zahn_cup_events.csv")

    def test_dataframe_export(self):
        log = EventLog(events=[make_event()])
        frame = log.to_dataframe()
        assert list(frame.columns) == list(CSV_COLUMNS)
        assert len(frame) == 1


class TestSummarise:
    def test_counts_by_status(self):
        events = [
            make_event(),
            make_event(status=EventStatus.REVIEW, confidence=0.4),
            make_event(status=EventStatus.REVIEW, confidence=0.4),
        ]
        counts = summarise(events)
        assert counts == {
            "total": 3,
            "confirmed": 1,
            "review": 2,
            "failed": 0,
            "total_active_seconds": 69.0,
        }


class TestEventValidation:
    def test_event_cannot_end_before_it_starts(self):
        with pytest.raises(ConfigurationError):
            make_event(start=10.0, end=5.0)

    def test_confidence_outside_zero_to_one_rejected(self):
        with pytest.raises(ConfigurationError):
            make_event(confidence=1.5)

    def test_build_event_log_helper(self):
        log = build_event_log([make_event()], None, mode="motion_scan")
        assert log.mode == "motion_scan"
        assert len(log.events) == 1
