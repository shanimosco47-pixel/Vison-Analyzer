"""Robot / machine activity in long factory recordings.

This is the first *plant-oriented* detector, and it exists as much to prove the
extension point as to be useful: it is a thin subclass of
:class:`~app.analysis.coarse_to_fine.CoarseToFineDetector` that adds naming and
one derived event type.  A future dip detector, cycle counter or
operator-intervention detector is written the same way, and none of them
require touching the reader, the scanner, the event log or the web layer.

What it reports today, and only what it can actually evidence:

*   **Machine active** - a period in which the watched region genuinely
    changed, with accurate boundaries from the dense pass.
*   **Extended pause** - a long gap *between* two active periods.  On a
    repetitive production line an unexpected quiet spell is usually the thing
    the engineer is looking for, and it is derivable with no extra decoding.

It deliberately does *not* claim to recognise "dip start", "cycle finished" or
"operator intervention" yet: those need either plant-specific geometry (which
the ROI mechanism already supports) or object detection, and inventing labels
the software cannot substantiate would be worse than useless.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config import apply_overrides
from ..errors import ConfigurationError
from ..logging_setup import get_logger
from .base_detector import Event, EventStatus
from .candidate_refinement import RefinedEvent
from .coarse_to_fine import CoarseToFineDetector
from .temporal import clamp

logger = get_logger(__name__)


class RobotActivityConfig:
    """Settings specific to the robot/machine mode.

    Kept as a small mutable object (rather than a frozen dataclass) so the
    generic ``apply_overrides`` helper can populate it from the UI's flat
    parameter bag.
    """

    # A quiet period longer than this *between* two detected active periods is
    # reported as an extended pause.  Default 5 minutes: long enough that a
    # normal inter-cycle wait on most lines does not trigger it.  Set to 0 to
    # switch the derived pause events off.
    idle_pause_s: float = 300.0

    # An active period longer than this is flagged in its label, because on a
    # cyclic machine an unusually long run usually means something jammed.
    long_run_s: float = 0.0  # 0 = disabled until the user knows their cycle time

    def __init__(self) -> None:
        self.idle_pause_s = RobotActivityConfig.idle_pause_s
        self.long_run_s = RobotActivityConfig.long_run_s

    def validate(self) -> None:
        if self.idle_pause_s < 0 or self.long_run_s < 0:
            raise ConfigurationError("Pause and run-length settings must not be negative.")


class RobotActivityDetector(CoarseToFineDetector):
    """Mode 2 in the UI: robot / machine activity in long recordings."""

    name: ClassVar[str] = "robot_activity"
    display_name: ClassVar[str] = "Robot / machine activity"
    description: ClassVar[str] = (
        "Finds the periods in which a machine or robot was working, and the "
        "unexpected quiet spells between them."
    )
    default_label: ClassVar[str] = "Machine active"

    def configure(self) -> None:
        super().configure()
        self.robot_config = apply_overrides(RobotActivityConfig(), self.params, ignore={"roi"})
        self.robot_config.validate()

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "idle_pause_s": self.robot_config.idle_pause_s,
                "long_run_s": self.robot_config.long_run_s,
            }
        )
        return description

    def label_for(self, event: RefinedEvent) -> str:
        long_run = self.robot_config.long_run_s
        if long_run and event.interval.duration_s > long_run:
            return "Machine active (unusually long)"
        return self.default_label

    def derive_additional_events(self, events: list[Event]) -> list[Event]:
        """Report long quiet spells between two detected active periods.

        Only gaps *between* activity are reported: the quiet before the first
        event and after the last one are not evidence of a stoppage, they are
        simply the parts of the recording outside the machine's working span.
        """
        threshold = self.robot_config.idle_pause_s
        if not threshold or len(events) < 2:
            return []

        active = sorted(
            (e for e in events if e.detector == self.name and e.label.startswith("Machine active")),
            key=lambda e: e.start_s,
        )
        pauses: list[Event] = []
        for previous, following in zip(active, active[1:], strict=False):
            gap = following.start_s - previous.end_s
            if gap < threshold:
                continue
            # A pause is only as trustworthy as the two detections that bound
            # it, so its confidence is the weaker of the two.
            confidence = clamp(min(previous.confidence, following.confidence), 0.05, 0.95)
            pauses.append(
                Event(
                    label="Extended pause",
                    start_s=previous.end_s,
                    end_s=following.start_s,
                    confidence=confidence,
                    detector=self.name,
                    status=(
                        EventStatus.CONFIRMED
                        if confidence >= self.refine_config.review_confidence
                        else EventStatus.REVIEW
                    ),
                    notes=("No machine activity was detected between two working periods.",),
                    details={
                        "gap_seconds": round(gap, 3),
                        "threshold_seconds": threshold,
                        "bounded_by": [
                            round(previous.end_s, 3),
                            round(following.start_s, 3),
                        ],
                    },
                )
            )
        if pauses:
            logger.info("Derived %d extended pause event(s)", len(pauses))
        return pauses
