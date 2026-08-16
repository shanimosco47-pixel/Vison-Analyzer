"""The generic coarse-to-fine detector.

Stage A (cheap scan of everything) followed by Stage B (dense re-analysis of
the few interesting places) is the same for every activity-style mode, so it
lives here once.  A concrete detector only has to say:

*   which scorer to use (:meth:`make_scorer`),
*   what to call the things it finds (:meth:`label_for`),
*   optionally, what extra events to derive afterwards
    (:meth:`derive_additional_events`).

That is the seam a future plant-specific detector plugs into: a dip detector,
a cycle counter, or a YOLO-backed scorer can each reuse the entire scanning,
refinement, confidence and reporting machinery.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config import ROI, CoarseScanConfig, RefinementConfig, apply_overrides
from ..logging_setup import get_logger
from ..video.reader import VideoReader
from .base_detector import (
    ActivityScorer,
    BaseDetector,
    DetectorResult,
    Event,
    EventStatus,
    ProgressReporter,
    null_progress,
)
from .candidate_refinement import RefinedEvent, refine_candidates
from .coarse_scan import plan_coarse_scan, run_coarse_scan
from .motion_detector import MotionActivityScorer, MotionScorerConfig

logger = get_logger(__name__)


class CoarseToFineDetector(BaseDetector):
    """Scan cheaply, refine locally, report events with evidence."""

    name: ClassVar[str] = "coarse_to_fine"
    display_name: ClassVar[str] = "Motion / change scan"
    description: ClassVar[str] = (
        "Scans the whole recording quickly for change, then re-analyses only "
        "the interesting parts in detail."
    )
    default_label: ClassVar[str] = "Activity"

    def configure(self) -> None:
        self.scan_config = apply_overrides(CoarseScanConfig(), self.params, ignore={"roi"})
        self.refine_config = apply_overrides(RefinementConfig(), self.params, ignore={"roi"})
        self.scorer_config = apply_overrides(MotionScorerConfig(), self.params, ignore={"roi"})
        self.scan_config.validate()
        self.refine_config.validate()
        self.scorer_config.validate()

        roi_param = self.params.get("roi")
        self.roi: ROI | None = None
        if roi_param:
            roi = ROI.from_dict(roi_param)
            roi.validate(self.video.width, self.video.height)
            self.roi = roi

        if self.video.duration_s is None:
            logger.warning(
                "%s has no known duration; progress reporting will be approximate",
                self.video.path.name,
            )

    # -- hooks for subclasses ---------------------------------------------- #

    def make_scorer(self) -> ActivityScorer:
        """Create a fresh scorer (each pass starts with a clean background)."""
        return MotionActivityScorer(self.scorer_config)

    def label_for(self, event: RefinedEvent) -> str:
        """Name the event. Subclasses give plant-specific meaning here."""
        return self.default_label

    def derive_additional_events(self, events: list[Event]) -> list[Event]:
        """Optionally add events inferred from the detected ones (e.g. pauses)."""
        return []

    # -- description -------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        plan = plan_coarse_scan(self.video, self.scan_config, roi=self.roi)
        return self._describe_common(
            roi=self.roi.to_dict() if self.roi else None,
            shortest_event_s=self.scan_config.shortest_event_s,
            sampling_interval_s=round(plan.effective_interval_s, 4),
            sampling_safety_factor=self.scan_config.safety_factor,
            scan_scale=round(plan.scale, 4),
            estimated_samples=plan.estimated_samples,
            pre_roll_s=self.refine_config.pre_roll_s,
            post_roll_s=self.refine_config.post_roll_s,
            min_event_duration_s=self.refine_config.min_event_duration_s,
            sensitivity=self.scan_config.sensitivity,
        )

    # -- execution ---------------------------------------------------------- #

    def run(
        self, reader: VideoReader, progress: ProgressReporter = null_progress
    ) -> DetectorResult:
        progress(stage="scanning", fraction=0.0, message="Scanning video for activity")
        scan = run_coarse_scan(
            reader,
            self.make_scorer(),
            self.scan_config,
            roi=self.roi,
            progress=progress,
        )

        warnings: list[str] = []
        if scan.truncated:
            warnings.append(
                f"More than {self.scan_config.max_candidates} candidate periods were found; "
                "only the strongest were analysed in detail. Consider lowering the "
                "sensitivity or selecting a region of interest."
            )
        if not scan.candidates:
            logger.info("No candidate windows in %s", self.video.path.name)
            return DetectorResult(
                events=[],
                diagnostics={
                    "coarse_scan": scan.to_dict(),
                    "roi": self.roi.to_dict() if self.roi else None,
                },
                trace=scan.trace,
                summary=self._summary(scan, refined_count=0, event_count=0, refine_elapsed=0.0),
                warnings=warnings
                + [
                    "No activity above the detection threshold was found. If activity "
                    "was expected, raise the sensitivity or narrow the region of interest."
                ],
            )

        progress(
            stage="refining",
            fraction=0.0,
            message=f"{len(scan.candidates)} candidate window(s) found; refining",
        )
        outcome = refine_candidates(
            reader,
            self.make_scorer,
            scan.candidates,
            self.refine_config,
            coarse_interval_s=scan.plan.effective_interval_s,
            roi=self.roi,
            progress=progress,
        )

        events = [self._to_event(refined) for refined in outcome.events]
        events.extend(self.derive_additional_events(events))
        events.sort(key=lambda event: event.start_s)

        progress(stage="complete", fraction=1.0, message="Analysis complete")
        return DetectorResult(
            events=events,
            diagnostics={
                "coarse_scan": scan.to_dict(),
                "refinement": {
                    "windows": outcome.windows_analysed,
                    "elapsed_s": round(outcome.elapsed_s, 2),
                    "evidence": [refined.evidence() for refined in outcome.events],
                },
                "roi": self.roi.to_dict() if self.roi else None,
            },
            trace=scan.trace,
            summary=self._summary(
                scan,
                refined_count=len(outcome.events),
                event_count=len(events),
                refine_elapsed=outcome.elapsed_s,
            ),
            warnings=warnings,
        )

    def _to_event(self, refined: RefinedEvent) -> Event:
        status = (
            EventStatus.CONFIRMED
            if refined.confidence >= self.refine_config.review_confidence
            else EventStatus.REVIEW
        )
        notes: list[str] = []
        if refined.clipped_start or refined.clipped_end:
            notes.append(
                "The event reaches the edge of the analysed window, so its true "
                "boundary may lie slightly outside the reported times."
            )
        if refined.disturbed_ratio > 0.05:
            notes.append(
                f"{refined.disturbed_ratio:.0%} of the frames in this window were "
                "disturbed (camera movement or a scene-wide light change)."
            )
        return Event(
            label=self.label_for(refined),
            start_s=refined.interval.start_s,
            end_s=refined.interval.end_s,
            confidence=refined.confidence,
            detector=self.name,
            status=status,
            notes=tuple(notes),
            details=refined.evidence(),
        )

    def _summary(
        self,
        scan: Any,
        *,
        refined_count: int,
        event_count: int,
        refine_elapsed: float,
    ) -> dict[str, Any]:
        duration = self.video.duration_s or 0.0
        scan_elapsed = scan.elapsed_s or 1e-9
        return {
            "mode": self.name,
            "fps": round(self.video.fps, 4),
            "duration_s": duration,
            "samples_analysed": len(scan.trace),
            "sample_interval_s": round(scan.plan.effective_interval_s, 4),
            "scan_scale": round(scan.plan.scale, 4),
            "candidate_windows": len(scan.candidates),
            "refined_events": refined_count,
            "event_count": event_count,
            "scan_seconds": round(scan.elapsed_s, 2),
            "refine_seconds": round(refine_elapsed, 2),
            "processing_seconds": round(scan.elapsed_s + refine_elapsed, 2),
            # How many times faster than real time the coarse pass ran; the
            # headline number for "can this handle a 12 hour recording?".
            "scan_speed_x_realtime": round(duration / scan_elapsed, 1) if duration else None,
        }


class GenericMotionDetector(CoarseToFineDetector):
    """Mode 3 in the UI: find anything that changes, without interpreting it."""

    name: ClassVar[str] = "motion_scan"
    display_name: ClassVar[str] = "Generic motion / change scan"
    description: ClassVar[str] = (
        "Reports every period in which the picture changed meaningfully. Use "
        "this when you do not yet know what you are looking for."
    )
    default_label: ClassVar[str] = "Motion"
