"""Multi-engine configuration contract.

Supervisor-approved product requirement (PR #4 comment thread, second
authorization): the eventual workflow submits one uploaded video to
multiple independently-configured engines, shows every engine's result
side by side, and never averages or picks a winner. This module is the
backend data model and orchestration for that - **not** the Settings UI or
any web-layer wiring, which stays deferred until reviewed (see
``diagnostics/llm_spike/DESIGN.md`` §10).

Nothing here ever holds a secret's actual value. ``EngineConfig.credential_ref``
is a *reference* (an environment variable name, or a secret-store key) that a
real ``TimingProvider`` implementation resolves for itself, server-side, at
call time. See :func:`_looks_like_a_raw_secret` for the one guard this module
enforces in code, per the requirement that secrets must never be committed,
logged, or otherwise handled as plain config values.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ...errors import ConfigurationError
from .pipeline import PipelineConfig, PipelineOutcome, run_llm_timing
from .provider import TimingProvider
from .redaction import sanitize_untrusted_text

# Heuristic patterns for common vendor API key shapes (OpenAI "sk-...",
# Anthropic "sk-ant-...", generic long opaque tokens). This is a safety net,
# not a guarantee - it exists to catch the easy mistake of pasting an actual
# key into what should be a *reference* to one, not to replace secret
# scanning elsewhere in the toolchain. Anchored, for validating a value that
# should be nothing *but* a reference (see ``redaction.py`` for the
# unanchored versions used to scrub a secret out of free text).
_RAW_SECRET_PATTERNS = (
    re.compile(r"^sk-[A-Za-z0-9_-]{10,}$"),
    re.compile(r"^[A-Za-z0-9_-]{32,}$"),
)


def _looks_like_a_raw_secret(value: str) -> bool:
    return any(pattern.match(value) for pattern in _RAW_SECRET_PATTERNS)


@dataclass(frozen=True)
class EngineConfig:
    """One configured engine entry - the backend shape behind a future
    Settings UI row.

    Attributes:
        engine_id: stable identifier for this entry (UI list key, log
            correlation). Not a display name.
        provider_name: which ``TimingProvider`` implementation to use (e.g.
            "openai", "anthropic", "google"). Resolved by a caller-supplied
            factory, never imported here - keeps this module vendor-free.
        model_id: the specific model to request from that provider.
        enabled: engines default to a single enabled entry in the UI; more
            can be added and any can be disabled without being removed.
        credential_ref: a *reference* to a credential (an environment
            variable name, or a secret-store key) - never the credential's
            value. Rejected outright if it looks like an actual key (see
            ``_looks_like_a_raw_secret``); a real value belongs in the
            environment or secret store, resolved server-side by the
            provider implementation, never in this dataclass, never logged.

            ``credential_ref`` itself is server-side configuration and is
            deliberately excluded from :meth:`to_dict` - see that method.
            A future UI must never read or set it directly: accepting an
            arbitrary reference from a browser would let a client pick
            which server secret gets resolved (a secret-selection/oracle
            surface). A future "add credential" UI action is write-only
            into a server-side secret store; the reference it produces is
            never echoed back to the client.
        display_name: optional label for the UI; purely cosmetic.
    """

    engine_id: str
    provider_name: str
    model_id: str
    credential_ref: str
    enabled: bool = True
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.engine_id.strip():
            raise ConfigurationError("An engine entry needs a non-empty engine_id.")
        if not self.provider_name.strip():
            raise ConfigurationError("An engine entry needs a non-empty provider_name.")
        if not self.model_id.strip():
            raise ConfigurationError("An engine entry needs a non-empty model_id.")
        if not self.credential_ref.strip():
            raise ConfigurationError(
                "An engine entry needs a credential_ref (an environment variable "
                "name or secret-store key) - even a disabled entry, so it's ready "
                "to enable without re-entering configuration."
            )
        if _looks_like_a_raw_secret(self.credential_ref):
            raise ConfigurationError(
                "credential_ref must be a reference to a credential (e.g. an "
                "environment variable name), not the credential's actual value. "
                "Store the real value in the environment or a secret store."
            )

    def to_dict(self) -> dict:
        """The client-safe DTO.

        Deliberately omits ``credential_ref`` - even though it is only a
        reference, not a secret value, it is still server-side
        configuration that a client has no legitimate need to see (which
        env var name, which secret-store key). Only whether a credential is
        configured is exposed. See the ``credential_ref`` attribute
        docstring for why a future UI must treat this as write-only.
        """
        return {
            "engine_id": self.engine_id,
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "credential_configured": bool(self.credential_ref),
            "enabled": self.enabled,
            "display_name": self.display_name,
        }


class EngineRunStatus(str, Enum):
    RAN = "ran"  # produced a PipelineOutcome (itself CONFIRMED or ABSTAIN)
    SKIPPED = "skipped"  # entry was disabled
    ENGINE_ERROR = "engine_error"  # crashed before producing any verdict


@dataclass
class EngineOutcome:
    """One engine's result, kept separate from every other engine's.

    Never merged, averaged, or ranked against sibling engines - that
    aggregation is explicitly out of scope per the authorization: the UI is
    meant to show disagreement, not hide it behind a single number.
    """

    engine_id: str
    provider_name: str
    model_id: str
    status: EngineRunStatus
    outcome: PipelineOutcome | None = None  # set only when status is RAN
    error_message: str | None = None  # set only when status is ENGINE_ERROR

    def to_dict(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "status": self.status.value,
            "outcome": self.outcome.to_dict() if self.outcome is not None else None,
            "error_message": self.error_message,
        }


def run_llm_timing_for_engines(
    video_path: Path,
    engines: list[EngineConfig],
    provider_factory: Callable[[EngineConfig], TimingProvider],
    *,
    prompt_version: str,
    prompt_text: str,
    config: PipelineConfig | None = None,
) -> list[EngineOutcome]:
    """Run every enabled engine independently and return every result.

    Isolation is the point: ``provider_factory`` (credential resolution,
    client construction) and ``run_llm_timing`` both run inside a per-engine
    try/except, so one engine being misconfigured, unreachable, or erroring
    never prevents the others from producing a result. Results are returned
    in input order, one ``EngineOutcome`` per entry - no winner is chosen and
    nothing is averaged; that judgement belongs to whoever reads the list.
    """
    results: list[EngineOutcome] = []
    for engine in engines:
        if not engine.enabled:
            results.append(
                EngineOutcome(
                    engine_id=engine.engine_id,
                    provider_name=engine.provider_name,
                    model_id=engine.model_id,
                    status=EngineRunStatus.SKIPPED,
                )
            )
            continue

        try:
            provider = provider_factory(engine)
            outcome = run_llm_timing(
                video_path,
                provider,
                prompt_version=prompt_version,
                prompt_text=prompt_text,
                config=config,
            )
            results.append(
                EngineOutcome(
                    engine_id=engine.engine_id,
                    provider_name=engine.provider_name,
                    model_id=engine.model_id,
                    status=EngineRunStatus.RAN,
                    outcome=outcome,
                )
            )
        except Exception as exc:  # deliberately broad: per-engine isolation boundary
            results.append(
                EngineOutcome(
                    engine_id=engine.engine_id,
                    provider_name=engine.provider_name,
                    model_id=engine.model_id,
                    status=EngineRunStatus.ENGINE_ERROR,
                    error_message=sanitize_untrusted_text(str(exc)),
                )
            )
    return results
