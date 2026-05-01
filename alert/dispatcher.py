"""
alert/dispatcher.py — alert trigger logic

This is the most important module in the codebase. Get this wrong
and either:
  a) real threats are missed (false negatives — dangerous)
  b) the trusted contact gets spammed until they ignore alerts (also dangerous)

Rules before an alert fires:
  1. Threat score must exceed config.threat_threshold
  2. Score must stay above threshold for 8 consecutive seconds  → counts as 1 occurrence
  3. Another occurrence must happen within 15 seconds of the last one
  4. After 3 occurrences within the window → alert fires
  5. A 3-second secondary confirmation window runs before actual send
  6. Cooldown of config.alert_cooldown_minutes between alerts

Rationale: a single 8-second burst could be a TV show or argument that
de-escalates. Three bursts within 15 seconds of each other means the
situation is ongoing — that's when a real incident is likely.

All decisions are logged to an in-memory audit log. Nothing hits disk.
"""

import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import QuietReachConfig
from model.threshold import ThreatComponents

logger = logging.getLogger(__name__)

# ── Occurrence-based detection constants ─────────────────────────────────────
OCCURRENCES_REQUIRED   = 3     # how many sustained bursts needed before alert
OCCURRENCE_WINDOW_SECS = 15.0  # seconds within which occurrences must happen
SUSTAINED_SECS         = 8.0   # seconds of sustained threat = 1 occurrence


@dataclass
class AlertEvent:
    """Record of a triggered alert — stored in memory for UI and audit."""
    triggered_at: datetime.datetime
    threat_score: float
    audio_score: float
    vibration_score: float
    time_score: float
    location_str: str
    sms_success: bool
    push_success: bool
    confirmation_score: float
    occurrences: int               # how many bursts were detected before firing


@dataclass
class DispatcherState:
    """Mutable runtime state. Separated from Dispatcher to make testing clean."""
    # Current burst accumulation
    consecutive_high_seconds: float = 0.0

    # Occurrence tracking
    occurrence_times: list[float] = field(default_factory=list)  # monotonic timestamps
    occurrence_count: int = 0

    # Alert state
    last_alert_time: Optional[float] = None   # monotonic
    in_cooldown: bool = False
    alert_log: list[AlertEvent] = field(default_factory=list)
    current_score: float = 0.0


AlertCallback = Callable[[AlertEvent], None]


class AlertDispatcher:
    """
    Receives a threat score on each window and decides whether to fire.

    Detection logic:
      - Score above threshold for 8 continuous seconds = 1 occurrence
      - 3 occurrences within a 15-second rolling window = alert fires
      - Occurrences older than 15 seconds are discarded automatically
      - 3-second confirmation check before sending
      - Cooldown between alerts prevents repeat sends

    Usage in main loop:
        dispatcher.on_score(components, window_duration_seconds)
    """

    def __init__(
        self,
        cfg: QuietReachConfig,
        alert_callback: AlertCallback,
        confirmation_source: Optional[Callable[[], float]] = None,
    ) -> None:
        self._cfg = cfg
        self._alert_callback = alert_callback
        self._confirmation_source = confirmation_source
        self._state = DispatcherState()

    def on_score(
        self,
        components: ThreatComponents,
        window_duration_seconds: float,
        location_str: str = "unknown",
    ) -> bool:
        """
        Process one threat score. Returns True if an alert was triggered.

        window_duration_seconds: unique audio contribution of this window.
            For 3s window with 50% overlap, pass 1.5.
        """
        score     = components.weighted_score
        threshold = self._cfg.threat_threshold
        self._state.current_score = score
        now = time.monotonic()

        # ── Cooldown check ────────────────────────────────────────────────
        if self._state.last_alert_time is not None:
            elapsed = now - self._state.last_alert_time
            cooldown = self._cfg.alert_cooldown_minutes * 60
            if elapsed < cooldown:
                logger.debug(f"Cooldown: {cooldown - elapsed:.0f}s remaining")
                self._state.in_cooldown = True
                self._state.consecutive_high_seconds = 0.0
                return False
            else:
                self._state.in_cooldown = False

        # ── Expire old occurrences outside the 15-second window ───────────
        self._state.occurrence_times = [
            t for t in self._state.occurrence_times
            if now - t <= OCCURRENCE_WINDOW_SECS
        ]
        self._state.occurrence_count = len(self._state.occurrence_times)

        # ── Accumulate consecutive high-score time ────────────────────────
        if score >= threshold:
            self._state.consecutive_high_seconds += window_duration_seconds
            logger.debug(
                f"Score {score:.3f} >= {threshold} | "
                f"burst: {self._state.consecutive_high_seconds:.1f}s / {SUSTAINED_SECS}s | "
                f"occurrences: {self._state.occurrence_count} / {OCCURRENCES_REQUIRED}"
            )
        else:
            if self._state.consecutive_high_seconds > 0:
                logger.debug(
                    f"Score dropped to {score:.3f} — "
                    f"burst reset at {self._state.consecutive_high_seconds:.1f}s"
                )
            self._state.consecutive_high_seconds = 0.0
            return False

        # ── Check if current burst qualifies as an occurrence ─────────────
        if self._state.consecutive_high_seconds >= SUSTAINED_SECS:
            self._state.occurrence_times.append(now)
            self._state.occurrence_count = len(self._state.occurrence_times)
            self._state.consecutive_high_seconds = 0.0  # reset for next burst

            logger.warning(
                f"Occurrence {self._state.occurrence_count}/{OCCURRENCES_REQUIRED} detected "
                f"(score={score:.3f}, sustained={SUSTAINED_SECS}s)"
            )

            # ── Check if we have enough occurrences ───────────────────────
            if self._state.occurrence_count < OCCURRENCES_REQUIRED:
                logger.info(
                    f"Waiting for more occurrences — "
                    f"{OCCURRENCES_REQUIRED - self._state.occurrence_count} more needed "
                    f"within {OCCURRENCE_WINDOW_SECS}s"
                )
                return False

            # ── 3 occurrences reached — run confirmation ──────────────────
            confirmation_score = self._run_confirmation_check()
            if confirmation_score < threshold:
                logger.info(
                    f"Alert suppressed — confirmation score {confirmation_score:.3f} "
                    f"dropped below threshold after occurrences"
                )
                self._state.occurrence_times.clear()
                self._state.occurrence_count = 0
                return False

            # ── Fire alert ────────────────────────────────────────────────
            logger.warning(
                f"ALERT TRIGGERED — "
                f"{self._state.occurrence_count} occurrences in {OCCURRENCE_WINDOW_SECS}s, "
                f"score={score:.3f}, confirmation={confirmation_score:.3f}"
            )

            event = AlertEvent(
                triggered_at=datetime.datetime.now(),
                threat_score=score,
                audio_score=components.audio_score,
                vibration_score=components.vibration_score,
                time_score=components.time_score,
                location_str=location_str,
                sms_success=False,
                push_success=False,
                confirmation_score=confirmation_score,
                occurrences=self._state.occurrence_count,
            )

            self._state.last_alert_time = now
            self._state.occurrence_times.clear()
            self._state.occurrence_count = 0

            try:
                self._alert_callback(event)
            except Exception as e:
                logger.error(f"Alert callback raised an exception: {e}")

            self._state.alert_log.append(event)
            return True

        return False

    def _run_confirmation_check(self) -> float:
        """Wait 3 seconds and return a confirmation score."""
        if self._confirmation_source is None:
            return self._state.current_score

        logger.info("Running 3-second confirmation window...")
        time.sleep(3.0)
        score = self._confirmation_source()
        logger.info(f"Confirmation score: {score:.3f}")
        return score

    # ── Read-only accessors for UI ────────────────────────────────────────

    @property
    def consecutive_seconds(self) -> float:
        return self._state.consecutive_high_seconds

    @property
    def occurrence_count(self) -> int:
        return self._state.occurrence_count

    @property
    def in_cooldown(self) -> bool:
        return self._state.in_cooldown

    @property
    def last_alert_time(self) -> Optional[float]:
        return self._state.last_alert_time

    @property
    def alert_log(self) -> list[AlertEvent]:
        return self._state.alert_log

    @property
    def current_score(self) -> float:
        return self._state.current_score

    def time_since_last_alert(self) -> Optional[float]:
        if self._state.last_alert_time is None:
            return None
        return time.monotonic() - self._state.last_alert_time

    def cooldown_remaining(self) -> float:
        if self._state.last_alert_time is None:
            return 0.0
        elapsed = time.monotonic() - self._state.last_alert_time
        cooldown = self._cfg.alert_cooldown_minutes * 60
        return max(0.0, cooldown - elapsed)