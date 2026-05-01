"""
model/threshold.py — ensemble threat scorer

Combines three signals into a single threat score:
  - Audio classifier confidence       (weight 0.60)
  - Vibration / impact anomaly score  (weight 0.25)
  - Time-of-day context score         (weight 0.15)

# weights tuned after testing in 3 different home environments.
# night weighting was suggested by a friend who works in crisis counseling —
# she said incidents are significantly more likely after 10pm and the
# system should reflect that without being the only factor.

All inputs and the output are floats in [0.0, 1.0].
"""

import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Ensemble weights — must sum to 1.0 ───────────────────────────────────────
_W_AUDIO = 0.60
_W_VIBRATION = 0.25
_W_TIME = 0.15

assert abs(_W_AUDIO + _W_VIBRATION + _W_TIME - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Night hours — local time ──────────────────────────────────────────────────
# "Night" is 10pm–6am. Score ramps between boundary hours to avoid a cliff.
_NIGHT_START_HOUR = 22   # 10 PM
_NIGHT_END_HOUR = 6      # 6 AM
_NIGHT_SCORE = 0.75      # time score during full night window
_DAY_SCORE = 0.25        # time score during the day
_TRANSITION_HOURS = 1    # ramp length on each boundary


@dataclass
class ThreatComponents:
    """
    All inputs and intermediate values that produced the final score.
    Logged on every alert for post-hoc review.
    """
    audio_score: float
    vibration_score: float
    time_score: float
    weighted_score: float
    hour: int


def _time_of_day_score(now: datetime.datetime | None = None) -> tuple[float, int]:
    """
    Returns (score, hour). Score is higher at night.

    Uses a linear ramp in the transition windows so the score doesn't
    jump discontinuously at exactly 22:00 or 06:00.
    """
    if now is None:
        now = datetime.datetime.now()
    hour = now.hour + now.minute / 60.0  # fractional hour

    # Check if we're in full night
    in_full_night = hour >= _NIGHT_START_HOUR or hour < _NIGHT_END_HOUR

    if in_full_night:
        # Check transition zones
        if _NIGHT_START_HOUR <= hour < _NIGHT_START_HOUR + _TRANSITION_HOURS:
            # Ramping into night
            t = (hour - _NIGHT_START_HOUR) / _TRANSITION_HOURS
            score = _DAY_SCORE + t * (_NIGHT_SCORE - _DAY_SCORE)
        elif _NIGHT_END_HOUR - _TRANSITION_HOURS <= hour < _NIGHT_END_HOUR:
            # Ramping out of night
            t = (hour - (_NIGHT_END_HOUR - _TRANSITION_HOURS)) / _TRANSITION_HOURS
            score = _NIGHT_SCORE - t * (_NIGHT_SCORE - _DAY_SCORE)
        else:
            score = _NIGHT_SCORE
    else:
        # Check transition zones from day side
        if _NIGHT_END_HOUR <= hour < _NIGHT_END_HOUR + _TRANSITION_HOURS:
            # Just left night — ramping down
            t = (hour - _NIGHT_END_HOUR) / _TRANSITION_HOURS
            score = _NIGHT_SCORE - t * (_NIGHT_SCORE - _DAY_SCORE)
        elif _NIGHT_START_HOUR - _TRANSITION_HOURS <= hour < _NIGHT_START_HOUR:
            # Approaching night — ramping up
            t = (hour - (_NIGHT_START_HOUR - _TRANSITION_HOURS)) / _TRANSITION_HOURS
            score = _DAY_SCORE + t * (_NIGHT_SCORE - _DAY_SCORE)
        else:
            score = _DAY_SCORE

    return float(max(0.0, min(1.0, score))), int(now.hour)


def compute_threat_score(
    audio_score: float,
    vibration_score: float,
    now: datetime.datetime | None = None,
) -> ThreatComponents:
    """
    Compute the ensemble threat score from the three input signals.

    Args:
        audio_score:     Output from classifier.predict() — [0.0, 1.0]
        vibration_score: Output from sensors/vibration.py — [0.0, 1.0]
        now:             Datetime to use for time scoring. None = current time.
                         Exposed for testing with fixed times.

    Returns:
        ThreatComponents with the final weighted_score and all intermediates.
    """
    # Clamp inputs — don't trust callers to stay in range
    audio_score = float(max(0.0, min(1.0, audio_score)))
    vibration_score = float(max(0.0, min(1.0, vibration_score)))

    time_score, hour = _time_of_day_score(now)

    weighted = (
        _W_AUDIO * audio_score
        + _W_VIBRATION * vibration_score
        + _W_TIME * time_score
    )
    weighted = float(max(0.0, min(1.0, weighted)))

    components = ThreatComponents(
        audio_score=audio_score,
        vibration_score=vibration_score,
        time_score=time_score,
        weighted_score=weighted,
        hour=hour,
    )

    logger.debug(
        f"Threat score: audio={audio_score:.3f} vib={vibration_score:.3f} "
        f"time={time_score:.3f} → {weighted:.3f} (hour={hour})"
    )

    return components
