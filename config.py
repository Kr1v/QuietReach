"""
config.py — QuietReach configuration loader

Load all env vars at startup and validate them hard.
Learned the hard way that silent missing keys cause really bad bugs
at 2am. Now everything fails loudly on startup instead.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class QuietReachConfig:
    # Twilio
    twilio_sid: str = ""
    twilio_token: str = ""
    twilio_from: str = ""
    trusted_number: str = ""

    # Firebase (optional fallback)
    firebase_key: Optional[str] = None

    # Encryption
    encryption_key: str = ""

    # Detection thresholds
    threat_threshold: float = 0.72
    window_size_seconds: int = 3
    calibration_duration: int = 10
    alert_cooldown_minutes: int = 10
    consecutive_seconds_required: int = 8

    # Audio
    sample_rate: int = 16000
    chunk_size: int = 1024
    mfcc_coefficients: int = 40

    # Sensor mode — "demo" uses mic RMS proxy, "phone" expects flask endpoint
    sensor_mode: str = "demo"
    phone_sensor_port: int = 5050

    # Model
    model_path: str = "model/saved/quietreach_v1.tflite"
    sklearn_fallback_path: str = "model/saved/quietreach_v1.pkl"

    # Misc
    debug: bool = False
    log_level: str = "INFO"


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning(f"Config: {key} is not a valid float, using default {default}")
        return default


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning(f"Config: {key} is not a valid int, using default {default}")
        return default


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def load_config() -> QuietReachConfig:
    """
    Build config from environment. Call this once at startup.

    Returns a frozen-ish config object. Don't mutate it after loading —
    there's no protection against that but please don't.
    """
    cfg = QuietReachConfig(
        twilio_sid=os.getenv("TWILIO_SID", ""),
        twilio_token=os.getenv("TWILIO_TOKEN", ""),
        twilio_from=os.getenv("TWILIO_FROM", ""),
        trusted_number=os.getenv("TRUSTED_NUMBER", ""),
        firebase_key=os.getenv("FIREBASE_KEY"),  # optional
        encryption_key=os.getenv("ENCRYPTION_KEY", ""),
        threat_threshold=_get_float("THREAT_THRESHOLD", 0.72),
        window_size_seconds=_get_int("WINDOW_SIZE_SECONDS", 3),
        calibration_duration=_get_int("CALIBRATION_DURATION", 10),
        alert_cooldown_minutes=_get_int("ALERT_COOLDOWN_MINUTES", 10),
        consecutive_seconds_required=_get_int("CONSECUTIVE_SECONDS_REQUIRED", 8),
        sample_rate=_get_int("SAMPLE_RATE", 16000),
        chunk_size=_get_int("CHUNK_SIZE", 1024),
        mfcc_coefficients=_get_int("MFCC_COEFFICIENTS", 40),
        sensor_mode=os.getenv("SENSOR_MODE", "demo").lower(),
        phone_sensor_port=_get_int("PHONE_SENSOR_PORT", 5050),
        model_path=os.getenv("MODEL_PATH", "model/saved/quietreach_v1.tflite"),
        sklearn_fallback_path=os.getenv("SKLEARN_FALLBACK_PATH", "model/saved/quietreach_v1.pkl"),
        debug=_get_bool("DEBUG", False),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
    return cfg


# Keys that must exist for the app to actually function.
# Firebase is omitted — it's a fallback and not strictly required.
_REQUIRED_KEYS = [
    ("TWILIO_SID", "twilio_sid"),
    ("TWILIO_TOKEN", "twilio_token"),
    ("TWILIO_FROM", "twilio_from"),
    ("TRUSTED_NUMBER", "trusted_number"),
    ("ENCRYPTION_KEY", "encryption_key"),
]


def validate_config(cfg: QuietReachConfig) -> bool:
    """
    Check that all required config values are present.

    Exits hard if anything is missing. Better to crash on startup
    than to silently fail when someone actually needs the alert to go out.

    Returns True if everything is valid (mostly useful in tests).
    """
    errors: list[str] = []

    for env_key, attr in _REQUIRED_KEYS:
        val = getattr(cfg, attr, None)
        if not val:
            errors.append(f"  Missing required env var: {env_key}")

    if cfg.sensor_mode not in ("demo", "phone"):
        errors.append(f"  SENSOR_MODE must be 'demo' or 'phone', got: '{cfg.sensor_mode}'")

    if not (0.0 < cfg.threat_threshold <= 1.0):
        errors.append(f"  THREAT_THRESHOLD must be between 0 and 1, got: {cfg.threat_threshold}")

    if cfg.consecutive_seconds_required < 3:
        # Below 3 seconds you get way too many false positives
        errors.append("  CONSECUTIVE_SECONDS_REQUIRED should be at least 3 seconds")

    if errors:
        logger.critical("QuietReach config validation failed:")
        for err in errors:
            logger.critical(err)
        logger.critical("Copy .env.example to .env and fill in your values.")
        sys.exit(1)

    logger.info("Config validated OK.")
    return True


def setup_logging(cfg: QuietReachConfig) -> None:
    """Configure root logger. Call right after load_config()."""
    level = getattr(logging, cfg.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if cfg.debug:
        logging.getLogger("quietreach").setLevel(logging.DEBUG)
