"""
main.py — QuietReach entry point

Thin orchestration layer. All logic lives in the submodules.
This file wires them together and runs the main detection loop.

Run with:
    python main.py
    python main.py --no-ui       # headless / server mode
    python main.py --calibrate   # re-run calibration, then start normally
"""

import argparse
import datetime
import logging
import signal
import sys
import threading
import time
from typing import Optional

# ── Config must load before anything else ─────────────────────────────────────
from config import load_config, setup_logging, validate_config

cfg = load_config()
setup_logging(cfg)
# validate_config is called inside main() after --demo flag is checked

logger = logging.getLogger("quietreach.main")

# ── Module imports (after config) ─────────────────────────────────────────────
from alert.dispatcher import AlertDispatcher, AlertEvent
from alert.location import get_current_location
from alert.notification import send_push_notification
from alert.sms import send_alert_sms
from audio.calibrator import Calibrator
from audio.capture import MicCapture, make_capture_queue
from audio.processor import AudioProcessor
from model.classifier import load_classifier
from model.threshold import compute_threat_score
from privacy.encryptor import PayloadEncryptor
from privacy.memory_cleaner import wipe_and_release
from sensors.vibration import make_vibration_sensor


# ── UI data source adapter ────────────────────────────────────────────────────

class AppState:
    """
    Central read/write state container. Passed to UI as the UIDataSource.
    Written by the detection loop, read by the UI thread.

    All fields are safe to read without locks — they're scalars
    assigned atomically in CPython. If we move to multiprocessing
    this needs proper synchronization.
    """
    def __init__(self) -> None:
        self.current_threat_score: float = 0.0
        self.current_raw_rms: float = 0.0
        self.consecutive_seconds: float = 0.0
        self.consecutive_seconds_required: float = float(cfg.consecutive_seconds_required)
        self.calibrated: bool = False
        self.classifier_backend: str = "none"
        self.in_cooldown: bool = False
        self.cooldown_remaining: float = 0.0
        self.occurrence_count: int = 0
        self.last_alert_time: Optional[datetime.datetime] = None
        self.is_running: bool = False
        self._stop: bool = False


# ── Alert callback ────────────────────────────────────────────────────────────

def make_alert_callback(state: AppState, encryptor: PayloadEncryptor):
    """
    Returns the function that fires when the dispatcher decides to alert.
    Closure over state and encryptor so the dispatcher stays stateless.
    """
    def on_alert(event: AlertEvent) -> None:
        logger.warning(f"ALERT: score={event.threat_score:.3f} at {event.triggered_at}")

        # Resolve location
        location = get_current_location()
        alert_time_str = event.triggered_at.strftime("%Y-%m-%d %H:%M:%S")

        # Encrypt payload (location + time only — never audio)
        _encrypted = encryptor.build_alert_payload(
            lat=location.lat,
            lng=location.lng,
            address=location.address,
            alert_time=alert_time_str,
            location_source=location.source,
        )
        # _encrypted is available for logging/audit; not sent in SMS body directly
        # (SMS goes plaintext to trusted number — encryption is for any future
        # server storage path in v2)

        # Send SMS
        sms_result = send_alert_sms(
            twilio_sid=cfg.twilio_sid,
            twilio_token=cfg.twilio_token,
            from_number=cfg.twilio_from,
            to_number=cfg.trusted_number,
            location=location,
            alert_time=alert_time_str,
        )
        event.sms_success = sms_result.success

        # Push notification fallback
        push_result = send_push_notification(
            firebase_key=cfg.firebase_key,
            location=location,
            alert_time=alert_time_str,
        )
        event.push_success = push_result.success

        # Update UI state
        state.last_alert_time = event.triggered_at

        if sms_result.success:
            logger.info(f"Alert delivered via SMS (SID={sms_result.sid})")
        else:
            logger.error(f"SMS failed: {sms_result.error}")
            if not push_result.skipped and push_result.success:
                logger.info("Push notification sent as fallback")

    return on_alert


# ── Detection loop ────────────────────────────────────────────────────────────

def run_detection_loop(
    state: AppState,
    processor: AudioProcessor,
    classifier,
    vibration_sensor,
    dispatcher: AlertDispatcher,
) -> None:
    """
    Main detection loop. Runs on the calling thread.
    Exits when state._stop is set.

    Window step for consecutive-time accounting: the processor uses a
    50% sliding window, so each window contributes 1.5 seconds of unique
    audio to the consecutive counter (half the 3-second window size).
    """
    window_contribution_seconds = cfg.window_size_seconds / 2.0

    logger.info("Detection loop started")
    state.is_running = True

    while not state._stop:
        # Pull next feature vector (blocks up to 0.5s waiting for audio)
        fv = processor.next_feature_vector(timeout=0.5)
        if fv is None:
            continue

        # Run classifier
        audio_score = classifier.predict(fv.vec)

        # Update vibration sensor with current RMS
        vibration_sensor.update(fv.raw_rms)
        vib_score = vibration_sensor.score()

        # Ensemble scoring
        components = compute_threat_score(
            audio_score=audio_score,
            vibration_score=vib_score,
        )

        # Wipe audio buffer immediately after inference
        # Audio feature extraction is done — zero the underlying array
        wipe_and_release(fv.vec)

        # Update UI state
        state.current_threat_score = components.weighted_score
        state.current_raw_rms = fv.raw_rms
        state.consecutive_seconds = dispatcher.consecutive_seconds
        state.occurrence_count = dispatcher.occurrence_count
        state.in_cooldown = dispatcher.in_cooldown
        state.cooldown_remaining = dispatcher.cooldown_remaining()

        # Dispatcher decides whether to fire
        # Pass a lambda as the confirmation source so it reads live score
        dispatcher.on_score(
            components=components,
            window_duration_seconds=window_contribution_seconds,
        )

    state.is_running = False
    logger.info("Detection loop stopped")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuietReach — passive threat detection")
    p.add_argument("--no-ui",       action="store_true", help="Run headless (no terminal dashboard)")
    p.add_argument("--calibrate",   action="store_true", help="Force re-calibration on startup")
    p.add_argument("--list-devices",action="store_true", help="List audio input devices and exit")
    p.add_argument("--device",      type=int, default=None, help="Mic device index (default: system default)")
    p.add_argument("--demo",        action="store_true",
                   help="Demo mode — detection runs but no SMS/alerts sent (no Twilio needed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_devices:
        devices = MicCapture.list_devices()
        print("Available audio input devices:")
        for d in devices:
            print(f"  [{d['index']}] {d['name']} ({d['sample_rate']}Hz)")
        return

    state = AppState()

    # ── Validate config — skip Twilio checks in demo mode ─────────────────
    if args.demo:
        logger.info("Demo mode — skipping Twilio/encryption config validation")
    else:
        validate_config(cfg)

    # ── Graceful shutdown ──────────────────────────────────────────────────
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received")
        state._stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Load classifier ────────────────────────────────────────────────────
    logger.info("Loading classifier...")
    try:
        classifier = load_classifier(
            tflite_path=cfg.model_path,
            sklearn_path=cfg.sklearn_fallback_path,
        )
        state.classifier_backend = classifier.backend
    except FileNotFoundError as e:
        logger.warning(f"{e}")
        logger.warning("Using DummyClassifier — run model/trainer.py to train a real model")
        from model.classifier import DummyClassifier
        classifier = DummyClassifier(fixed_score=0.05)
        state.classifier_backend = "dummy"

    # ── Vibration sensor ───────────────────────────────────────────────────
    vibration_sensor = make_vibration_sensor(
        sensor_mode=cfg.sensor_mode,
        phone_port=cfg.phone_sensor_port,
    )

    # ── Encryption + alert callback ────────────────────────────────────────
    if args.demo:
        logger.info("DEMO MODE — detection active, alerts suppressed (no SMS will be sent)")
        def alert_callback(event: AlertEvent) -> None:
            logger.warning(
                f"[DEMO] Alert would fire: score={event.threat_score:.3f} "
                f"at {event.triggered_at.strftime('%H:%M:%S')} — SMS suppressed in demo mode"
            )
            state.last_alert_time = event.triggered_at
    else:
        encryptor      = PayloadEncryptor(cfg.encryption_key)
        alert_callback = make_alert_callback(state, encryptor)

    # ── Audio pipeline ─────────────────────────────────────────────────────
    audio_queue = make_capture_queue()
    capture = MicCapture(
        out_queue=audio_queue,
        sample_rate=cfg.sample_rate,
        chunk_size=cfg.chunk_size,
        device_index=args.device,
    )

    logger.info("Starting microphone capture...")
    capture.start()

    # ── Calibration ────────────────────────────────────────────────────────
    cal_duration = cfg.calibration_duration
    if args.calibrate:
        # Force a longer calibration pass — useful if the environment changed
        cal_duration = max(cfg.calibration_duration, 20)
        logger.info(f"--calibrate flag set: extended calibration ({cal_duration}s)")
    else:
        logger.info(f"Calibrating ({cal_duration}s — stay quiet)...")

    calibrator = Calibrator(
        audio_queue=audio_queue,
        sample_rate=cfg.sample_rate,
        chunk_size=cfg.chunk_size,
        mfcc_n=cfg.mfcc_coefficients,
        calibration_duration=cal_duration,
        window_size_seconds=cfg.window_size_seconds,
    )
    baseline = calibrator.calibrate()
    state.calibrated = baseline.calibrated

    # ── Processor ──────────────────────────────────────────────────────────
    processor = AudioProcessor(
        audio_queue=audio_queue,
        baseline=baseline,
        sample_rate=cfg.sample_rate,
        chunk_size=cfg.chunk_size,
        window_seconds=cfg.window_size_seconds,
        mfcc_n=cfg.mfcc_coefficients,
    )

    # ── Dispatcher ─────────────────────────────────────────────────────────
    dispatcher = AlertDispatcher(
        cfg=cfg,
        alert_callback=alert_callback,
        confirmation_source=lambda: state.current_threat_score,
    )

    # ── UI ─────────────────────────────────────────────────────────────────
    if not args.no_ui:
        from ui.terminal_ui import TerminalDashboard
        dashboard = TerminalDashboard(source=state, refresh_interval=0.5)
        ui_thread = threading.Thread(target=dashboard.run, name="ui", daemon=True)
        ui_thread.start()
    else:
        logger.info("Running headless (--no-ui)")

    # ── Detection loop (blocks until shutdown) ────────────────────────────
    try:
        run_detection_loop(
            state=state,
            processor=processor,
            classifier=classifier,
            vibration_sensor=vibration_sensor,
            dispatcher=dispatcher,
        )
    finally:
        logger.info("Cleaning up...")
        capture.stop()
        logger.info("QuietReach stopped.")


if __name__ == "__main__":
    main()