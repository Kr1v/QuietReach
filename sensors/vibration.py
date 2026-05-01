"""
sensors/vibration.py — vibration / impact anomaly sensor

Two modes depending on SENSOR_MODE in config:

  DEMO MODE  ("demo")
    Uses mic RMS energy spikes as a vibration proxy.
    Works on any laptop — no extra hardware. Not ideal but good enough
    for demos and development. A sudden loud impact registers clearly.

  PHONE MODE ("phone")
    Receives accelerometer data via POST from a companion Flask endpoint
    running on a phone (e.g. QPython + requests). The phone detects
    physical impacts (table slams, door impacts) independently of the mic.
    Returns a higher-fidelity vibration signal.

Both modes return an anomaly score in [0.0, 1.0].

The Flask receiver for PHONE MODE runs in a background thread so it
never blocks the main detection loop.
"""

import logging
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Demo mode constants ───────────────────────────────────────────────────────
# RMS spike detection window — compare current RMS to rolling average
_DEMO_WINDOW_SIZE = 20          # samples in rolling baseline
_DEMO_SPIKE_RATIO = 2.5         # current / baseline ratio to count as spike
_DEMO_SCORE_DECAY = 0.85        # per-call score decay when no spike

# ── Phone mode constants ──────────────────────────────────────────────────────
_PHONE_ACCEL_SPIKE_THRESHOLD = 12.0   # m/s² above gravity baseline
_PHONE_SCORE_DECAY = 0.80


class DemoVibrationSensor:
    """
    Mic RMS proxy for vibration detection on laptop/desktop.

    Feed RMS values from AudioProcessor.raw_rms via update().
    Call score() to get the current anomaly score.
    """

    def __init__(self) -> None:
        self._rms_history: deque[float] = deque(maxlen=_DEMO_WINDOW_SIZE)
        self._current_score: float = 0.0
        self._lock = threading.Lock()

    def update(self, raw_rms: float) -> None:
        """
        Called by the main loop each time a new feature vector is produced.
        raw_rms is the un-normalized RMS from FeatureVector.raw_rms.
        """
        with self._lock:
            self._rms_history.append(raw_rms)

            if len(self._rms_history) < 5:
                # not enough history yet
                return

            baseline_rms = float(np.mean(list(self._rms_history)[:-1]))

            if baseline_rms < 1e-6:
                # silent environment — avoid div/0
                self._current_score *= _DEMO_SCORE_DECAY
                return

            spike_ratio = raw_rms / baseline_rms

            if spike_ratio >= _DEMO_SPIKE_RATIO:
                # map ratio to [0, 1] — ratio of 2.5 → ~0.5, ratio of 5.0 → ~1.0
                normalized = min((spike_ratio - _DEMO_SPIKE_RATIO) / _DEMO_SPIKE_RATIO, 1.0)
                self._current_score = max(self._current_score, normalized)
                logger.debug(f"Demo vibration spike: ratio={spike_ratio:.2f}, score={self._current_score:.3f}")
            else:
                self._current_score *= _DEMO_SCORE_DECAY

    def score(self) -> float:
        with self._lock:
            return float(np.clip(self._current_score, 0.0, 1.0))


class PhoneVibrationSensor:
    """
    Receives accelerometer readings from a companion Flask endpoint on a phone.

    The phone app POSTs JSON payloads like:
        {"x": 0.12, "y": 9.83, "z": 0.44, "timestamp": 1711234567.891}

    This class starts a Flask server in a background thread on startup.
    score() returns the current anomaly level derived from recent payloads.
    """

    def __init__(self, port: int = 5050) -> None:
        self._port = port
        self._score_queue: queue.Queue[float] = queue.Queue(maxsize=50)
        self._current_score: float = 0.0
        self._lock = threading.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._last_reading_time: float = 0.0

    def start(self) -> None:
        """Start the Flask receiver in a daemon thread."""
        self._server_thread = threading.Thread(
            target=self._run_flask,
            name="phone-vibration-server",
            daemon=True,
        )
        self._server_thread.start()
        logger.info(f"Phone vibration receiver started on port {self._port}")

    def _run_flask(self) -> None:
        try:
            from flask import Flask, request, jsonify
        except ImportError:
            logger.error("Flask not installed — phone sensor mode unavailable")
            return

        app = Flask("quietreach_sensor")
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)  # suppress Flask request logs

        @app.route("/sensor/accelerometer", methods=["POST"])
        def receive_accel():
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "no json"}), 400

            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            z = float(data.get("z", 0.0))

            # Total acceleration magnitude
            magnitude = float(np.sqrt(x**2 + y**2 + z**2))

            # Remove gravity (~9.81 m/s²) — we want dynamic acceleration
            dynamic = abs(magnitude - 9.81)

            anomaly = min(dynamic / _PHONE_ACCEL_SPIKE_THRESHOLD, 1.0)
            self._score_queue.put_nowait(anomaly)
            self._last_reading_time = time.monotonic()

            return jsonify({"received": True, "anomaly": anomaly}), 200

        @app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "ok"}), 200

        app.run(host="0.0.0.0", port=self._port, threaded=True)

    def update(self, raw_rms: float) -> None:  # noqa: ARG002
        """
        Called by main loop on each window. Drains pending readings
        from the Flask thread and updates current score.
        raw_rms is unused in phone mode — included for interface parity
        with DemoVibrationSensor.
        """
        with self._lock:
            # Drain incoming readings, keep the max anomaly from this batch
            max_incoming = 0.0
            drained = 0
            while not self._score_queue.empty():
                try:
                    val = self._score_queue.get_nowait()
                    max_incoming = max(max_incoming, val)
                    drained += 1
                except queue.Empty:
                    break

            if drained > 0:
                self._current_score = max(self._current_score, max_incoming)
                logger.debug(f"Phone vibration: {drained} readings, score={self._current_score:.3f}")
            else:
                # No new readings — decay score
                self._current_score *= _PHONE_SCORE_DECAY

                # If phone hasn't sent anything in 30s, warn once
                gap = time.monotonic() - self._last_reading_time
                if self._last_reading_time > 0 and gap > 30:
                    logger.warning(
                        f"No phone sensor data for {gap:.0f}s. "
                        "Is the companion app running?"
                    )

    def score(self) -> float:
        with self._lock:
            return float(np.clip(self._current_score, 0.0, 1.0))


def make_vibration_sensor(sensor_mode: str, phone_port: int = 5050):
    """
    Factory — returns the right sensor for the configured mode.

    sensor_mode: "demo" or "phone" (from config.sensor_mode)
    """
    if sensor_mode == "phone":
        sensor = PhoneVibrationSensor(port=phone_port)
        sensor.start()
        logger.info("Vibration sensor: PHONE MODE (accelerometer endpoint active)")
        return sensor
    else:
        logger.info("Vibration sensor: DEMO MODE (mic RMS proxy)")
        return DemoVibrationSensor()
