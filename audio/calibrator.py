"""
audio/calibrator.py — ambient noise baseline calibration

On startup, QuietReach listens to the environment for a few seconds
to establish what "normal" sounds like in this specific space.
That baseline is then used by processor.py to normalize incoming features.

Everything stays in memory. Nothing goes to disk.
"""

import logging
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Threshold above which we warn the user that ambient noise is too loud
# for reliable detection. ~65 dB equivalent in RMS terms.
_LOUD_AMBIENT_RMS_THRESHOLD = 0.08

# If std is near zero the environment is suspiciously quiet (anechoic chamber
# or a dead mic). Warn but don't block startup.
_SUSPICIOUSLY_QUIET_RMS = 0.0001


@dataclass
class AmbientBaseline:
    """
    Per-feature baseline computed during calibration.

    All arrays are shape (n_features,) — one mean/std per coefficient or
    scalar feature. Scalars (rms, zcr, centroid) are stored as 1-element arrays
    for uniform handling in processor.py.
    """
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.zeros(40))
    mfcc_std: np.ndarray = field(default_factory=lambda: np.ones(40))

    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 1.0

    zcr_mean: float = 0.0
    zcr_std: float = 1.0

    rms_mean: float = 0.0
    rms_std: float = 1.0

    # Populated after calibration — used for UI display
    ambient_rms_level: float = 0.0
    calibrated: bool = False
    calibration_duration_actual: float = 0.0


class Calibrator:
    """
    Listens to ambient audio and computes feature baselines.

    Call calibrate() once at startup. The returned AmbientBaseline
    is passed to AudioProcessor for normalization.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        mfcc_n: int = 40,
        calibration_duration: int = 10,
        window_size_seconds: int = 3,
    ) -> None:
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.mfcc_n = mfcc_n
        self.calibration_duration = calibration_duration
        self.window_size_seconds = window_size_seconds

    def calibrate(self) -> AmbientBaseline:
        """
        Collect audio for calibration_duration seconds, extract features,
        compute mean + std for each. Returns AmbientBaseline.

        Blocks the calling thread for calibration_duration seconds.
        Call this before starting the main detection loop.
        """
        logger.info(f"Calibrating ambient baseline ({self.calibration_duration}s)...")

        chunks_needed = int(
            (self.sample_rate / self.chunk_size) * self.calibration_duration
        )

        raw_chunks: list[bytes] = []
        start_time = time.monotonic()

        for _ in range(chunks_needed):
            try:
                chunk = self.audio_queue.get(timeout=2.0)
                raw_chunks.append(chunk)
            except queue.Empty:
                logger.warning("Calibration: audio queue empty, mic might not be running")
                break

        elapsed = time.monotonic() - start_time

        if not raw_chunks:
            logger.error("Calibration failed: no audio received")
            return AmbientBaseline()  # uncalibrated fallback

        # Convert all chunks to one float32 array
        audio_bytes = b"".join(raw_chunks)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        baseline = self._compute_baseline(audio)
        baseline.calibrated = True
        baseline.calibration_duration_actual = elapsed

        self._log_calibration_result(baseline)
        return baseline

    def _compute_baseline(self, audio: np.ndarray) -> AmbientBaseline:
        """
        Extract features from the full calibration audio and compute
        per-feature statistics.

        Uses sliding windows (matching processor.py window size) so the
        baseline stats reflect the same feature distribution the classifier sees.
        """
        window_samples = self.sample_rate * self.window_size_seconds  # matches processor.py window
        step = window_samples // 2             # 50% overlap during calibration

        mfcc_frames: list[np.ndarray] = []
        centroids: list[float] = []
        zcrs: list[float] = []
        rmss: list[float] = []

        i = 0
        while i + window_samples <= len(audio):
            window = audio[i : i + window_samples]

            # MFCCs
            mfcc = librosa.feature.mfcc(
                y=window, sr=self.sample_rate, n_mfcc=self.mfcc_n
            )
            mfcc_frames.append(np.mean(mfcc, axis=1))

            # Spectral centroid (mean across time)
            centroid = librosa.feature.spectral_centroid(y=window, sr=self.sample_rate)
            centroids.append(float(np.mean(centroid)))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(window)
            zcrs.append(float(np.mean(zcr)))

            # RMS energy
            rms = librosa.feature.rms(y=window)
            rmss.append(float(np.mean(rms)))

            i += step

        if not mfcc_frames:
            # Audio was too short for even one window — use whole clip
            logger.warning("Calibration audio shorter than one window; using full clip stats")
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.mfcc_n)
            mfcc_frames = [np.mean(mfcc, axis=1)]
            centroids = [float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)))]
            zcrs = [float(np.mean(librosa.feature.zero_crossing_rate(audio)))]
            rmss = [float(np.mean(librosa.feature.rms(y=audio)))]

        mfcc_arr = np.array(mfcc_frames)  # shape (n_windows, n_mfcc)

        baseline = AmbientBaseline(
            mfcc_mean=np.mean(mfcc_arr, axis=0),
            mfcc_std=np.std(mfcc_arr, axis=0) + 1e-8,  # epsilon avoids div/0
            spectral_centroid_mean=float(np.mean(centroids)),
            spectral_centroid_std=float(np.std(centroids)) + 1e-8,
            zcr_mean=float(np.mean(zcrs)),
            zcr_std=float(np.std(zcrs)) + 1e-8,
            rms_mean=float(np.mean(rmss)),
            rms_std=float(np.std(rmss)) + 1e-8,
            ambient_rms_level=float(np.mean(rmss)),
        )
        return baseline

    def _log_calibration_result(self, baseline: AmbientBaseline) -> None:
        rms = baseline.ambient_rms_level

        if rms > _LOUD_AMBIENT_RMS_THRESHOLD:
            logger.warning(
                f"Ambient noise level is high (RMS={rms:.4f}). "
                "Detection accuracy may be reduced. "
                "Try calibrating in a quieter moment, or raise THREAT_THRESHOLD in .env."
            )
        elif rms < _SUSPICIOUSLY_QUIET_RMS:
            logger.warning(
                f"Ambient RMS is suspiciously low ({rms:.6f}). "
                "Check that the microphone is working."
            )
        else:
            logger.info(
                f"Calibration complete. Ambient RMS={rms:.4f} — environment looks good."
            )

        logger.debug(
            f"MFCC baseline mean[:5]={baseline.mfcc_mean[:5].round(3)}, "
            f"centroid={baseline.spectral_centroid_mean:.1f}Hz, "
            f"zcr={baseline.zcr_mean:.4f}"
        )


def make_uncalibrated_baseline(mfcc_n: int = 40) -> AmbientBaseline:
    """
    Returns a do-nothing baseline (mean=0, std=1 everywhere).

    Used in tests or when calibration is explicitly skipped.
    Normalization becomes a no-op with these values.
    """
    return AmbientBaseline(
        mfcc_mean=np.zeros(mfcc_n),
        mfcc_std=np.ones(mfcc_n),
        spectral_centroid_mean=0.0,
        spectral_centroid_std=1.0,
        zcr_mean=0.0,
        zcr_std=1.0,
        rms_mean=0.0,
        rms_std=1.0,
        ambient_rms_level=0.0,
        calibrated=False,
    )
