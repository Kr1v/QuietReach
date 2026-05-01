"""
audio/processor.py — feature extraction pipeline

Pulls raw audio chunks from the capture queue, assembles them into
3-second windows, extracts features with librosa, normalizes against
the calibrated ambient baseline, and returns a flat feature vector
ready for the classifier.

# normalization against baseline was the key fix — raw features were
# garbage in noisy apartments. same threshold was triggering on TV audio
# in one place and missing shouting in another. calibrated z-scores made
# it consistent across environments.
"""

import logging
import queue
import time
from typing import Optional

import librosa
import numpy as np

from audio.calibrator import AmbientBaseline

logger = logging.getLogger(__name__)


# ── Feature vector layout (documented here, referenced in classifier.py) ────
#
#   [0:40]   — MFCC coefficients (40), mean-pooled over time axis
#   [40]     — spectral centroid (normalized)
#   [41]     — zero crossing rate (normalized)
#   [42]     — RMS energy (normalized)
#
# Total: 43 features. Keep this stable — changing it breaks saved models.
FEATURE_DIM = 43
MFCC_N = 40


class FeatureVector:
    """
    Thin wrapper around a numpy array so callers get named access
    alongside the raw array the classifier needs.
    """

    def __init__(self, vec: np.ndarray, raw_rms: float, window_timestamp: float) -> None:
        assert vec.shape == (FEATURE_DIM,), f"Expected ({FEATURE_DIM},), got {vec.shape}"
        self.vec = vec
        self.raw_rms = raw_rms          # un-normalized, used by UI and vibration proxy
        self.window_timestamp = window_timestamp

    @property
    def mfcc(self) -> np.ndarray:
        return self.vec[:MFCC_N]

    @property
    def spectral_centroid(self) -> float:
        return float(self.vec[MFCC_N])

    @property
    def zcr(self) -> float:
        return float(self.vec[MFCC_N + 1])

    @property
    def rms(self) -> float:
        return float(self.vec[MFCC_N + 2])


class AudioProcessor:
    """
    Accumulates raw audio chunks into fixed-size windows, extracts
    features, normalizes them, returns FeatureVector objects.

    Designed to be called in a tight loop:

        while running:
            fv = processor.next_feature_vector()
            if fv is not None:
                score = classifier.predict(fv)
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        baseline: AmbientBaseline,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        window_seconds: int = 3,
        mfcc_n: int = MFCC_N,
    ) -> None:
        self.audio_queue = audio_queue
        self.baseline = baseline
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.window_samples = sample_rate * window_seconds
        self.mfcc_n = mfcc_n

        # rolling accumulation buffer
        self._buffer: list[bytes] = []
        self._buffer_samples = 0
        self._chunks_per_window = self.window_samples // chunk_size

        # sliding window: 50% step — each window shares half its audio
        # with the next. keeps threat meter responsive without over-counting.
        self._step_samples = self.window_samples // 2
        self._step_chunks = self._step_samples // chunk_size

        # holds leftover chunks after stepping forward
        self._leftover: list[bytes] = []

        # for inference timing in debug mode
        self._last_extraction_ms: float = 0.0

    def drain_queue_into_buffer(self, max_chunks: int = 32) -> int:
        """
        Non-blocking drain of the audio queue into internal buffer.
        Returns number of chunks consumed.

        max_chunks cap prevents starvation — we process what's available,
        not everything that's ever accumulated.
        """
        consumed = 0
        while consumed < max_chunks:
            try:
                chunk = self.audio_queue.get_nowait()
                self._buffer.append(chunk)
                self._buffer_samples += self.chunk_size
                consumed += 1
            except queue.Empty:
                break
        return consumed

    def next_feature_vector(self, timeout: float = 0.5) -> Optional[FeatureVector]:
        """
        Block until we have enough audio for a full window, then extract
        and return a FeatureVector.

        Returns None if audio stops arriving (mic error, shutdown, etc.)

        This is the main call site — processor.next_feature_vector() in a loop.
        """
        # First try to top up buffer from queue
        deadline = time.monotonic() + timeout
        while self._buffer_samples < self.window_samples:
            try:
                chunk = self.audio_queue.get(timeout=0.05)
                self._buffer.append(chunk)
                self._buffer_samples += self.chunk_size
            except queue.Empty:
                if time.monotonic() > deadline:
                    logger.debug("next_feature_vector: timed out waiting for audio")
                    return None

        # We have enough — extract features from the front of buffer
        window_chunks = self._buffer[: self._chunks_per_window]
        audio_bytes = b"".join(window_chunks)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        t0 = time.monotonic()
        fv = self._extract(audio)
        self._last_extraction_ms = (time.monotonic() - t0) * 1000
        logger.debug(f"Feature extraction: {self._last_extraction_ms:.1f}ms")

        # Slide forward by step_chunks — discard oldest half of buffer
        self._buffer = self._buffer[self._step_chunks :]
        self._buffer_samples = len(self._buffer) * self.chunk_size

        return fv

    def _extract(self, audio: np.ndarray) -> FeatureVector:
        """
        Core feature extraction. All librosa calls live here.

        Returns normalized FeatureVector. Raw RMS is preserved
        un-normalized for the UI level meter.
        """
        # Raw RMS before normalization — used by UI and vibration proxy in demo mode
        raw_rms = float(np.sqrt(np.mean(audio ** 2)))

        # ── MFCCs ──────────────────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.mfcc_n,
            n_fft=512,
            hop_length=256,
        )
        mfcc_mean = np.mean(mfcc, axis=1)  # (n_mfcc,)

        # ── Spectral centroid ───────────────────────────────────────────────
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=512, hop_length=256
        )
        centroid_mean = float(np.mean(centroid))

        # ── Zero crossing rate ──────────────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=256)
        zcr_mean = float(np.mean(zcr))

        # ── RMS energy ──────────────────────────────────────────────────────
        rms = librosa.feature.rms(y=audio, hop_length=256)
        rms_mean = float(np.mean(rms))

        # ── Normalize against baseline (z-score) ───────────────────────────
        norm_mfcc = (mfcc_mean - self.baseline.mfcc_mean) / self.baseline.mfcc_std

        norm_centroid = (centroid_mean - self.baseline.spectral_centroid_mean) / \
            self.baseline.spectral_centroid_std

        norm_zcr = (zcr_mean - self.baseline.zcr_mean) / self.baseline.zcr_std

        norm_rms = (rms_mean - self.baseline.rms_mean) / self.baseline.rms_std

        # ── Assemble final vector ───────────────────────────────────────────
        vec = np.concatenate([
            norm_mfcc,                          # [0:40]
            [norm_centroid, norm_zcr, norm_rms] # [40, 41, 42]
        ]).astype(np.float32)

        return FeatureVector(
            vec=vec,
            raw_rms=raw_rms,
            window_timestamp=time.monotonic(),
        )

    def update_baseline(self, new_baseline: AmbientBaseline) -> None:
        """
        Hot-swap the ambient baseline without restarting.
        Used if the user triggers a re-calibration mid-session.
        """
        self.baseline = new_baseline
        logger.info("AudioProcessor: baseline updated")

    @property
    def last_extraction_ms(self) -> float:
        """Inference timing — exposed for debug UI."""
        return self._last_extraction_ms

    @property
    def buffer_fullness(self) -> float:
        """0.0–1.0 indicating how full the accumulation buffer is."""
        return min(self._buffer_samples / self.window_samples, 1.0)
