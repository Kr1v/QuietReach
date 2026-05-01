"""
tests/test_audio.py — unit tests for audio pipeline

Tests calibrator, processor, and feature vector shape/layout.
No microphone required — uses synthetic numpy audio.
"""

import queue
import struct
import unittest

import numpy as np


def _make_audio_bytes(duration_seconds: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM int16 audio as bytes."""
    n_samples = int(duration_seconds * sample_rate)
    samples = np.zeros(n_samples, dtype=np.int16)
    return samples.tobytes()


def _make_noise_bytes(duration_seconds: float = 3.0, sample_rate: int = 16000, amplitude: float = 0.1) -> bytes:
    """Generate white noise PCM audio."""
    n_samples = int(duration_seconds * sample_rate)
    rng = np.random.default_rng(42)
    samples = (rng.standard_normal(n_samples) * amplitude * 32767).astype(np.int16)
    return samples.tobytes()


def _fill_queue(q: queue.Queue, audio_bytes: bytes, chunk_size: int = 1024) -> None:
    """Split audio bytes into chunks and fill a queue."""
    for i in range(0, len(audio_bytes) - chunk_size, chunk_size):
        q.put(audio_bytes[i : i + chunk_size])


class TestCalibrator(unittest.TestCase):

    def test_calibrate_returns_baseline(self):
        from audio.calibrator import Calibrator
        q = queue.Queue()
        audio = _make_noise_bytes(duration_seconds=12.0)
        _fill_queue(q, audio)

        cal = Calibrator(audio_queue=q, calibration_duration=10)
        baseline = cal.calibrate()

        self.assertTrue(baseline.calibrated)
        self.assertEqual(baseline.mfcc_mean.shape, (40,))
        self.assertEqual(baseline.mfcc_std.shape, (40,))
        # std should be non-zero (epsilon was added)
        self.assertTrue(np.all(baseline.mfcc_std > 0))

    def test_calibrate_empty_queue_returns_uncalibrated(self):
        from audio.calibrator import Calibrator
        q = queue.Queue()
        # Don't add anything — queue is empty

        cal = Calibrator(audio_queue=q, calibration_duration=2)
        baseline = cal.calibrate()

        self.assertFalse(baseline.calibrated)

    def test_make_uncalibrated_baseline_shape(self):
        from audio.calibrator import make_uncalibrated_baseline
        b = make_uncalibrated_baseline(mfcc_n=40)
        self.assertFalse(b.calibrated)
        self.assertEqual(b.mfcc_mean.shape, (40,))
        self.assertTrue(np.all(b.mfcc_std == 1.0))


class TestAudioProcessor(unittest.TestCase):

    def _make_processor(self, audio_bytes: bytes):
        from audio.calibrator import make_uncalibrated_baseline
        from audio.processor import AudioProcessor

        q = queue.Queue(maxsize=200)
        _fill_queue(q, audio_bytes)
        baseline = make_uncalibrated_baseline()
        return AudioProcessor(audio_queue=q, baseline=baseline), q

    def test_feature_vector_shape(self):
        from audio.processor import FEATURE_DIM
        audio = _make_noise_bytes(duration_seconds=6.0)
        proc, _ = self._make_processor(audio)

        fv = proc.next_feature_vector(timeout=1.0)
        self.assertIsNotNone(fv)
        self.assertEqual(fv.vec.shape, (FEATURE_DIM,))

    def test_feature_vector_dtype(self):
        audio = _make_noise_bytes(duration_seconds=6.0)
        proc, _ = self._make_processor(audio)
        fv = proc.next_feature_vector(timeout=1.0)
        self.assertIsNotNone(fv)
        self.assertEqual(fv.vec.dtype, np.float32)

    def test_raw_rms_positive(self):
        """Raw RMS should be > 0 for non-silent audio."""
        audio = _make_noise_bytes(duration_seconds=6.0, amplitude=0.05)
        proc, _ = self._make_processor(audio)
        fv = proc.next_feature_vector(timeout=1.0)
        self.assertIsNotNone(fv)
        self.assertGreater(fv.raw_rms, 0.0)

    def test_silent_audio_low_rms(self):
        audio = _make_audio_bytes(duration_seconds=6.0)
        proc, _ = self._make_processor(audio)
        fv = proc.next_feature_vector(timeout=1.0)
        self.assertIsNotNone(fv)
        self.assertAlmostEqual(fv.raw_rms, 0.0, places=4)

    def test_sliding_window_produces_multiple_vectors(self):
        """12 seconds of audio should yield at least 3 feature vectors."""
        audio = _make_noise_bytes(duration_seconds=12.0)
        proc, _ = self._make_processor(audio)

        vectors = []
        for _ in range(6):
            fv = proc.next_feature_vector(timeout=1.0)
            if fv is not None:
                vectors.append(fv)

        self.assertGreaterEqual(len(vectors), 3)

    def test_buffer_fullness_property(self):
        from audio.calibrator import make_uncalibrated_baseline
        from audio.processor import AudioProcessor

        q = queue.Queue(maxsize=200)
        baseline = make_uncalibrated_baseline()
        proc = AudioProcessor(audio_queue=q, baseline=baseline)

        self.assertEqual(proc.buffer_fullness, 0.0)

        # Add partial audio
        audio = _make_noise_bytes(duration_seconds=1.5)
        _fill_queue(q, audio)
        proc.drain_queue_into_buffer()

        self.assertGreater(proc.buffer_fullness, 0.0)
        self.assertLessEqual(proc.buffer_fullness, 1.0)


class TestFeatureVectorLayout(unittest.TestCase):
    """
    Sanity check that FEATURE_DIM and named accessors are consistent
    with what trainer.py expects.
    """

    def test_mfcc_slice(self):
        from audio.processor import FEATURE_DIM, MFCC_N, FeatureVector
        vec = np.arange(FEATURE_DIM, dtype=np.float32)
        fv = FeatureVector(vec=vec, raw_rms=0.01, window_timestamp=0.0)
        np.testing.assert_array_equal(fv.mfcc, vec[:MFCC_N])

    def test_scalar_accessors(self):
        from audio.processor import FEATURE_DIM, MFCC_N, FeatureVector
        vec = np.ones(FEATURE_DIM, dtype=np.float32)
        vec[MFCC_N] = 2.0
        vec[MFCC_N + 1] = 3.0
        vec[MFCC_N + 2] = 4.0
        fv = FeatureVector(vec=vec, raw_rms=0.0, window_timestamp=0.0)
        self.assertAlmostEqual(fv.spectral_centroid, 2.0)
        self.assertAlmostEqual(fv.zcr, 3.0)
        self.assertAlmostEqual(fv.rms, 4.0)


if __name__ == "__main__":
    unittest.main()
