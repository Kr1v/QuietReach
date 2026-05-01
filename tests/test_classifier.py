"""
tests/test_classifier.py — classifier and threshold scorer tests

Uses DummyClassifier so no model file is required.
Threshold scorer is tested with fixed inputs and known expected outputs.
"""

import datetime
import unittest

import numpy as np

from audio.processor import FEATURE_DIM


class TestDummyClassifier(unittest.TestCase):

    def test_returns_fixed_score(self):
        from model.classifier import DummyClassifier
        clf = DummyClassifier(fixed_score=0.55)
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.assertAlmostEqual(clf.predict(vec), 0.55)

    def test_backend_name(self):
        from model.classifier import DummyClassifier
        self.assertEqual(DummyClassifier().backend, "dummy")

    def test_load_classifier_raises_on_missing_files(self):
        from model.classifier import load_classifier
        with self.assertRaises(FileNotFoundError):
            load_classifier(
                tflite_path="/nonexistent/model.tflite",
                sklearn_path="/nonexistent/model.pkl",
            )


class TestThresholdScorer(unittest.TestCase):

    def test_output_in_range(self):
        from model.threshold import compute_threat_score
        result = compute_threat_score(audio_score=0.5, vibration_score=0.3)
        self.assertGreaterEqual(result.weighted_score, 0.0)
        self.assertLessEqual(result.weighted_score, 1.0)

    def test_zero_inputs_still_has_time_component(self):
        """Even with zero audio/vibration, time weight contributes."""
        from model.threshold import compute_threat_score
        result = compute_threat_score(audio_score=0.0, vibration_score=0.0)
        # Time score is always > 0 (day or night), so weighted > 0
        self.assertGreater(result.weighted_score, 0.0)

    def test_high_inputs_produce_high_score(self):
        from model.threshold import compute_threat_score
        result = compute_threat_score(audio_score=1.0, vibration_score=1.0)
        self.assertGreater(result.weighted_score, 0.8)

    def test_clamps_out_of_range_inputs(self):
        from model.threshold import compute_threat_score
        result = compute_threat_score(audio_score=2.0, vibration_score=-0.5)
        self.assertGreaterEqual(result.weighted_score, 0.0)
        self.assertLessEqual(result.weighted_score, 1.0)

    def test_night_score_higher_than_day(self):
        from model.threshold import compute_threat_score
        night = datetime.datetime.now().replace(hour=2, minute=0)
        day = datetime.datetime.now().replace(hour=14, minute=0)

        r_night = compute_threat_score(0.5, 0.3, now=night)
        r_day = compute_threat_score(0.5, 0.3, now=day)

        self.assertGreater(r_night.weighted_score, r_day.weighted_score)
        self.assertGreater(r_night.time_score, r_day.time_score)

    def test_components_match_weighted_sum(self):
        """Manually verify the weighted sum."""
        from model.threshold import compute_threat_score, _W_AUDIO, _W_VIBRATION, _W_TIME
        r = compute_threat_score(0.6, 0.4)
        expected = (
            _W_AUDIO * r.audio_score
            + _W_VIBRATION * r.vibration_score
            + _W_TIME * r.time_score
        )
        self.assertAlmostEqual(r.weighted_score, expected, places=5)


if __name__ == "__main__":
    unittest.main()
