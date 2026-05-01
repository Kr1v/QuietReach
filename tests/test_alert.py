"""
tests/test_alert.py — dispatcher, SMS formatting, and encryptor tests

No external services called. SMS/push are mocked.
Dispatcher is tested with a controlled confirmation source so
timing behavior is deterministic.
"""

import datetime
import time
import unittest
from unittest.mock import MagicMock, patch

from model.threshold import ThreatComponents


def _make_components(score: float) -> ThreatComponents:
    return ThreatComponents(
        audio_score=score,
        vibration_score=0.0,
        time_score=0.25,
        weighted_score=score,
        hour=14,
    )


def _make_cfg(threshold: float = 0.72, consecutive: int = 8, cooldown: int = 10):
    from config import QuietReachConfig
    cfg = QuietReachConfig(
        twilio_sid="test",
        twilio_token="test",
        twilio_from="+10000000000",
        trusted_number="+10000000001",
        # encryption_key intentionally blank here — dispatcher tests never
        # touch the encryptor. TestEncryptor generates its own valid key.
        encryption_key="",
        threat_threshold=threshold,
        consecutive_seconds_required=consecutive,
        alert_cooldown_minutes=cooldown,
        window_size_seconds=3,
    )
    return cfg


class TestAlertDispatcher(unittest.TestCase):

    def _make_dispatcher(self, threshold=0.72, consecutive=8, callback=None, confirmation_score=None):
        from alert.dispatcher import AlertDispatcher
        cfg = _make_cfg(threshold=threshold, consecutive=consecutive)
        cb = callback or MagicMock()
        conf_source = (lambda: confirmation_score) if confirmation_score is not None else None
        return AlertDispatcher(cfg=cfg, alert_callback=cb, confirmation_source=conf_source), cb

    def test_no_alert_below_threshold(self):
        dispatcher, cb = self._make_dispatcher(threshold=0.72)
        for _ in range(20):
            dispatcher.on_score(_make_components(0.5), window_duration_seconds=1.5)
        cb.assert_not_called()

    def test_no_alert_before_consecutive_time_met(self):
        dispatcher, cb = self._make_dispatcher(threshold=0.5, consecutive=8, confirmation_score=0.9)
        # 3 windows × 1.5s = 4.5s — not enough
        for _ in range(3):
            fired = dispatcher.on_score(_make_components(0.8), window_duration_seconds=1.5)
            self.assertFalse(fired)
        cb.assert_not_called()

    def test_alert_fires_after_consecutive_time_met(self):
        dispatcher, cb = self._make_dispatcher(threshold=0.5, consecutive=6, confirmation_score=0.9)
        fired = False
        for _ in range(10):
            result = dispatcher.on_score(_make_components(0.8), window_duration_seconds=1.5)
            if result:
                fired = True
                break
        self.assertTrue(fired)
        cb.assert_called_once()

    def test_consecutive_counter_resets_on_sub_threshold(self):
        dispatcher, cb = self._make_dispatcher(threshold=0.5, consecutive=6, confirmation_score=0.9)
        # Build up some consecutive time
        for _ in range(3):
            dispatcher.on_score(_make_components(0.8), window_duration_seconds=1.5)
        self.assertGreater(dispatcher.consecutive_seconds, 0)

        # One sub-threshold window resets it
        dispatcher.on_score(_make_components(0.3), window_duration_seconds=1.5)
        self.assertEqual(dispatcher.consecutive_seconds, 0.0)
        cb.assert_not_called()

    def test_confirmation_suppresses_alert(self):
        """If confirmation score drops below threshold, alert is suppressed."""
        dispatcher, cb = self._make_dispatcher(
            threshold=0.5,
            consecutive=4,
            confirmation_score=0.3,   # below threshold
        )
        for _ in range(10):
            dispatcher.on_score(_make_components(0.8), window_duration_seconds=1.5)
        cb.assert_not_called()

    def test_cooldown_prevents_second_alert(self):
        dispatcher, cb = self._make_dispatcher(
            threshold=0.3, consecutive=2, confirmation_score=0.9
        )
        # Fire first alert
        for _ in range(5):
            dispatcher.on_score(_make_components(0.9), window_duration_seconds=1.5)
        self.assertEqual(cb.call_count, 1)

        # Reset consecutive to simulate new high-score run
        # (dispatcher is in cooldown — should not fire again)
        for _ in range(20):
            dispatcher.on_score(_make_components(0.9), window_duration_seconds=1.5)
        self.assertEqual(cb.call_count, 1)  # still just 1

    def test_cooldown_remaining_decreases(self):
        dispatcher, _ = self._make_dispatcher(threshold=0.3, consecutive=2, confirmation_score=0.9)
        for _ in range(5):
            dispatcher.on_score(_make_components(0.9), window_duration_seconds=1.5)
        r1 = dispatcher.cooldown_remaining()
        time.sleep(0.1)
        r2 = dispatcher.cooldown_remaining()
        self.assertGreater(r1, r2)


class TestSMSFormatting(unittest.TestCase):

    def test_format_location_with_coords(self):
        from alert.location import LocationResult, format_location_for_sms
        loc = LocationResult(
            lat=37.7749, lng=-122.4194,
            address="San Francisco, CA",
            source="ip",
            accuracy_note="~1km"
        )
        result = format_location_for_sms(loc)
        self.assertIn("37.7749", result)
        self.assertIn("-122.4194", result)

    def test_format_location_fallback(self):
        from alert.location import LocationResult, format_location_for_sms
        loc = LocationResult(lat=None, lng=None, address=None, source="fallback", accuracy_note="")
        result = format_location_for_sms(loc)
        self.assertIn("unavailable", result.lower())

    def test_sms_body_has_no_audio_content(self):
        """Smoke test: message template should never contain audio-related words."""
        from alert.sms import _MESSAGE_TEMPLATE
        body = _MESSAGE_TEMPLATE.format(location="test", time="2024-01-01 12:00:00")
        forbidden = ["audio", "sound", "recording", "wav", "noise", "voice", "crying", "impact"]
        for word in forbidden:
            self.assertNotIn(word, body.lower(), f"SMS body contains forbidden word: {word}")


class TestEncryptor(unittest.TestCase):

    def _make_encryptor(self):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        from privacy.encryptor import PayloadEncryptor
        return PayloadEncryptor(key)

    def test_encrypt_decrypt_roundtrip(self):
        enc = self._make_encryptor()
        payload = {"lat": 37.77, "lng": -122.41, "time": "12:00:00"}
        token = enc.encrypt_payload(payload)
        result = enc.decrypt_payload(token)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["lat"], 37.77)

    def test_bad_token_returns_none(self):
        enc = self._make_encryptor()
        result = enc.decrypt_payload("this-is-not-a-valid-fernet-token")
        self.assertIsNone(result)

    def test_alert_payload_has_no_audio_keys(self):
        enc = self._make_encryptor()
        token = enc.build_alert_payload(37.77, -122.41, "SF", "12:00", "ip")
        payload = enc.decrypt_payload(token)
        audio_keys = {"audio", "waveform", "recording", "sound", "transcript"}
        self.assertTrue(audio_keys.isdisjoint(set(payload.keys())))

    def test_invalid_key_raises(self):
        from privacy.encryptor import PayloadEncryptor
        with self.assertRaises(ValueError):
            PayloadEncryptor("not-a-valid-key")


class TestMemoryCleaner(unittest.TestCase):

    def test_wipe_zeros_array(self):
        import numpy as np
        from privacy.memory_cleaner import wipe_audio_buffer
        buf = np.ones(1024, dtype=np.float32)
        wipe_audio_buffer(buf)
        self.assertTrue(np.all(buf == 0))

    def test_wipe_bytearray(self):
        from privacy.memory_cleaner import wipe_bytes_buffer
        buf = bytearray(b"\xff" * 512)
        wipe_bytes_buffer(buf)
        self.assertTrue(all(b == 0 for b in buf))

    def test_wipe_empty_array_no_crash(self):
        import numpy as np
        from privacy.memory_cleaner import wipe_audio_buffer
        buf = np.array([], dtype=np.float32)
        wipe_audio_buffer(buf)  # should not raise


if __name__ == "__main__":
    unittest.main()
