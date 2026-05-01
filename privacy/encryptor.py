"""
privacy/encryptor.py — location payload encryption

Encrypts the location + timestamp before transmission using Fernet
symmetric encryption (AES-128-CBC + HMAC-SHA256).

# considered asymmetric (RSA/EC) but it's overkill for MVP:
# the trusted contact needs the key anyway to decrypt, and managing
# a keypair adds friction without meaningful security gain at this scale.
# noted in roadmap for v2 — asymmetric makes sense if we ever store
# alerts server-side rather than SMS direct-to-contact.

The encryption key is stored in .env and loaded via config.py.
It is never hardcoded and never logged.
"""

import base64
import json
import logging
from dataclasses import asdict
from typing import Optional

logger = logging.getLogger(__name__)


class PayloadEncryptor:
    """
    Wraps Fernet to encrypt/decrypt alert payloads.

    Initialize once at startup with the key from config.encryption_key.
    The key must be a valid Fernet key — 32 url-safe base64 bytes.
    Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    """

    def __init__(self, key: str) -> None:
        try:
            from cryptography.fernet import Fernet, InvalidToken
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
            self._InvalidToken = InvalidToken
        except ImportError:
            raise RuntimeError("cryptography library not installed")
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}") from e

    def encrypt_payload(self, payload: dict) -> str:
        """
        Serialize payload dict to JSON and encrypt.
        Returns a url-safe base64 string (the Fernet token).
        """
        raw = json.dumps(payload, default=str).encode("utf-8")
        token = self._fernet.encrypt(raw)
        return token.decode("utf-8")

    def decrypt_payload(self, token: str) -> Optional[dict]:
        """
        Decrypt a Fernet token back to a dict.
        Returns None on decryption failure (wrong key, tampered data).
        """
        try:
            raw = self._fernet.decrypt(token.encode("utf-8"))
            return json.loads(raw.decode("utf-8"))
        except self._InvalidToken:
            logger.error("Decryption failed: invalid token or wrong key")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Decryption succeeded but JSON parse failed: {e}")
            return None

    def build_alert_payload(
        self,
        lat: Optional[float],
        lng: Optional[float],
        address: Optional[str],
        alert_time: str,
        location_source: str,
    ) -> str:
        """
        Build and encrypt the standard alert payload.

        Only location + time metadata go in here.
        No audio data, no waveforms, no transcriptions — ever.
        """
        payload = {
            "lat": lat,
            "lng": lng,
            "address": address,
            "alert_time": alert_time,
            "location_source": location_source,
            "app": "quietreach",
            "version": "1.0",
        }
        return self.encrypt_payload(payload)
