"""
alert/notification.py — Firebase Cloud Messaging push notification fallback

Used when SMS fails or as a secondary channel if FIREBASE_KEY is set.
Requires a companion mobile app registered with the same Firebase project.

This is a best-effort fallback — if Firebase isn't configured, it logs
and returns gracefully. The SMS path is primary.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from alert.location import LocationResult, format_location_for_sms

logger = logging.getLogger(__name__)


@dataclass
class PushResult:
    success: bool
    message_id: Optional[str]
    error: Optional[str]
    skipped: bool = False   # True if Firebase not configured


def send_push_notification(
    firebase_key: Optional[str],
    location: LocationResult,
    alert_time: str,
) -> PushResult:
    """
    Send a Firebase push notification as a secondary alert channel.

    If firebase_key is None or empty, skips silently and returns
    PushResult(skipped=True). This keeps the caller clean — it doesn't
    need to know whether Firebase is configured.
    """
    if not firebase_key:
        logger.debug("Firebase key not configured — skipping push notification")
        return PushResult(success=False, message_id=None, error=None, skipped=True)

    location_str = format_location_for_sms(location)

    try:
        import firebase_admin
        from firebase_admin import credentials, messaging

        # Initialize app only if not already done
        # firebase_admin raises ValueError if initialized twice
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_key)
            firebase_admin.initialize_app(cred)

        message = messaging.Message(
            notification=messaging.Notification(
                title="QuietReach Alert",
                body=f"Situation detected at {alert_time}",
            ),
            data={
                "location": location_str,
                "time": alert_time,
                "type": "threat_alert",
                # No audio content ever goes in here
            },
            topic="quietreach_alerts",  # device must subscribe to this topic
        )

        response = messaging.send(message)
        logger.info(f"Firebase push sent: {response}")
        return PushResult(success=True, message_id=response, error=None)

    except ImportError:
        logger.warning("firebase-admin not installed — push notification unavailable")
        return PushResult(success=False, message_id=None, error="firebase-admin not installed")

    except Exception as e:
        logger.error(f"Firebase push failed: {e}")
        return PushResult(success=False, message_id=None, error=str(e))
