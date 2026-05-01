"""
alert/sms.py — Twilio SMS alert sender

Sends a short, fixed-format message to the trusted contact number.
Message never contains audio descriptions, transcriptions, or
anything that could identify the source device user.

Retries once on failure with a 5-second delay.
"""

import logging
import time
from dataclasses import dataclass

from alert.location import LocationResult, format_location_for_sms

logger = logging.getLogger(__name__)

# Fixed message template — do not add audio content here ever.
# The receiver needs enough to act on, not a description of what happened.
_MESSAGE_TEMPLATE = (
    "QuietReach: Situation detected. "
    "Location: {location}. "
    "Time: {time}. "
    "This is an automated silent alert."
)


@dataclass
class SMSResult:
    success: bool
    sid: str | None        # Twilio message SID if successful
    error: str | None
    attempts: int


def send_alert_sms(
    twilio_sid: str,
    twilio_token: str,
    from_number: str,
    to_number: str,
    location: LocationResult,
    alert_time: str,
) -> SMSResult:
    """
    Send the alert SMS via Twilio. Retries once on failure.

    Args:
        twilio_sid:    Twilio account SID
        twilio_token:  Twilio auth token
        from_number:   Twilio source number (E.164 format)
        to_number:     Trusted contact number (E.164 format)
        location:      LocationResult from alert/location.py
        alert_time:    Human-readable time string for the message body

    Returns:
        SMSResult with success status and Twilio SID if sent.
    """
    location_str = format_location_for_sms(location)
    body = _MESSAGE_TEMPLATE.format(location=location_str, time=alert_time)

    logger.info(f"Sending SMS alert to {_mask_number(to_number)}")
    logger.debug(f"SMS body: {body}")

    for attempt in range(1, 3):  # max 2 attempts
        try:
            from twilio.rest import Client
            from twilio.base.exceptions import TwilioRestException

            client = Client(twilio_sid, twilio_token)
            message = client.messages.create(
                body=body,
                from_=from_number,
                to=to_number,
            )

            logger.info(f"SMS sent (SID={message.sid}, attempt={attempt})")
            return SMSResult(success=True, sid=message.sid, error=None, attempts=attempt)

        except TwilioRestException as e:
            logger.error(f"Twilio error (attempt {attempt}): {e.msg}")
            if attempt < 2:
                logger.info("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                return SMSResult(success=False, sid=None, error=str(e.msg), attempts=attempt)

        except ImportError:
            logger.error("twilio library not installed — cannot send SMS")
            return SMSResult(success=False, sid=None, error="twilio not installed", attempts=attempt)

        except Exception as e:
            logger.error(f"Unexpected SMS error (attempt {attempt}): {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                return SMSResult(success=False, sid=None, error=str(e), attempts=attempt)

    # unreachable but satisfies type checker
    return SMSResult(success=False, sid=None, error="max retries", attempts=2)


def _mask_number(number: str) -> str:
    """Mask phone number in logs — show only last 4 digits."""
    if len(number) <= 4:
        return "****"
    return f"****{number[-4:]}"
