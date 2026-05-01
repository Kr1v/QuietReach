"""
alert/location.py — location resolution for alert payloads

Primary: geocoder IP-based lookup (city-level, ~1km accuracy)
Fallback: returns a placeholder so the alert still sends even if
          location lookup fails entirely.

# IP location is ~1km accurate — good enough for city-level response.
# Real phone GPS would be integrated in v2 via the Flutter mobile wrapper.
# For now the trusted contact gets enough to know which neighborhood.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LocationResult:
    lat: Optional[float]
    lng: Optional[float]
    address: Optional[str]
    source: str          # "ip", "fallback"
    accuracy_note: str


def get_current_location() -> LocationResult:
    """
    Resolve approximate current location.

    Tries geocoder IP lookup first. If that fails (no internet, rate limit,
    library error) returns a graceful fallback so the alert still goes out —
    location-unknown alert is better than no alert.
    """
    try:
        import geocoder
        g = geocoder.ip("me")

        if g.ok and g.latlng:
            lat, lng = g.latlng
            address = g.address or f"{g.city}, {g.country}"
            logger.info(f"Location resolved: {address} ({lat:.4f}, {lng:.4f})")
            return LocationResult(
                lat=lat,
                lng=lng,
                address=address,
                source="ip",
                accuracy_note="IP-based, ~1km accuracy",
            )
        else:
            logger.warning(f"geocoder.ip returned ok=False: {g.status}")

    except ImportError:
        logger.error("geocoder library not installed")
    except Exception as e:
        logger.error(f"Location lookup failed: {e}")

    # Fallback — alert still goes out
    logger.warning("Sending alert without precise location")
    return LocationResult(
        lat=None,
        lng=None,
        address=None,
        source="fallback",
        accuracy_note="Location unavailable",
    )


def format_location_for_sms(loc: LocationResult) -> str:
    """
    Format location for inclusion in the SMS alert body.
    Keeps it short — SMS has a 160 char limit and the message
    template uses most of that.
    """
    if loc.lat is not None and loc.lng is not None:
        coord_str = f"{loc.lat:.4f}, {loc.lng:.4f}"
        if loc.address:
            return f"{loc.address} ({coord_str})"
        return coord_str
    return "Location unavailable"
