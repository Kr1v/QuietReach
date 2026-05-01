"""
privacy/memory_cleaner.py — secure audio buffer zeroing

Python's garbage collector does not guarantee when (or whether) memory
containing sensitive data gets overwritten. For a safety app processing
ambient audio, this matters: a memory dump or swap file could in theory
expose audio content after the process has "discarded" the buffer.

This module explicitly overwrites audio buffers with zeros before
releasing them, reducing the window during which raw audio lives
in addressable memory.

Note on limitations: This is a best-effort mitigation. CPython may
keep references to the underlying memory longer than expected (interning,
buffer protocol, etc.), and OS-level swap files are outside our control.
Full mitigation would require running on an encrypted swap partition and
using ctypes to zero C-level buffers directly. That's roadmap territory.
The zeroing here covers the most common case: the numpy array the
processor holds during a window.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def wipe_audio_buffer(buf: np.ndarray) -> None:
    """
    Overwrite a numpy audio buffer with zeros in-place.

    Call this immediately after feature extraction is complete and
    before releasing the reference. The array must be writable.

    Args:
        buf: numpy array containing raw audio samples (any dtype).
             Must be C-contiguous for the fill to reliably cover
             all underlying memory.
    """
    if not isinstance(buf, np.ndarray):
        logger.warning(f"wipe_audio_buffer: expected ndarray, got {type(buf).__name__}")
        return

    if buf.size == 0:
        return

    try:
        buf.fill(0)
        logger.debug(f"Wiped audio buffer ({buf.nbytes} bytes)")
    except ValueError:
        # Array is read-only (e.g. from np.frombuffer on a bytes object)
        # Can't wipe it in-place — log and move on. The bytes object
        # itself will be GC'd, just not overwritten.
        logger.debug("Audio buffer is read-only — cannot wipe in-place (bytes-backed array)")


def wipe_bytes_buffer(buf: bytearray) -> None:
    """
    Zero a bytearray in-place. Use this for raw PCM chunk accumulation buffers.

    bytes objects are immutable and cannot be zeroed — use bytearray
    for any audio data you want to be able to wipe.
    """
    if not isinstance(buf, bytearray):
        logger.warning(f"wipe_bytes_buffer: expected bytearray, got {type(buf).__name__}")
        return

    for i in range(len(buf)):
        buf[i] = 0

    logger.debug(f"Wiped bytearray buffer ({len(buf)} bytes)")


def wipe_and_release(buf: np.ndarray) -> None:
    """
    Zero the buffer, then return. The caller should immediately
    drop their reference (e.g. del audio or audio = None).

    Convenience wrapper used in AudioProcessor after each window.

    Example:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).copy()
        features = extract(audio)
        wipe_and_release(audio)
        del audio
    """
    wipe_audio_buffer(buf)
    # Nothing else to do here — caller drops the ref.
    # Keeping this as a named function rather than inlining makes
    # grepping for wipe calls easy during security review.
