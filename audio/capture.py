"""
audio/capture.py — microphone stream handler

Opens a PyAudio stream and feeds raw chunks into a queue.
Runs on its own thread so the main loop never blocks waiting for audio.

Nothing here touches disk. Chunks go into the queue and nowhere else.
"""

import logging
import queue
import threading
from typing import Optional

import pyaudio

logger = logging.getLogger(__name__)

# These match what the model was trained on — don't change without retraining
_FORMAT = pyaudio.paInt16
_CHANNELS = 1
_RATE = 16000
_CHUNK = 1024


class MicCapture:
    """
    Wraps a PyAudio input stream and feeds audio chunks into a queue.

    Usage:
        cap = MicCapture(audio_queue)
        cap.start()
        # ... do stuff ...
        cap.stop()
    """

    def __init__(
        self,
        out_queue: queue.Queue,
        sample_rate: int = _RATE,
        chunk_size: int = _CHUNK,
        device_index: Optional[int] = None,
    ) -> None:
        self.out_queue = out_queue
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index  # None = system default

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self) -> None:
        """Open mic and start capture thread."""
        if self._running:
            logger.warning("MicCapture.start() called but already running")
            return

        self._pa = pyaudio.PyAudio()

        try:
            self._stream = self._pa.open(
                format=_FORMAT,
                channels=_CHANNELS,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )
        except OSError as e:
            logger.error(f"Failed to open microphone: {e}")
            self._pa.terminate()
            raise

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="mic-capture",
            daemon=True,  # dies with main process, no cleanup needed
        )
        self._thread.start()
        logger.info(f"Mic capture started (rate={self.sample_rate}, chunk={self.chunk_size})")

    def stop(self) -> None:
        """Signal capture thread to stop and clean up PyAudio resources."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("Capture thread didn't stop cleanly within timeout")

        self._cleanup_stream()
        logger.info("Mic capture stopped")

    def _cleanup_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except OSError:
                pass  # already closed, fine
            self._stream = None

        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _capture_loop(self) -> None:
        """
        Read chunks from mic and push to queue.

        Runs in dedicated thread. On any unrecoverable read error,
        logs and exits — main loop will notice queue going dry.
        """
        logger.debug("Capture loop running")

        while not self._stop_event.is_set():
            try:
                chunk = self._stream.read(self.chunk_size, exception_on_overflow=False)
                # exception_on_overflow=False: skip frames on overflow instead of crashing.
                # A few dropped frames during a high-load spike beats a fatal exception.
                self.out_queue.put(chunk)
            except OSError as e:
                logger.error(f"Mic read error: {e} — capture stopping")
                break
            except Exception as e:
                # unexpected — log and keep going once, break on repeat
                logger.exception(f"Unexpected error in capture loop: {e}")
                break

        logger.debug("Capture loop exited")

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def list_devices() -> list[dict]:
        """
        Utility: list available audio input devices.
        Useful during setup if default mic isn't the right one.
        """
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "sample_rate": int(info["defaultSampleRate"]),
                })
        pa.terminate()
        return devices


def make_capture_queue() -> queue.Queue:
    """
    Returns a bounded queue for audio chunks.

    Bounded at 100 chunks (~6 seconds at 16kHz/1024) so a slow processor
    doesn't silently accumulate minutes of backlog in RAM.
    If full, capture loop will block — that's intentional backpressure.
    """
    return queue.Queue(maxsize=100)
