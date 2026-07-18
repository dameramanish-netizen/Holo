"""Microphone capture -> StreamingTapDetector, using sounddevice (cross-platform,
works with WASAPI/MME on Windows)."""
from __future__ import annotations
import queue
import threading
from typing import Callable, List, Optional

import numpy as np

from .detector import StreamingTapDetector
from .signal_models import DetectedTap

BLOCK_SIZE = 512


class MicCapture:
    def __init__(self, sample_rate: int = 44100, preferred_channels: int = 2, device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.preferred_channels = preferred_channels
        self.device = device
        self.channels = 1          # resolved against the device in start()
        self.detector: Optional[StreamingTapDetector] = None
        self._stream = None
        self._on_tap: Optional[Callable[[DetectedTap], None]] = None
        self._on_level: Optional[Callable[[float], None]] = None
        self._q: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False

    def start(self, on_tap: Callable[[DetectedTap], None], on_level: Optional[Callable[[float], None]] = None):
        import sounddevice as sd
        self._on_tap = on_tap
        self._on_level = on_level
        self._running = True

        # Negotiate channel count against what the selected device actually
        # supports. Many laptop mic arrays only expose 2 channels (or 1) --
        # using fewer than the device offers silently kills the left/right
        # interchannel features the classifier relies on for left-vs-right
        # zones, so prefer the device's max, capped at preferred_channels.
        info = sd.query_devices(self.device, "input") if self.device is not None else sd.query_devices(kind="input")
        max_input_channels = max(int(info.get("max_input_channels", 1)), 1)
        self.channels = max(1, min(self.preferred_channels, max_input_channels))
        self.detector = StreamingTapDetector(sample_rate=self.sample_rate, channel_count=self.channels)

        def callback(indata, frames, time_info, status):
            self._q.put(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=BLOCK_SIZE,
            device=self.device,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    def _process_loop(self):
        while self._running:
            try:
                block = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            channels = [block[:, c] for c in range(block.shape[1])]
            if self._on_level:
                rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
                self._on_level(rms)
            taps = self.detector.process(channels)
            for tap in taps:
                if self._on_tap:
                    self._on_tap(tap)

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        with self._q.mutex:
            self._q.queue.clear()

    @staticmethod
    def list_input_devices():
        import sounddevice as sd
        devices = sd.query_devices()
        return [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
