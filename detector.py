"""Low-allocation streaming onset detector. Ported from StreamingTapDetector.swift."""
from __future__ import annotations
import math
from typing import List, Optional

import numpy as np

from .signal_models import DetectedTap
from . import gate


class StreamingTapDetector:
    def __init__(
        self,
        sample_rate: float,
        channel_count: int,
        analysis_duration: float = 0.090,
        pre_roll_duration: float = 0.012,
        warm_up_duration: float = 0.75,
        initial_noise_floor_rms: float = 0.0005,
    ):
        self.sample_rate = sample_rate
        self.channel_count = max(channel_count, 1)
        self.analysis_window_samples = max(int(sample_rate * analysis_duration), 1024)
        self.pre_roll_samples = max(int(sample_rate * pre_roll_duration), 128)
        self.warm_up_samples = max(int(sample_rate * warm_up_duration), 0)
        self.initial_noise_floor_rms = max(initial_noise_floor_rms, 1e-5)
        self.noise_floor_rms = self.initial_noise_floor_rms
        self.total_samples = 0
        self._pre_roll: List[List[float]] = [[] for _ in range(self.channel_count)]
        self._capture: Optional[List[List[float]]] = None
        self._capture_onset_offset = 0
        self._capture_stream_index = 0
        self._capture_noise_floor = 0.0
        self._refractory_remaining = 0
        self._adapt_during_refractory = False
        self._warm_up_remaining = self.warm_up_samples
        self._onset_filter_state = [0.0, 0.0, 0.0, 0.0]

    def reset(self):
        self.total_samples = 0
        self.noise_floor_rms = self.initial_noise_floor_rms
        self._pre_roll = [[] for _ in range(self.channel_count)]
        self._capture = None
        self._refractory_remaining = 0
        self._adapt_during_refractory = False
        self._warm_up_remaining = self.warm_up_samples
        self._onset_filter_state = [0.0, 0.0, 0.0, 0.0]

    def process(self, incoming: List[np.ndarray]) -> List[DetectedTap]:
        if not incoming:
            return []
        frame_count = min(len(c) for c in incoming)
        if frame_count <= 0:
            return []

        channels = self._normalized_channels(incoming, frame_count)
        mono = self._mix_down(channels)
        onset_signal = self._low_pass_for_onset(mono)

        rms = self._rms(onset_signal)
        peak = float(np.max(np.abs(onset_signal))) if onset_signal.size else 0.0

        try:
            if self._warm_up_remaining > 0:
                self._adapt_noise_floor(rms, is_warm_up=True)
                self._warm_up_remaining = max(0, self._warm_up_remaining - frame_count)
                self._append_to_pre_roll(channels)
                return []

            if self._refractory_remaining > 0:
                if self._adapt_during_refractory:
                    self._adapt_noise_floor(rms, is_warm_up=False)
                self._refractory_remaining = max(0, self._refractory_remaining - frame_count)
                if self._refractory_remaining == 0:
                    self._adapt_during_refractory = False
                self._append_to_pre_roll(channels)
                return []

            if self._capture is not None:
                self._append_to_capture(channels)
                event = self._complete_capture_if_ready()
                return [event] if event else []

            rms_threshold = max(self.noise_floor_rms * 1.18, 0.0008)
            peak_threshold = max(self.noise_floor_rms * 4.0, 0.007)
            crest = peak / max(rms, 1e-6)
            strong_sample_threshold = max(peak_threshold, peak * 0.55)
            strong_fraction = (
                float(np.count_nonzero(np.abs(onset_signal) >= strong_sample_threshold)) / max(frame_count, 1)
            )
            is_impulse = (
                rms > rms_threshold and peak > peak_threshold and crest > 2.0 and strong_fraction < 0.20
            )

            if is_impulse:
                over = np.nonzero(np.abs(onset_signal) >= peak_threshold)[0]
                crossing = int(over[0]) if over.size else 0
                self._capture = [list(ch) for ch in self._pre_roll]
                self._capture_onset_offset = len(self._pre_roll[0]) + crossing if self._pre_roll[0] else crossing
                self._capture_stream_index = self.total_samples + crossing
                self._capture_noise_floor = self.noise_floor_rms
                self._append_to_capture(channels)
                event = self._complete_capture_if_ready()
                return [event] if event else []
            else:
                self._adapt_noise_floor(rms, is_warm_up=False)
                self._append_to_pre_roll(channels)
                return []
        finally:
            self.total_samples += frame_count

    def _normalized_channels(self, incoming: List[np.ndarray], frame_count: int) -> List[np.ndarray]:
        result = []
        for ch in range(self.channel_count):
            source = incoming[min(ch, len(incoming) - 1)]
            result.append(np.asarray(source[:frame_count], dtype=np.float32))
        return result

    def _mix_down(self, channels: List[np.ndarray]) -> np.ndarray:
        if len(channels) == 1:
            return channels[0]
        acc = np.zeros(len(channels[0]), dtype=np.float32)
        for c in channels:
            acc += c / len(channels)
        return acc

    def _rms(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))

    def _adapt_noise_floor(self, measured_rms: float, is_warm_up: bool):
        if not math.isfinite(measured_rms):
            return
        measured = max(measured_rms, 1e-5)
        if is_warm_up:
            alpha = 0.14
            self.noise_floor_rms = (1 - alpha) * self.noise_floor_rms + alpha * measured
            return
        capped = min(measured, max(self.noise_floor_rms * 3.5, 0.020))
        alpha = 0.06 if capped > self.noise_floor_rms else 0.025
        self.noise_floor_rms = (1 - alpha) * self.noise_floor_rms + alpha * capped

    def _low_pass_for_onset(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        cutoff = min(6000.0, self.sample_rate * 0.20)
        alpha = 1 - math.exp(-2 * math.pi * cutoff / self.sample_rate)
        out = np.empty_like(values, dtype=np.float32)
        states = self._onset_filter_state
        for i in range(values.size):
            filtered = float(values[i])
            for s in range(4):
                states[s] += alpha * (filtered - states[s])
                filtered = states[s]
            out[i] = filtered
        return out

    def _append_to_pre_roll(self, channels: List[np.ndarray]):
        for i in range(self.channel_count):
            self._pre_roll[i].extend(channels[i].tolist())
            if len(self._pre_roll[i]) > self.pre_roll_samples:
                excess = len(self._pre_roll[i]) - self.pre_roll_samples
                del self._pre_roll[i][:excess]

    def _append_to_capture(self, channels: List[np.ndarray]):
        if self._capture is None:
            return
        for i in range(self.channel_count):
            self._capture[i].extend(channels[i].tolist())

    def _complete_capture_if_ready(self) -> Optional[DetectedTap]:
        if self._capture is None or len(self._capture[0]) < self.analysis_window_samples:
            return None
        trimmed = [c[: self.analysis_window_samples] for c in self._capture]
        event = DetectedTap(
            channels=trimmed,
            onset_offset=min(self._capture_onset_offset, self.analysis_window_samples - 1),
            stream_sample_index=self._capture_stream_index,
            noise_floor_rms=self._capture_noise_floor,
        )
        accepted = gate.accepts(event, self.sample_rate)
        self._capture = None
        self._pre_roll = [[] for _ in range(self.channel_count)]
        self._refractory_remaining = int(self.sample_rate * 0.14)
        self._adapt_during_refractory = not accepted
        return event if accepted else None
