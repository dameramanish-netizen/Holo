"""Passive tap-acoustics feature extraction. Ported from TapFeatureExtractor.swift.
(Active ultrasonic-probe sensing is not ported -- see README.)"""
from __future__ import annotations
import math
from typing import List, Tuple

import numpy as np

from .signal_models import DetectedTap, SignalQuality, TapFeatureVector

PASSIVE_FEATURE_NAMES = (
    ["log_rms", "crest_factor", "zero_crossing_rate", "attack_position",
     "temporal_centroid", "early_late_ratio",
     "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85", "spectral_flatness"]
    + [f"mfcc_{i}" for i in range(1, 9)]
    + [f"band_{i}" for i in range(10)]
    + ["channel_energy_delta", "interchannel_delay"]
)


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return max(p, 1)


def _power_spectrum(mono: np.ndarray, size: int) -> np.ndarray:
    """Zero-padded/truncated real FFT power spectrum of length size//2+1."""
    if mono.size == 0:
        return np.zeros(size // 2 + 1)
    padded = np.zeros(size, dtype=np.float64)
    n = min(mono.size, size)
    padded[:n] = mono[:n]
    spectrum = np.fft.rfft(padded)
    return (spectrum.real ** 2 + spectrum.imag ** 2)


def _frequency_bin(frequency: float, sample_rate: float, spectrum_count: int) -> int:
    fft_size = max((spectrum_count - 1) * 2, 1)
    idx = int(frequency / sample_rate * fft_size)
    return min(max(idx, 0), spectrum_count - 1)


def _hz_to_mel(hz: float) -> float:
    return 2595 * math.log10(1 + hz / 700)


def _mel_to_hz(mel: float) -> float:
    return 700 * (10 ** (mel / 2595) - 1)


class TapFeatureExtractor:
    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def extract(self, event: DetectedTap) -> TapFeatureVector:
        channels = [np.asarray(c, dtype=np.float64) for c in event.channels if len(c) > 0]
        mono = self._mix_down(channels)
        spectrum = _power_spectrum(mono, min(_next_power_of_two(mono.size), 4096))

        names, values = self._passive_features(mono, channels, event.onset_offset, spectrum)

        rms = self._rms(mono)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        clipping = float(np.count_nonzero(np.abs(mono) >= 0.995)) / max(mono.size, 1)
        snr = 20 * math.log10(max(rms, 1e-12) / max(event.noise_floor_rms, 1e-12))
        quality = SignalQuality(
            signal_to_noise_db=snr,
            peak_amplitude=peak,
            rms_amplitude=rms,
            clipping_fraction=clipping,
            noise_floor_rms=event.noise_floor_rms,
            duration_ms=mono.size / self.sample_rate * 1000,
        )
        clean_values = [v if math.isfinite(v) else 0.0 for v in values]
        return TapFeatureVector(names=names, values=clean_values, quality=quality, strategy="passive")

    def _passive_features(self, signal: np.ndarray, channels: List[np.ndarray], onset: int, spectrum: np.ndarray):
        if signal.size == 0:
            return PASSIVE_FEATURE_NAMES, [0.0] * len(PASSIVE_FEATURE_NAMES)

        rms = self._rms(signal)
        peak = float(np.max(np.abs(signal)))
        crest = peak / max(rms, 1e-12)
        signs = signal >= 0
        zero_crossings = int(np.count_nonzero(signs[:-1] != signs[1:])) if signal.size > 1 else 0
        zcr = zero_crossings / max(signal.size - 1, 1)
        peak_index = int(np.argmax(np.abs(signal)))
        attack_position = max(peak_index - onset, 0) / max(signal.size - onset, 1)

        energies = signal ** 2
        total_energy = float(np.sum(energies)) + 1e-15
        idx = np.arange(signal.size)
        temporal_centroid = float(np.sum(idx * energies)) / total_energy / max(signal.size - 1, 1)
        split = min(max(onset + int(self.sample_rate * 0.025), 1), signal.size - 1)
        early = float(np.sum(energies[:split]))
        late = float(np.sum(energies[split:]))
        early_late_ratio = math.log10((early + 1e-12) / (late + 1e-12))

        spectral = self._spectral_features(spectrum)
        mel = self._mel_cepstral_features(spectrum, filter_count=18, coefficient_count=8)
        bands = self._broad_band_features(spectrum, count=10)
        spatial = self._spatial_features(channels)

        temporal = [math.log10(rms + 1e-12), crest, zcr, attack_position, temporal_centroid, early_late_ratio]
        return PASSIVE_FEATURE_NAMES, temporal + spectral + mel + bands + spatial

    def _spectral_features(self, spectrum: np.ndarray) -> List[float]:
        fft_size = max((spectrum.size - 1) * 2, 1)
        freqs = np.arange(spectrum.size) * self.sample_rate / fft_size
        mask = (freqs >= 60) & (freqs <= min(18000.0, self.sample_rate * 0.48))
        usable_freqs = freqs[mask]
        usable_vals = spectrum[mask]
        total = float(np.sum(usable_vals)) + 1e-15
        if usable_vals.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        centroid = float(np.sum(usable_freqs * usable_vals)) / total
        bandwidth = math.sqrt(max(float(np.sum(((usable_freqs - centroid) ** 2) * usable_vals)) / total, 0))

        cumulative = np.cumsum(usable_vals)
        rolloff = 0.0
        threshold = total * 0.85
        over = np.nonzero(cumulative >= threshold)[0]
        if over.size:
            rolloff = float(usable_freqs[over[0]])

        arithmetic = total / max(usable_vals.size, 1)
        geometric = math.exp(float(np.sum(np.log(usable_vals + 1e-15))) / max(usable_vals.size, 1))
        flatness = geometric / max(arithmetic, 1e-15)
        return [centroid / self.sample_rate, bandwidth / self.sample_rate, rolloff / self.sample_rate, flatness]

    def _mel_cepstral_features(self, spectrum: np.ndarray, filter_count: int, coefficient_count: int) -> List[float]:
        min_mel = _hz_to_mel(80)
        max_mel = _hz_to_mel(min(16000.0, self.sample_rate * 0.46))
        points = [
            _mel_to_hz(min_mel + (max_mel - min_mel) * i / (filter_count + 1))
            for i in range(filter_count + 2)
        ]
        bins = [_frequency_bin(p, self.sample_rate, spectrum.size) for p in points]
        log_energies = np.zeros(filter_count)
        for f in range(filter_count):
            left = bins[f]
            center = max(bins[f + 1], left + 1)
            right = max(bins[f + 2], center + 1)
            energy = 0.0
            if left < spectrum.size:
                for i in range(left, min(center, spectrum.size)):
                    energy += spectrum[i] * (i - left) / max(center - left, 1)
                if center < spectrum.size:
                    for i in range(center, min(right, spectrum.size)):
                        energy += spectrum[i] * (right - i) / max(right - center, 1)
            log_energies[f] = math.log(energy + 1e-15)

        coeffs = []
        for c in range(1, coefficient_count + 1):
            total = 0.0
            for i, le in enumerate(log_energies):
                total += le * math.cos(math.pi * c * (i + 0.5) / filter_count)
            coeffs.append(total / filter_count)
        return coeffs

    def _broad_band_features(self, spectrum: np.ndarray, count: int) -> List[float]:
        minimum, maximum = 80.0, min(self.sample_rate * 0.46, 18000.0)
        total = float(np.sum(spectrum)) + 1e-15
        result = []
        for band in range(count):
            low = minimum * (maximum / minimum) ** (band / count)
            high = minimum * (maximum / minimum) ** ((band + 1) / count)
            start = _frequency_bin(low, self.sample_rate, spectrum.size)
            end = max(start, _frequency_bin(high, self.sample_rate, spectrum.size))
            energy = float(np.sum(spectrum[start: min(end, spectrum.size - 1) + 1])) / total
            result.append(math.log10(energy + 1e-12))
        return result

    def _spatial_features(self, channels: List[np.ndarray]) -> List[float]:
        if len(channels) < 2 or channels[0].size == 0 or channels[1].size == 0:
            return [0.0, 0.0]
        first, second = channels[0], channels[1]
        first_rms = self._rms(first)
        second_rms = self._rms(second)
        energy_delta = math.log10((first_rms + 1e-12) / (second_rms + 1e-12))
        max_lag = min(int(self.sample_rate * 0.00075), 36)
        best_lag, best_corr = 0, -math.inf
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a, b = first[: len(first) - lag], second[lag: lag + len(first) - lag]
            else:
                a, b = first[-lag: -lag + len(first) + lag], second[: len(second) + lag]
            n = min(len(a), len(b))
            if n <= 0:
                continue
            corr = float(np.dot(a[:n], b[:n]))
            if corr > best_corr:
                best_corr, best_lag = corr, lag
        return [energy_delta, best_lag / max(max_lag, 1)]

    def _mix_down(self, channels: List[np.ndarray]) -> np.ndarray:
        if not channels:
            return np.zeros(0)
        if len(channels) == 1:
            return channels[0]
        n = min(c.size for c in channels)
        acc = np.zeros(n, dtype=np.float64)
        for c in channels:
            acc += c[:n] / len(channels)
        return acc

    def _rms(self, values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
