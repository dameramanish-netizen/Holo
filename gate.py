"""Rejects sustained (speech-like) events after the detector captures a
candidate window. Ported from ImpactEventGate.swift."""
from __future__ import annotations
import math
import numpy as np

from .signal_models import DetectedTap


def _low_pass(values: np.ndarray, sample_rate: float) -> np.ndarray:
    if values.size == 0:
        return values
    cutoff = min(6000.0, sample_rate * 0.20)
    alpha = 1 - math.exp(-2 * math.pi * cutoff / sample_rate)
    out = np.empty_like(values, dtype=np.float64)
    states = [0.0, 0.0, 0.0, 0.0]
    for i, v in enumerate(values):
        filtered = v
        for s in range(4):
            states[s] += alpha * (filtered - states[s])
            filtered = states[s]
        out[i] = filtered
    return out


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))


def _energy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sum(values.astype(np.float64) ** 2))


def accepts(event: DetectedTap, sample_rate: float) -> bool:
    metrics = _metrics(event, sample_rate)
    if metrics is None:
        return False
    onset_contrast, effective_duration, early_energy_fraction, late_to_impact_rms = metrics

    if onset_contrast < 1.8:
        return False

    is_clearly_sustained = (
        effective_duration >= 0.040
        and late_to_impact_rms >= 0.36
        and early_energy_fraction < 0.60
    )
    return not is_clearly_sustained


def _metrics(event: DetectedTap, sample_rate: float):
    if sample_rate <= 0 or not event.channels or len(event.channels[0]) == 0:
        return None

    usable = [np.asarray(c, dtype=np.float64) for c in event.channels if len(c) > 0]
    if not usable:
        return None
    frame_count = min(len(c) for c in usable)
    if frame_count <= 0:
        return None
    mono = np.zeros(frame_count, dtype=np.float64)
    for c in usable:
        mono += c[:frame_count] / len(usable)

    filtered = _low_pass(mono, sample_rate)
    onset = min(max(event.onset_offset, 0), len(filtered) - 1)
    frame_samples = max(int(sample_rate * 0.005), 32)

    frame_rms = []
    frame_lengths = []
    offset = onset
    while offset < len(filtered):
        end = min(offset + frame_samples, len(filtered))
        frame_rms.append(_rms(filtered[offset:end]))
        frame_lengths.append(end - offset)
        offset = end
    if not frame_rms:
        return None

    impact_search_frames = min(len(frame_rms), max(int(math.ceil(0.025 * sample_rate / frame_samples)), 1))
    impact_rms = max(frame_rms[:impact_search_frames]) if frame_rms[:impact_search_frames] else 0
    if impact_rms <= 0:
        return None

    pre_start = max(0, onset - int(sample_rate * 0.012))
    pre_rms = _rms(filtered[pre_start:onset])
    reference_floor = max(pre_rms, event.noise_floor_rms, 1e-6)
    impact_search_end = min(onset + impact_search_frames * frame_samples, len(filtered))
    impact_peak = float(np.max(np.abs(filtered[onset:impact_search_end]))) if impact_search_end > onset else 0.0
    onset_contrast = max(impact_rms / reference_floor, 0.45 * impact_peak / reference_floor)

    effective_threshold = max(impact_rms * 0.40, event.noise_floor_rms * 2.2)
    effective_samples = sum(
        length for rms_val, length in zip(frame_rms, frame_lengths) if rms_val >= effective_threshold
    )
    effective_duration = effective_samples / sample_rate

    early_end = min(onset + int(sample_rate * 0.025), len(filtered))
    total_energy = _energy(filtered[onset:])
    early_energy_fraction = _energy(filtered[onset:early_end]) / max(total_energy, 1e-15)

    late_start = min(onset + int(sample_rate * 0.040), len(filtered))
    late_rms = _rms(filtered[late_start:])

    return (
        onset_contrast,
        effective_duration,
        early_energy_fraction,
        late_rms / max(impact_rms, 1e-12),
    )
