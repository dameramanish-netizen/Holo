"""Ported from SignalModels.swift."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional
import math


MIN_RELIABLE_PEAK_AMPLITUDE = 0.003
MAX_RELIABLE_CLIPPING_FRACTION = 0.20
MIN_CLASSIFICATION_SNR_DB = 6.0


@dataclass
class SignalQuality:
    signal_to_noise_db: float
    peak_amplitude: float
    rms_amplitude: float
    clipping_fraction: float
    noise_floor_rms: float
    duration_ms: float

    @property
    def score(self) -> float:
        snr = min(max((self.signal_to_noise_db - 4) / 30, 0), 1)
        strength = min(max((self.peak_amplitude - MIN_RELIABLE_PEAK_AMPLITUDE) / 0.15, 0), 1)
        clean = 1 - min(self.clipping_fraction * 4, 1)
        return 0.55 * snr + 0.25 * strength + 0.20 * clean

    @property
    def summary(self) -> str:
        if self.clipping_fraction > MAX_RELIABLE_CLIPPING_FRACTION:
            return "Clipped"
        if self.peak_amplitude < MIN_RELIABLE_PEAK_AMPLITUDE:
            return "Weak"
        if self.signal_to_noise_db < MIN_CLASSIFICATION_SNR_DB:
            return "Noisy"
        s = self.score
        if s > 0.72:
            return "Excellent"
        if s > 0.48:
            return "Good"
        return "Fair"

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return SignalQuality(**d)


@dataclass
class TapFeatureVector:
    names: List[str]
    values: List[float]
    quality: SignalQuality
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    strategy: str = "passive"
    version: int = 1

    def to_dict(self):
        return {
            "version": self.version,
            "strategy": self.strategy,
            "names": self.names,
            "values": self.values,
            "quality": self.quality.to_dict(),
            "capturedAt": self.captured_at,
        }

    @staticmethod
    def from_dict(d):
        return TapFeatureVector(
            names=d["names"],
            values=d["values"],
            quality=SignalQuality.from_dict(d["quality"]),
            captured_at=d.get("capturedAt", ""),
            strategy=d.get("strategy", "passive"),
            version=d.get("version", 1),
        )


@dataclass
class LabeledTap:
    feature: TapFeatureVector
    zone: Optional[int] = None          # DeskZone.value, or None for negative examples
    negative_label: Optional[str] = None

    def to_dict(self):
        return {
            "zone": self.zone,
            "negativeLabel": self.negative_label,
            "feature": self.feature.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        return LabeledTap(
            feature=TapFeatureVector.from_dict(d["feature"]),
            zone=d.get("zone"),
            negative_label=d.get("negativeLabel"),
        )


@dataclass
class DetectedTap:
    channels: List[List[float]]
    onset_offset: int
    stream_sample_index: int
    noise_floor_rms: float
