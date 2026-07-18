"""Profiles: zone action configuration + persisted classifier. Ported from Profile.swift."""
from __future__ import annotations
import json
import os
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from .classifier import TrainedTapClassifier
from .zone import DeskZone, ZoneActionKind

CURRENT_VERSION = 1


@dataclass
class ZoneActionConfiguration:
    kind: ZoneActionKind = ZoneActionKind.NONE
    text: str = ""
    path: str = ""   # file/app path for OPEN_APPLICATION / OPEN_ITEM / RUN_SCRIPT

    def to_dict(self):
        return {"kind": self.kind.value, "text": self.text, "path": self.path}

    @staticmethod
    def from_dict(d):
        return ZoneActionConfiguration(kind=ZoneActionKind(d.get("kind", "none")), text=d.get("text", ""),
                                        path=d.get("path", ""))


@dataclass
class ZoneConfiguration:
    zone: DeskZone
    action: ZoneActionConfiguration = field(default_factory=ZoneActionConfiguration)

    def to_dict(self):
        return {"zone": self.zone.value, "action": self.action.to_dict()}

    @staticmethod
    def from_dict(d):
        return ZoneConfiguration(zone=DeskZone(d["zone"]), action=ZoneActionConfiguration.from_dict(d["action"]))


@dataclass
class CalibrationSummary:
    sample_count: int
    samples_per_zone: List[int]
    leave_one_out_accuracy: Optional[float]
    per_zone_accuracy: Optional[dict] = None   # {DeskZone.value: float}
    captured_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return {
            "sampleCount": self.sample_count,
            "samplesPerZone": self.samples_per_zone,
            "leaveOneOutAccuracy": self.leave_one_out_accuracy,
            "perZoneAccuracy": self.per_zone_accuracy,
            "capturedAt": self.captured_at,
        }

    @staticmethod
    def from_dict(d):
        return CalibrationSummary(sample_count=d["sampleCount"], samples_per_zone=d["samplesPerZone"],
                                   leave_one_out_accuracy=d.get("leaveOneOutAccuracy"),
                                   per_zone_accuracy=d.get("perZoneAccuracy"),
                                   captured_at=d.get("capturedAt", ""))


@dataclass
class HoloProfile:
    name: str
    surface_description: str
    laptop_position_note: str
    classifier: TrainedTapClassifier
    calibration: CalibrationSummary
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    zones: List[ZoneConfiguration] = field(default_factory=lambda: [ZoneConfiguration(z) for z in DeskZone.all()])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = CURRENT_VERSION

    def action_for(self, zone: DeskZone) -> ZoneActionConfiguration:
        for z in self.zones:
            if z.zone == zone:
                return z.action
        return ZoneActionConfiguration()

    def to_dict(self):
        return {
            "version": self.version,
            "id": self.id,
            "name": self.name,
            "surfaceDescription": self.surface_description,
            "laptopPositionNote": self.laptop_position_note,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "classifier": self.classifier.to_dict(),
            "calibration": self.calibration.to_dict(),
            "zones": [z.to_dict() for z in self.zones],
        }

    @staticmethod
    def from_dict(d):
        return HoloProfile(
            id=d["id"], name=d["name"], surface_description=d.get("surfaceDescription", ""),
            laptop_position_note=d.get("laptopPositionNote", ""),
            classifier=TrainedTapClassifier.from_dict(d["classifier"]),
            calibration=CalibrationSummary.from_dict(d["calibration"]),
            zones=[ZoneConfiguration.from_dict(z) for z in d["zones"]],
            created_at=d.get("createdAt", ""), updated_at=d.get("updatedAt", ""),
            version=d.get("version", CURRENT_VERSION),
        )


def data_directory() -> str:
    """%APPDATA%/Holo on Windows, ~/.holo elsewhere."""
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        path = os.path.join(base, "Holo")
    else:
        path = os.path.join(os.path.expanduser("~"), ".holo")
    os.makedirs(os.path.join(path, "Profiles"), exist_ok=True)
    os.makedirs(os.path.join(path, "Evaluations"), exist_ok=True)
    return path


def profiles_directory() -> str:
    return os.path.join(data_directory(), "Profiles")


def list_profiles() -> List[str]:
    d = profiles_directory()
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json"))


def save_profile(profile: HoloProfile):
    profile.updated_at = datetime.now(timezone.utc).isoformat()
    path = os.path.join(profiles_directory(), f"{profile.id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)


def load_profile(profile_id: str) -> HoloProfile:
    path = os.path.join(profiles_directory(), f"{profile_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return HoloProfile.from_dict(json.load(f))


def delete_profile(profile_id: str):
    path = os.path.join(profiles_directory(), f"{profile_id}.json")
    if os.path.exists(path):
        os.remove(path)
    if get_last_profile_id() == profile_id:
        set_last_profile_id(None)


def _settings_path() -> str:
    return os.path.join(data_directory(), "settings.json")


def _read_settings() -> dict:
    path = _settings_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_settings(settings: dict):
    with open(_settings_path(), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def get_last_profile_id() -> Optional[str]:
    return _read_settings().get("lastProfileId")


def set_last_profile_id(profile_id: Optional[str]):
    settings = _read_settings()
    settings["lastProfileId"] = profile_id
    _write_settings(settings)


def load_most_recent_profile() -> Optional[HoloProfile]:
    """Loads the profile recorded as 'last used'; if that's missing or gone,
    falls back to whichever saved profile was updated most recently. Returns
    None if there are no profiles at all."""
    last_id = get_last_profile_id()
    if last_id:
        try:
            return load_profile(last_id)
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    candidates = []
    for pid in list_profiles():
        try:
            candidates.append(load_profile(pid))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.updated_at, reverse=True)
    return candidates[0]
