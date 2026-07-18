"""Ties microphone capture, feature extraction, calibration, and action
dispatch together for the GUI."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import actions
from .capture import MicCapture
from .classifier import TrainedTapClassifier, ClassifierTrainingError
from .features import TapFeatureExtractor
from .profile import (
    HoloProfile, CalibrationSummary, ZoneConfiguration, save_profile,
    set_last_profile_id, load_most_recent_profile,
)
from .signal_models import DetectedTap, LabeledTap, TapFeatureVector, SignalQuality
from .zone import DeskZone

SAMPLES_PER_ZONE = 10
SAMPLE_RATE = 44100


class CalibrationSession:
    """Collects SAMPLES_PER_ZONE accepted taps per zone, in order."""

    def __init__(self):
        self.zone_order: List[DeskZone] = DeskZone.all()
        self.samples: dict = {z: [] for z in self.zone_order}
        self.armed_index = 0

    @property
    def current_zone(self) -> Optional[DeskZone]:
        if self.armed_index >= len(self.zone_order):
            return None
        return self.zone_order[self.armed_index]

    @property
    def is_complete(self) -> bool:
        return self.armed_index >= len(self.zone_order)

    def add_sample(self, feature: TapFeatureVector) -> bool:
        zone = self.current_zone
        if zone is None:
            return False
        self.samples[zone].append(LabeledTap(feature=feature, zone=zone.value))
        if len(self.samples[zone]) >= SAMPLES_PER_ZONE:
            self.armed_index = self._first_incomplete_index(after=self.zone_order.index(zone))
        return True

    def undo_last(self):
        zone = self.current_zone
        if zone is None and self.armed_index > 0:
            zone = self.zone_order[self.armed_index - 1]
        if zone and self.samples[zone]:
            self.samples[zone].pop()
            self.armed_index = self._first_incomplete_index()

    def redo_zone(self, zone: DeskZone):
        """Clears one zone's samples and re-arms it, without disturbing
        samples already collected for other zones."""
        self.samples[zone] = []
        self.armed_index = self._first_incomplete_index()

    def _first_incomplete_index(self, after: int = -1) -> int:
        for i in range(after + 1, len(self.zone_order)):
            if len(self.samples[self.zone_order[i]]) < SAMPLES_PER_ZONE:
                return i
        return len(self.zone_order)

    def all_labeled_taps(self) -> List[LabeledTap]:
        result = []
        for z in self.zone_order:
            result.extend(self.samples[z])
        return result

    def counts(self) -> dict:
        return {z: len(self.samples[z]) for z in self.zone_order}


class AppController:
    def __init__(self):
        self.mic = MicCapture(sample_rate=SAMPLE_RATE, preferred_channels=2)
        self.extractor = TapFeatureExtractor(sample_rate=SAMPLE_RATE)
        self.profile: Optional[HoloProfile] = None
        self.calibration: Optional[CalibrationSession] = None
        self.desk_active = False
        self.test_mode = False
        self.test_expected_zone: Optional[DeskZone] = None
        self.test_results: List[tuple] = []   # (expected_zone_or_None, decision)
        self.on_tap_feedback: Optional[Callable] = None  # (zone_or_None, reason_or_None, quality_summary)
        self.on_test_result: Optional[Callable] = None   # (decision, expected_zone_or_None)
        self.on_level: Optional[Callable[[float], None]] = None
        self.on_action_status: Optional[Callable[[str], None]] = None

    # ---- microphone ---------------------------------------------------
    def start_listening(self, device: Optional[int] = None):
        self.mic.device = device
        self.mic.start(on_tap=self._handle_tap, on_level=self._handle_level)

    def stop_listening(self):
        self.mic.stop()

    def _handle_level(self, rms: float):
        if self.on_level:
            self.on_level(rms)

    def _handle_tap(self, tap: DetectedTap):
        feature = self.extractor.extract(tap)

        if self.calibration is not None:
            self.calibration.add_sample(feature)
            if self.on_tap_feedback:
                zone = self.calibration.zone_order[
                    min(self.calibration.armed_index, len(self.calibration.zone_order) - 1)
                ]
                self.on_tap_feedback("calibration", None, feature.quality.summary)
            return

        if self.profile is None:
            return

        decision = self.profile.classifier.predict(feature)
        if self.on_tap_feedback:
            zone_name = DeskZone(decision.zone).display_name if decision.zone is not None else None
            reason = decision.rejection_reason.display_name if decision.rejection_reason else None
            self.on_tap_feedback(zone_name, reason, feature.quality.summary)

        if self.test_mode:
            # Safety: test mode never dispatches actions, regardless of the
            # Desk-active setting, so you can safely tap-test a fresh profile.
            self.test_results.append((self.test_expected_zone, decision))
            if self.on_test_result:
                self.on_test_result(decision, self.test_expected_zone)
            return

        if self.desk_active and decision.was_accepted:
            zone = DeskZone(decision.zone)
            action = self.profile.action_for(zone)
            actions.run_action(action, on_status=self.on_action_status)

    # ---- test mode (classify + score, never dispatch actions) -----------
    def begin_test_mode(self):
        self.test_mode = True
        self.test_results = []

    def end_test_mode(self):
        self.test_mode = False

    def set_test_expected_zone(self, zone: Optional[DeskZone]):
        self.test_expected_zone = zone

    def test_tally(self):
        """Returns (correct, total, per_zone: {DeskZone: [correct, total]})."""
        correct = 0
        per_zone = {z: [0, 0] for z in DeskZone.all()}
        for expected, decision in self.test_results:
            if expected is None:
                continue
            per_zone[expected][1] += 1
            if decision.zone == expected.value:
                correct += 1
                per_zone[expected][0] += 1
        total = sum(v[1] for v in per_zone.values())
        return correct, total, per_zone

    # ---- calibration ----------------------------------------------------
    def begin_calibration(self):
        self.calibration = CalibrationSession()

    def cancel_calibration(self):
        self.calibration = None

    def calibration_quality_breakdown(self):
        """Leave-one-out accuracy per zone for the *current, uncommitted*
        calibration session -- lets you spot a weak zone and redo just that
        one before saving, instead of only seeing one overall number."""
        if self.calibration is None or not self.calibration.is_complete:
            return None
        labeled = self.calibration.all_labeled_taps()
        overall, per_zone = self._leave_one_out_breakdown(labeled)
        return overall, per_zone

    def save_calibration(self, name: str, surface: str, position_note: str) -> HoloProfile:
        if self.calibration is None or not self.calibration.is_complete:
            raise RuntimeError("Calibration is not complete")
        labeled = self.calibration.all_labeled_taps()
        classifier = TrainedTapClassifier.train(labeled)
        loo, per_zone = self._leave_one_out_breakdown(labeled)
        summary = CalibrationSummary(
            sample_count=len(labeled),
            samples_per_zone=[len(self.calibration.samples[z]) for z in self.calibration.zone_order],
            leave_one_out_accuracy=loo,
            per_zone_accuracy={z.value: acc for z, (acc, _, _) in per_zone.items()} if per_zone else None,
        )
        profile = HoloProfile(
            name=name, surface_description=surface, laptop_position_note=position_note,
            classifier=classifier, calibration=summary,
            zones=[ZoneConfiguration(z) for z in DeskZone.all()],
        )
        save_profile(profile)
        set_last_profile_id(profile.id)
        self.profile = profile
        self.calibration = None
        return profile

    def _leave_one_out_breakdown(self, labeled: List[LabeledTap]):
        """Returns (overall_accuracy, {DeskZone: (accuracy, correct, total)})."""
        if len(labeled) < 8:
            return None, None
        correct_total = 0
        scored_total = 0
        per_zone_correct = {z: 0 for z in DeskZone.all()}
        per_zone_total = {z: 0 for z in DeskZone.all()}
        for i in range(len(labeled)):
            train_set = labeled[:i] + labeled[i + 1:]
            try:
                model = TrainedTapClassifier.train(train_set)
            except ClassifierTrainingError:
                continue
            decision = model.predict(labeled[i].feature)
            expected_zone = DeskZone(labeled[i].zone)
            scored_total += 1
            per_zone_total[expected_zone] += 1
            if decision.zone == labeled[i].zone:
                correct_total += 1
                per_zone_correct[expected_zone] += 1
        overall = correct_total / scored_total if scored_total else None
        per_zone = {
            z: (
                (per_zone_correct[z] / per_zone_total[z]) if per_zone_total[z] else None,
                per_zone_correct[z],
                per_zone_total[z],
            )
            for z in DeskZone.all()
        }
        return overall, per_zone

    # ---- profile / actions ----------------------------------------------
    def auto_load_last_profile(self) -> Optional[HoloProfile]:
        profile = load_most_recent_profile()
        if profile is not None:
            self.profile = profile
        return profile

    def set_desk_active(self, active: bool):
        self.desk_active = active

    def save_action(self, zone: DeskZone, action):
        if self.profile is None:
            return
        for z in self.profile.zones:
            if z.zone == zone:
                z.action = action
        save_profile(self.profile)

    def test_action(self, zone: DeskZone):
        if self.profile is None:
            return False, "No profile loaded"
        action = self.profile.action_for(zone)
        return actions.run_action(action, on_status=self.on_action_status)
