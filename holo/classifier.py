"""Regularized linear zone model + nearest-example novelty/rejection gates.
Ported from TapClassifier.swift. Ridge regression is solved with numpy's
linear solver instead of a hand-rolled Cholesky factorization -- same math,
simpler code."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .signal_models import (
    LabeledTap, TapFeatureVector, SignalQuality,
    MIN_RELIABLE_PEAK_AMPLITUDE, MAX_RELIABLE_CLIPPING_FRACTION, MIN_CLASSIFICATION_SNR_DB,
)
from .zone import DeskZone, RejectionReason

MIN_CONFIDENCE = 0.36
MIN_RELATIVE_SEPARATION = 0.035
MIN_LINEAR_SCORE_MARGIN = 0.075


class ClassifierTrainingError(Exception):
    pass


@dataclass
class ClassificationDecision:
    zone: Optional[int]
    confidence: float
    signal_strength: float
    zone_distances: List[float]
    rejection_reason: Optional[RejectionReason]
    processing_latency_ms: float = 0.0

    @property
    def was_accepted(self) -> bool:
        return self.zone is not None and self.rejection_reason is None


def _median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else 0.0


def _quantile(values: List[float], probability: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.array(values), probability))


def _normalize(values: np.ndarray, center: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (values - center) / np.maximum(scales, 1e-9)


def _weighted_distance(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    delta = a - b
    weighted = float(np.sum(weights * delta * delta))
    return float(np.sqrt(weighted / max(float(np.sum(weights)), 1e-9)))


def _unweighted_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    delta = a[:n] - b[:n]
    return float(np.sqrt(np.sum(delta * delta) / max(n, 1)))


@dataclass
class TrainedTapClassifier:
    strategy: str
    feature_names: List[str]
    center: np.ndarray
    scales: np.ndarray
    feature_weights: np.ndarray
    positive_examples: List[LabeledTap]
    negative_examples: List[LabeledTap] = field(default_factory=list)
    outlier_threshold: float = 0.0
    positive_novelty_threshold: Optional[float] = None
    linear_coefficients: Optional[np.ndarray] = None  # shape (4, dims+1)
    minimum_confidence: float = MIN_CONFIDENCE

    # ---- training -----------------------------------------------------
    @staticmethod
    def train(positive_examples: List[LabeledTap], negative_examples: Optional[List[LabeledTap]] = None,
              minimum_confidence: float = MIN_CONFIDENCE) -> "TrainedTapClassifier":
        negative_examples = negative_examples or []
        positives = [p for p in positive_examples if p.zone is not None]
        grouped: dict = {}
        for p in positives:
            grouped.setdefault(p.zone, []).append(p)
        if len(grouped) < 2:
            raise ClassifierTrainingError("Examples from at least two zones are required.")
        if not all(len(v) >= 2 for v in grouped.values()):
            raise ClassifierTrainingError("At least two examples per represented zone are required.")
        if not positives:
            raise ClassifierTrainingError("At least two examples per represented zone are required.")

        names = positives[0].feature.names
        strategy = positives[0].feature.strategy
        all_examples = positives + negative_examples
        if not names or not all(
            e.feature.names == names and len(e.feature.values) == len(names) and e.feature.strategy == strategy
            for e in all_examples
        ):
            raise ClassifierTrainingError("All calibration examples must use the same feature schema.")

        dims = len(names)
        values_matrix = np.array([p.feature.values for p in positives], dtype=np.float64)  # (n, dims)
        center = np.median(values_matrix, axis=0)
        deviations = np.abs(values_matrix - center)
        robust_scale = np.median(deviations, axis=0) * 1.4826
        mean = np.mean(values_matrix, axis=0)
        std = np.sqrt(np.sum((values_matrix - mean) ** 2, axis=0) / max(values_matrix.shape[0] - 1, 1))
        scales = np.maximum(np.maximum(robust_scale, std * 0.35), 1e-6)

        normalized = _normalize(values_matrix, center, scales)  # (n, dims)
        zones_arr = np.array([p.zone for p in positives])

        weights = np.ones(dims)
        overall = np.mean(normalized, axis=0)
        for d in range(dims):
            between = 0.0
            within = 0.0
            for zone_val in grouped.keys():
                idx = np.nonzero(zones_arr == zone_val)[0]
                zone_values = normalized[idx, d]
                zone_mean = float(np.mean(zone_values))
                between += len(idx) * (zone_mean - overall[d]) ** 2
                within += float(np.sum((zone_values - zone_mean) ** 2))
            ratio = between / max(within, 1e-6)
            weights[d] = min(max((ratio + 0.02) ** 0.5, 0.12), 4.0)
        weight_mean = np.mean(weights) if weights.size else 1.0
        weights = weights / max(weight_mean, 1e-9)

        same_zone_nearest = []
        same_zone_novelty = []
        for i in range(len(positives)):
            candidates = [j for j in range(len(positives)) if j != i and positives[j].zone == positives[i].zone]
            if candidates:
                nearest = min(_weighted_distance(normalized[i], normalized[j], weights) for j in candidates)
                same_zone_nearest.append(nearest)
                nearest_nov = min(_unweighted_distance(normalized[i], normalized[j]) for j in candidates)
                same_zone_novelty.append(nearest_nov)

        percentile = _quantile(same_zone_nearest, 0.95)
        threshold = max(percentile * 2.6, 0.85)
        novelty_percentile = _quantile(same_zone_novelty, 0.95)
        novelty_threshold = max(novelty_percentile * 2.6, 0.95)

        linear_coeffs = _train_linear_zone_model(normalized, zones_arr, len(positives))

        return TrainedTapClassifier(
            strategy=strategy,
            feature_names=names,
            center=center,
            scales=scales,
            feature_weights=weights,
            positive_examples=positives,
            negative_examples=negative_examples,
            outlier_threshold=threshold,
            positive_novelty_threshold=novelty_threshold,
            linear_coefficients=linear_coeffs,
            minimum_confidence=minimum_confidence,
        )

    # ---- prediction -----------------------------------------------------
    def predict(self, feature: TapFeatureVector) -> ClassificationDecision:
        if feature.strategy != self.strategy or feature.names != self.feature_names or len(feature.values) != len(self.center):
            return self._rejected(feature, RejectionReason.SCHEMA_MISMATCH)
        if feature.quality.clipping_fraction > MAX_RELIABLE_CLIPPING_FRACTION:
            return self._rejected(feature, RejectionReason.CLIPPED)
        if feature.quality.peak_amplitude < MIN_RELIABLE_PEAK_AMPLITUDE:
            return self._rejected(feature, RejectionReason.WEAK_SIGNAL)
        if feature.quality.signal_to_noise_db < MIN_CLASSIFICATION_SNR_DB:
            return self._rejected(feature, RejectionReason.LOW_SNR)

        values = np.array(feature.values, dtype=np.float64)
        normalized_input = _normalize(values, self.center, self.scales)

        distances = [float("inf")] * 4
        pos_normalized = {
            i: _normalize(np.array(p.feature.values), self.center, self.scales)
            for i, p in enumerate(self.positive_examples)
        }
        for zone in DeskZone.all():
            idxs = [i for i, p in enumerate(self.positive_examples) if p.zone == zone.value]
            if not idxs:
                continue
            candidates = sorted(_weighted_distance(normalized_input, pos_normalized[i], self.feature_weights) for i in idxs)
            nearest = candidates[:3]
            rank_weights = [0.58, 0.28, 0.14][: len(nearest)]
            denom = sum(rank_weights)
            distances[zone.value] = sum(d * w for d, w in zip(nearest, rank_weights)) / denom

        ranked = sorted([(i, d) for i, d in enumerate(distances) if np.isfinite(d)], key=lambda t: t[1])
        if len(ranked) < 2:
            return self._rejected(feature, RejectionReason.SCHEMA_MISMATCH, distances=distances)
        best_idx, best = ranked[0]
        _, second = ranked[1]

        if best > self.outlier_threshold:
            return self._rejected(feature, RejectionReason.OUT_OF_DISTRIBUTION, distances=distances)

        positive_novelty = [
            _unweighted_distance(normalized_input, pos_normalized[i]) for i in range(len(self.positive_examples))
        ]
        nearest_positive_novelty = min(positive_novelty) if positive_novelty else float("inf")
        if self.positive_novelty_threshold is not None and nearest_positive_novelty > self.positive_novelty_threshold:
            return self._rejected(feature, RejectionReason.OUT_OF_DISTRIBUTION, distances=distances)

        if self.negative_examples:
            nearest_negative = min(
                _unweighted_distance(normalized_input, _normalize(np.array(n.feature.values), self.center, self.scales))
                for n in self.negative_examples
            )
            if nearest_negative <= nearest_positive_novelty * 1.10:
                return self._rejected(feature, RejectionReason.RESEMBLES_NEGATIVE, distances=distances)

        linear = self._linear_decision(normalized_input, nearest_positive_novelty, distances, feature)
        if linear is not None:
            return linear

        separation = (second - best) / max(second, 1e-9)
        separation_score = min(max(separation / 0.38, 0), 1)
        fit_score = min(max(1 - best / self.outlier_threshold, 0), 1)
        confidence = 0.72 * separation_score + 0.28 * fit_score

        if separation < MIN_RELATIVE_SEPARATION or confidence < self.minimum_confidence:
            return self._rejected(feature, RejectionReason.AMBIGUOUS, confidence=confidence, distances=distances)

        return ClassificationDecision(
            zone=best_idx, confidence=confidence, signal_strength=feature.quality.score,
            zone_distances=distances, rejection_reason=None,
        )

    def _linear_decision(self, normalized_input, nearest_positive_novelty, distances, feature):
        if self.linear_coefficients is None:
            return None
        represented = {p.zone for p in self.positive_examples}
        row = np.concatenate(([1.0], normalized_input))
        scores = []
        for zone in DeskZone.all():
            if zone.value not in represented:
                continue
            coeffs = self.linear_coefficients[zone.value]
            if len(coeffs) != len(row):
                return None
            score = float(np.dot(coeffs, row))
            if not np.isfinite(score):
                continue
            scores.append((zone.value, score))
        scores.sort(key=lambda t: -t[1])
        if len(scores) < 2:
            return None

        margin = scores[0][1] - scores[1][1]
        margin_score = min(max(margin / 0.30, 0), 1)
        fit_threshold = self.positive_novelty_threshold or max(self.outlier_threshold, 1e-9)
        fit_score = min(max(1 - nearest_positive_novelty / fit_threshold, 0), 1)
        confidence = 0.76 * margin_score + 0.24 * fit_score

        if margin < MIN_LINEAR_SCORE_MARGIN or confidence < self.minimum_confidence:
            return None

        return ClassificationDecision(
            zone=scores[0][0], confidence=confidence, signal_strength=feature.quality.score,
            zone_distances=distances, rejection_reason=None,
        )

    def _rejected(self, feature, reason, confidence=0.0, distances=None):
        return ClassificationDecision(
            zone=None, confidence=confidence, signal_strength=feature.quality.score,
            zone_distances=distances or [], rejection_reason=reason,
        )

    # ---- persistence -----------------------------------------------------
    def to_dict(self):
        return {
            "strategy": self.strategy,
            "featureNames": self.feature_names,
            "center": self.center.tolist(),
            "scales": self.scales.tolist(),
            "featureWeights": self.feature_weights.tolist(),
            "positiveExamples": [p.to_dict() for p in self.positive_examples],
            "negativeExamples": [n.to_dict() for n in self.negative_examples],
            "outlierThreshold": self.outlier_threshold,
            "positiveNoveltyThreshold": self.positive_novelty_threshold,
            "linearCoefficients": self.linear_coefficients.tolist() if self.linear_coefficients is not None else None,
            "minimumConfidence": self.minimum_confidence,
        }

    @staticmethod
    def from_dict(d):
        return TrainedTapClassifier(
            strategy=d["strategy"],
            feature_names=d["featureNames"],
            center=np.array(d["center"]),
            scales=np.array(d["scales"]),
            feature_weights=np.array(d["featureWeights"]),
            positive_examples=[LabeledTap.from_dict(p) for p in d["positiveExamples"]],
            negative_examples=[LabeledTap.from_dict(n) for n in d.get("negativeExamples", [])],
            outlier_threshold=d["outlierThreshold"],
            positive_novelty_threshold=d.get("positiveNoveltyThreshold"),
            linear_coefficients=np.array(d["linearCoefficients"]) if d.get("linearCoefficients") is not None else None,
            minimum_confidence=d.get("minimumConfidence", MIN_CONFIDENCE),
        )


def _train_linear_zone_model(normalized: np.ndarray, zones_arr: np.ndarray, sample_count: int) -> Optional[np.ndarray]:
    if normalized.size == 0:
        return None
    n, dims = normalized.shape
    param_count = dims + 1
    X = np.concatenate([np.ones((n, 1)), normalized], axis=1)  # (n, dims+1)
    Y = np.zeros((n, 4))
    for i in range(n):
        Y[i, int(zones_arr[i])] = 1.0

    regularization = max(3.0, sample_count * 0.25)
    reg_matrix = np.eye(param_count) * regularization
    reg_matrix[0, 0] = 0.0  # intercept unpenalized

    normal = X.T @ X + reg_matrix
    rhs = X.T @ Y
    try:
        solution = np.linalg.solve(normal, rhs)  # (dims+1, 4)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(solution)):
        return None
    return solution.T  # (4, dims+1)
