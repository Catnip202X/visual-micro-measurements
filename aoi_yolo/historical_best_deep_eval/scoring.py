from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .cache import PredictionRecord, read_cache
from .dataset import ImageRecord, TARGET_MAP


@dataclass(frozen=True)
class GroundTruth:
    ground_truth_id: str
    label: str
    polygon: tuple[tuple[float, float], ...]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _ground_truth(record: ImageRecord) -> list[GroundTruth]:
    payload = json.loads(record.json_path.read_text(encoding="utf-8"))
    rows: list[GroundTruth] = []
    for index, shape in enumerate(payload.get("shapes", [])):
        source_label = str(shape.get("label", "")).strip().lower()
        label = TARGET_MAP.get(source_label)
        points = shape.get("points") or []
        if label is None or len(points) < 3:
            continue
        polygon = tuple((float(point[0]), float(point[1])) for point in points)
        rows.append(GroundTruth(f"gt:{index}", label, polygon))
    return rows


def polygon_iou(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
    width: int,
    height: int,
) -> float:
    all_points = np.asarray(left + right, dtype=np.float64)
    x0 = max(0, int(math.floor(float(all_points[:, 0].min()))))
    y0 = max(0, int(math.floor(float(all_points[:, 1].min()))))
    x1 = min(width - 1, int(math.ceil(float(all_points[:, 0].max()))))
    y1 = min(height - 1, int(math.ceil(float(all_points[:, 1].max()))))
    if x1 < x0 or y1 < y0:
        return 0.0
    shape = (y1 - y0 + 1, x1 - x0 + 1)
    left_mask = np.zeros(shape, dtype=np.uint8)
    right_mask = np.zeros(shape, dtype=np.uint8)
    offset = np.asarray([x0, y0], dtype=np.float64)
    left_points = np.rint(np.asarray(left, dtype=np.float64) - offset).astype(np.int32)
    right_points = np.rint(np.asarray(right, dtype=np.float64) - offset).astype(np.int32)
    cv2.fillPoly(left_mask, [left_points], 1)
    cv2.fillPoly(right_mask, [right_points], 1)
    intersection = int(np.count_nonzero(left_mask & right_mask))
    union = int(np.count_nonzero(left_mask | right_mask))
    return intersection / union if union else 0.0


def _match_class(
    truths: list[GroundTruth],
    predictions: list[PredictionRecord],
    width: int,
    height: int,
    hit_iou: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for truth_index, truth in enumerate(truths):
        for prediction_index, prediction in enumerate(predictions):
            overlap = polygon_iou(truth.polygon, prediction.polygon, width, height)
            if overlap >= hit_iou:
                candidates.append((overlap, truth_index, prediction_index))
    used_truths: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, truth_index, prediction_index in sorted(candidates, reverse=True):
        if truth_index in used_truths or prediction_index in used_predictions:
            continue
        used_truths.add(truth_index)
        used_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, overlap))
    return matches, used_truths, used_predictions


def _metric_row(label: str, counts: Counter, positive_images: int, matched_ious: list[float]) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    recall_denominator = tp + fn
    precision_denominator = tp + fp
    recall = tp / recall_denominator if recall_denominator else None
    precision = tp / precision_denominator if precision_denominator else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    recall_ci = wilson_interval(tp, recall_denominator)
    precision_ci = wilson_interval(tp, precision_denominator)
    return {
        "label": label,
        "annotated_objects": recall_denominator,
        "positive_images": positive_images,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "recall_ci95_low": recall_ci[0],
        "recall_ci95_high": recall_ci[1],
        "precision": precision,
        "precision_ci95_low": precision_ci[0],
        "precision_ci95_high": precision_ci[1],
        "f1": f1,
        "mean_matched_mask_iou": float(np.mean(matched_ious)) if matched_ious else None,
    }


def _write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def score_cached_run(
    cache_path: Path,
    manifest: list[ImageRecord],
    expected_identity: dict[str, object],
    output_dir: Path,
    *,
    hit_iou: float,
) -> dict:
    identity, cached_rows = read_cache(cache_path)
    if identity != expected_identity:
        raise ValueError("prediction cache identity does not match frozen experiment identity")
    manifest_stems = [record.stem for record in manifest]
    cache_stems = [row.stem for row in cached_rows]
    if cache_stems != manifest_stems or len(set(cache_stems)) != len(cache_stems):
        raise ValueError("prediction cache rows do not exactly match manifest order and membership")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts_by_class = {label: Counter() for label in ("pinhole", "scratch")}
    ious_by_class = {label: [] for label in ("pinhole", "scratch")}
    positive_images = Counter()
    event_rows: list[dict] = []
    slip_rows: list[dict] = []

    for record, cached in zip(manifest, cached_rows):
        truths = _ground_truth(record)
        kept_predictions = [prediction for prediction in cached.predictions if prediction.kept]
        for label in ("pinhole", "scratch"):
            label_truths = [truth for truth in truths if truth.label == label]
            label_predictions = [prediction for prediction in kept_predictions if prediction.label == label]
            if label_truths:
                positive_images[label] += 1
            matches, used_truths, used_predictions = _match_class(
                label_truths, label_predictions, record.width, record.height, hit_iou
            )
            for truth_index, prediction_index, overlap in matches:
                truth = label_truths[truth_index]
                prediction = label_predictions[prediction_index]
                counts_by_class[label]["tp"] += 1
                ious_by_class[label].append(overlap)
                event_rows.append({
                    "stem": record.stem, "label": label, "outcome": "tp",
                    "ground_truth_id": truth.ground_truth_id,
                    "prediction_id": prediction.prediction_id,
                    "confidence": prediction.confidence, "mask_iou": overlap,
                })
            for prediction_index, prediction in enumerate(label_predictions):
                if prediction_index in used_predictions:
                    continue
                counts_by_class[label]["fp"] += 1
                event_rows.append({
                    "stem": record.stem, "label": label, "outcome": "fp",
                    "ground_truth_id": "", "prediction_id": prediction.prediction_id,
                    "confidence": prediction.confidence, "mask_iou": 0.0,
                })
            for truth_index, truth in enumerate(label_truths):
                if truth_index in used_truths:
                    continue
                counts_by_class[label]["fn"] += 1
                row = {
                    "stem": record.stem, "label": label, "outcome": "fn",
                    "ground_truth_id": truth.ground_truth_id,
                    "prediction_id": "", "confidence": "", "mask_iou": 0.0,
                }
                event_rows.append(row)
                slip_rows.append(row)

    by_class = {
        label: _metric_row(label, counts_by_class[label], positive_images[label], ious_by_class[label])
        for label in ("pinhole", "scratch")
    }
    overall_counts = Counter()
    overall_ious: list[float] = []
    for label in ("pinhole", "scratch"):
        overall_counts.update(counts_by_class[label])
        overall_ious.extend(ious_by_class[label])
    overall_positive_images = sum(
        1 for record in manifest if any(record.target_object_counts.get(label, 0) for label in ("pinhole", "scratch"))
    )
    overall = _metric_row("overall", overall_counts, overall_positive_images, overall_ious)
    result = {
        "identity": identity,
        "hit_iou": hit_iou,
        "images": len(manifest),
        "overall": overall,
        "by_class": by_class,
    }
    event_fields = [
        "stem", "label", "outcome", "ground_truth_id", "prediction_id", "confidence", "mask_iou"
    ]
    _write_csv(output_dir / "events.csv", event_rows, event_fields)
    _write_csv(output_dir / "slips.csv", slip_rows, event_fields)
    metric_fields = list(overall.keys())
    _write_csv(output_dir / "metrics.csv", [overall, *by_class.values()], metric_fields)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
