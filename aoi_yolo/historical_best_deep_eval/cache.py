from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    label: str
    confidence: float
    area_px: int
    polygon: tuple[tuple[float, float], ...]
    kept: bool


@dataclass(frozen=True)
class TimingRecord:
    index: int
    stem: str
    width: int
    height: int
    model_preprocess_ms: float
    inference_ms: float
    model_postprocess_ms: float
    conversion_cascade_ms: float
    total_production_ms: float
    raw_predictions: int
    kept_predictions: int


@dataclass(frozen=True)
class ImagePrediction:
    stem: str
    predictions: tuple[PredictionRecord, ...]
    timing: TimingRecord


def _prediction_from_dict(value: dict) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=value["prediction_id"],
        label=value["label"],
        confidence=float(value["confidence"]),
        area_px=int(value["area_px"]),
        polygon=tuple(tuple(float(v) for v in point) for point in value["polygon"]),
        kept=bool(value["kept"]),
    )


def _timing_from_dict(value: dict) -> TimingRecord:
    return TimingRecord(
        index=int(value["index"]),
        stem=value["stem"],
        width=int(value["width"]),
        height=int(value["height"]),
        model_preprocess_ms=float(value["model_preprocess_ms"]),
        inference_ms=float(value["inference_ms"]),
        model_postprocess_ms=float(value["model_postprocess_ms"]),
        conversion_cascade_ms=float(value["conversion_cascade_ms"]),
        total_production_ms=float(value["total_production_ms"]),
        raw_predictions=int(value["raw_predictions"]),
        kept_predictions=int(value["kept_predictions"]),
    )


def write_cache(path: Path, rows: Iterable[ImagePrediction], identity: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"type": "identity", "value": identity}, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps({"type": "image", "value": asdict(row)}, sort_keys=True) + "\n")


def read_cache(path: Path) -> tuple[dict[str, object], list[ImagePrediction]]:
    identity: dict[str, object] | None = None
    rows: list[ImagePrediction] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["type"] == "identity":
                identity = item["value"]
                continue
            value = item["value"]
            rows.append(ImagePrediction(
                stem=value["stem"],
                predictions=tuple(_prediction_from_dict(row) for row in value["predictions"]),
                timing=_timing_from_dict(value["timing"]),
            ))
    if identity is None:
        raise ValueError("prediction cache identity header missing")
    return identity, rows

