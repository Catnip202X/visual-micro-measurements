from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Iterable

import cv2
import numpy as np

from .cache import ImagePrediction, PredictionRecord, TimingRecord, write_cache
from .config import ExperimentConfig, ThresholdRule
from .dataset import ImageRecord


@dataclass(frozen=True)
class InferenceOutput:
    predictions: tuple[PredictionRecord, ...]
    model_preprocess_ms: float
    inference_ms: float
    model_postprocess_ms: float
    conversion_ms: float = 0.0


def apply_cascade(
    predictions: Iterable[PredictionRecord],
    rules: dict[str, ThresholdRule],
) -> list[PredictionRecord]:
    kept: list[PredictionRecord] = []
    for prediction in predictions:
        rule = rules.get(prediction.label)
        if rule and prediction.confidence >= rule.confidence and prediction.area_px >= rule.minimum_area:
            kept.append(replace(prediction, kept=True))
    return kept


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _default_loader(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def _write_timing(path: Path, rows: list[ImagePrediction]) -> None:
    fields = list(asdict(rows[0].timing).keys()) if rows else list(TimingRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row.timing))


def run_production_pass(
    run_id: str,
    records: list[ImageRecord],
    config: ExperimentConfig,
    output_root: Path,
    identity: dict[str, object],
    *,
    infer: Callable[[np.ndarray], InferenceOutput],
    image_loader: Callable[[Path], np.ndarray] = _default_loader,
    sync: Callable[[], None] = lambda: None,
    warmup_paths: Iterable[Path] = (),
) -> Path:
    if len(records) != config.images_per_run:
        raise ValueError(f"{run_id} manifest has {len(records)} rows, expected {config.images_per_run}")
    run_dir = Path(output_root) / run_id
    partial = Path(output_root) / f"{run_id}.partial"
    if run_dir.exists():
        raise FileExistsError(f"completed production run exists: {run_dir}")
    if partial.exists():
        raise FileExistsError(f"partial production run requires explicit archival: {partial}")
    partial.mkdir(parents=True)

    for path in tuple(warmup_paths)[: config.warmup_count]:
        image = image_loader(path)
        sync()
        infer(image)
        sync()

    rows: list[ImagePrediction] = []
    started_at = time.time()
    for index, record in enumerate(records, start=1):
        image = image_loader(record.image_path)
        sync()
        start_ns = time.perf_counter_ns()
        output = infer(image)
        sync()
        conversion_start_ns = time.perf_counter_ns()
        kept = apply_cascade(output.predictions, config.rules)
        kept_ids = {row.prediction_id for row in kept}
        cached_predictions = tuple(
            replace(row, kept=row.prediction_id in kept_ids) for row in output.predictions
        )
        end_ns = time.perf_counter_ns()
        timing = TimingRecord(
            index=index,
            stem=record.stem,
            width=record.width,
            height=record.height,
            model_preprocess_ms=output.model_preprocess_ms,
            inference_ms=output.inference_ms,
            model_postprocess_ms=output.model_postprocess_ms,
            conversion_cascade_ms=output.conversion_ms + (end_ns - conversion_start_ns) / 1e6,
            total_production_ms=(end_ns - start_ns) / 1e6,
            raw_predictions=len(cached_predictions),
            kept_predictions=len(kept),
        )
        rows.append(ImagePrediction(record.stem, cached_predictions, timing))
        if index % 25 == 0:
            print(f"{run_id}: production {index}/{len(records)}", flush=True)

    cache_path = partial / "predictions.jsonl.gz"
    timing_path = partial / "timing.csv"
    write_cache(cache_path, rows, identity)
    _write_timing(timing_path, rows)
    marker = {
        "run_id": run_id,
        "row_count": len(rows),
        "unique_stems": len({row.stem for row in rows}),
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "cache_sha256": _file_sha(cache_path),
        "timing_sha256": _file_sha(timing_path),
        "identity": identity,
    }
    (partial / "inference_complete.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8"
    )
    partial.rename(run_dir)
    return run_dir


def build_ultralytics_infer(weights: Path, config: ExperimentConfig):
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def infer(image: np.ndarray) -> InferenceOutput:
        result = model.predict(
            image,
            imgsz=config.imgsz,
            conf=config.raw_confidence,
            iou=0.7,
            max_det=config.max_det,
            device=config.device,
            retina_masks=True,
            verbose=False,
        )[0]
        conversion_start_ns = time.perf_counter_ns()
        predictions: list[PredictionRecord] = []
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            height, width = image.shape[:2]
            for index, (class_id, confidence, points) in enumerate(
                zip(classes, confidences, result.masks.xy)
            ):
                label = str(result.names[int(class_id)]).lower()
                if label not in config.rules or len(points) < 3:
                    continue
                polygon = tuple((float(x), float(y)) for x, y in points)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
                predictions.append(PredictionRecord(
                    prediction_id=f"pred:{index}",
                    label=label,
                    confidence=float(confidence),
                    area_px=int(mask.sum()),
                    polygon=polygon,
                    kept=False,
                ))
        speed = result.speed
        conversion_ms = (time.perf_counter_ns() - conversion_start_ns) / 1e6
        return InferenceOutput(
            tuple(predictions),
            float(speed.get("preprocess", 0.0)),
            float(speed.get("inference", 0.0)),
            float(speed.get("postprocess", 0.0)),
            conversion_ms,
        )

    return infer
