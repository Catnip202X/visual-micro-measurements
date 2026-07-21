from dataclasses import replace
import csv
from pathlib import Path

import numpy as np

from aoi_yolo.historical_best_deep_eval.cache import PredictionRecord
from aoi_yolo.historical_best_deep_eval.config import load_config
from aoi_yolo.historical_best_deep_eval.dataset import ImageRecord
from aoi_yolo.historical_best_deep_eval.production_pass import (
    InferenceOutput,
    apply_cascade,
    run_production_pass,
)


def record(stem: str) -> ImageRecord:
    return ImageRecord(
        stem=stem,
        image_path=Path(f"{stem}.bmp"),
        json_path=Path(f"{stem}.json"),
        sha256=f"sha-{stem}",
        width=10,
        height=10,
        group=f"g-{stem}",
        source_object_counts={},
        target_object_counts={},
        positive_labels=frozenset(),
        context_labels=frozenset(),
    )


def pred(label: str, confidence: float, area: int) -> PredictionRecord:
    return PredictionRecord(
        prediction_id=f"{label}-{confidence}",
        label=label,
        confidence=confidence,
        area_px=area,
        polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        kept=False,
    )


def test_historical_cascade():
    config = load_config(Path(__file__).parents[1] / "config.yaml", data_root=Path("."))
    rows = [
        pred("pinhole", 0.29, 500),
        pred("pinhole", 0.31, 149),
        pred("scratch", 0.21, 301),
    ]

    kept = apply_cascade(rows, config.rules)

    assert [(row.label, row.confidence) for row in kept] == [("scratch", 0.21)]


def test_each_measured_image_is_inferred_once(tmp_path):
    config = load_config(Path(__file__).parents[1] / "config.yaml", data_root=tmp_path)
    config = replace(config, images_per_run=3, warmup_count=0)
    records = [record("0001"), record("0002"), record("0003")]
    calls: list[int] = []

    def load_image(_path: Path):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def infer(image):
        calls.append(id(image))
        return InferenceOutput((), 1.0, 2.0, 3.0, conversion_ms=4.0)

    run_dir = run_production_pass(
        "run_1",
        records,
        config,
        tmp_path,
        {"weights_sha256": "abc"},
        infer=infer,
        image_loader=load_image,
        sync=lambda: None,
    )

    assert len(calls) == 3
    assert (run_dir / "inference_complete.json").exists()
    timing_lines = (run_dir / "timing.csv").read_text(encoding="utf-8").splitlines()
    assert len(timing_lines) == 4
    with (run_dir / "timing.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["conversion_cascade_ms"]) >= 4.0
