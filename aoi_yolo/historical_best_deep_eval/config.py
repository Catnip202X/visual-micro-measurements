from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ThresholdRule:
    confidence: float
    minimum_area: int


@dataclass(frozen=True)
class ExperimentConfig:
    weights: Path
    canonical_source: Path
    historical_dataset: Path
    imgsz: int
    raw_confidence: float
    hit_iou: float
    rules: dict[str, ThresholdRule]
    run_seeds: tuple[int, int]
    images_per_run: int
    sequence_group_size: int
    warmup_count: int
    timing_brackets_ms: tuple[float, ...]
    device: str
    max_det: int


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: Path, data_root: Path | None = None) -> ExperimentConfig:
    values: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(data_root or os.environ.get("AOI_DATA_ROOT") or Path(path).resolve().parents[2]).resolve()
    rules = {
        label: ThresholdRule(float(rule["confidence"]), int(rule["minimum_area"]))
        for label, rule in values["rules"].items()
    }
    edges = tuple(float("inf") if value == "inf" else float(value) for value in values["timing_brackets_ms"])
    config = ExperimentConfig(
        weights=_resolve(root, values["weights"]),
        canonical_source=_resolve(root, values["canonical_source"]),
        historical_dataset=_resolve(root, values["historical_dataset"]),
        imgsz=int(values["imgsz"]),
        raw_confidence=float(values["raw_confidence"]),
        hit_iou=float(values["hit_iou"]),
        rules=rules,
        run_seeds=tuple(int(seed) for seed in values["run_seeds"]),
        images_per_run=int(values["images_per_run"]),
        sequence_group_size=int(values["sequence_group_size"]),
        warmup_count=int(values["warmup_count"]),
        timing_brackets_ms=edges,
        device=str(values["device"]),
        max_det=int(values["max_det"]),
    )
    if config.run_seeds[0] == config.run_seeds[1]:
        raise ValueError("run seeds must differ")
    if len(config.timing_brackets_ms) < 2 or config.timing_brackets_ms[0] != 0:
        raise ValueError("timing brackets must start at zero")
    return config


def experiment_identity(config: ExperimentConfig, weights: Path | None = None) -> dict[str, object]:
    weight_path = Path(weights or config.weights)
    return {
        "weights_path": str(weight_path.resolve()),
        "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        "imgsz": config.imgsz,
        "raw_confidence": config.raw_confidence,
        "hit_iou": config.hit_iou,
        "rules": {
            label: {"confidence": rule.confidence, "minimum_area": rule.minimum_area}
            for label, rule in sorted(config.rules.items())
        },
        "run_seeds": list(config.run_seeds),
        "images_per_run": config.images_per_run,
        "sequence_group_size": config.sequence_group_size,
        "timing_brackets_ms": ["inf" if value == float("inf") else value for value in config.timing_brackets_ms],
    }


def identity_json(config: ExperimentConfig, weights: Path | None = None) -> str:
    return json.dumps(experiment_identity(config, weights), indent=2, sort_keys=True)

