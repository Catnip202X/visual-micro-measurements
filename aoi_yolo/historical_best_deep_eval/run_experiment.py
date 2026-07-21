from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import cv2

from .cache import read_cache
from .compare_runs import compare_runs
from .config import experiment_identity, load_config
from .dataset import ImageRecord, read_manifest
from .production_pass import build_ultralytics_infer, run_production_pass
from .scoring import score_cached_run
from .timing_stats import analyze_timing


PACKAGE_ROOT = Path(__file__).resolve().parent


def configure_ultralytics_environment(root: Path = PACKAGE_ROOT / "generated") -> Path:
    configured = Path(root) / "ultralytics_config"
    configured.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(configured))
    return Path(os.environ["YOLO_CONFIG_DIR"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_lock(lock_path: Path, identity: dict[str, object]) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("experiment") != identity:
        raise ValueError("manifest lock experiment identity mismatch")
    for name, expected_hash in lock.get("manifests", {}).items():
        path = lock_path.parent / name
        if not path.exists() or _sha256(path) != expected_hash:
            raise ValueError(f"manifest hash mismatch: {name}")


def _validate_completed_run(run_dir: Path, identity: dict[str, object], expected_rows: int) -> None:
    marker_path = run_dir / "inference_complete.json"
    cache_path = run_dir / "predictions.jsonl.gz"
    timing_path = run_dir / "timing.csv"
    if not marker_path.exists() or not cache_path.exists() or not timing_path.exists():
        raise ValueError(f"incomplete production artifact set: {run_dir}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("identity") != identity or marker.get("row_count") != expected_rows:
        raise ValueError(f"completed run identity/row mismatch: {run_dir}")
    if marker.get("cache_sha256") != _sha256(cache_path):
        raise ValueError(f"completed run cache hash mismatch: {run_dir}")
    if marker.get("timing_sha256") != _sha256(timing_path):
        raise ValueError(f"completed run timing hash mismatch: {run_dir}")


def _cuda_sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _warmup_paths(historical_dataset: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for split in ("train", "val"):
        paths.extend(sorted((historical_dataset / "images" / split).glob("*")))
    paths = [path for path in paths if path.is_file()]
    if len(paths) < count:
        raise ValueError(f"need {count} warm-up images, found {len(paths)}")
    return paths[:count]


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _smoke_records(config, count: int = 2) -> list[ImageRecord]:
    paths = _warmup_paths(config.historical_dataset, count)
    rows: list[ImageRecord] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode smoke image: {path}")
        height, width = image.shape[:2]
        rows.append(ImageRecord(
            stem=f"smoke-{path.stem}", image_path=path, json_path=path.with_suffix(".json"),
            sha256="smoke-not-a-manifest-member", width=width, height=height,
            group="historical-smoke", source_object_counts={}, target_object_counts={},
            positive_labels=frozenset(), context_labels=frozenset(),
        ))
    return rows


def run_smoke(config_path: Path, data_root: Path | None) -> Path:
    config = load_config(config_path, data_root=data_root)
    identity = experiment_identity(config)
    smoke_config = replace(config, images_per_run=2, warmup_count=1)
    configure_ultralytics_environment()
    infer = build_ultralytics_infer(config.weights, config)
    output_root = PACKAGE_ROOT / "generated" / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    return run_production_pass(
        "real_model_smoke",
        _smoke_records(config),
        smoke_config,
        output_root,
        identity,
        infer=infer,
        sync=_cuda_sync,
        warmup_paths=_warmup_paths(config.historical_dataset, smoke_config.warmup_count),
    )


def execute(config_path: Path, data_root: Path | None) -> dict:
    config = load_config(config_path, data_root=data_root)
    identity = experiment_identity(config)
    manifests_root = PACKAGE_ROOT / "generated" / "manifests"
    validate_manifest_lock(manifests_root / "manifest_lock.json", identity)
    manifests = {
        run_id: read_manifest(manifests_root / f"{run_id}.csv")
        for run_id in ("run_1", "run_2")
    }
    for run_id, rows in manifests.items():
        if len(rows) != config.images_per_run:
            raise ValueError(f"{run_id} does not contain exactly {config.images_per_run} images")
    if set(row.stem for row in manifests["run_1"]) & set(row.stem for row in manifests["run_2"]):
        raise ValueError("cross-run stem overlap detected before inference")

    production_root = PACKAGE_ROOT / "generated" / "production"
    private_results = PACKAGE_ROOT / "results" / "private"
    production_root.mkdir(parents=True, exist_ok=True)
    private_results.mkdir(parents=True, exist_ok=True)
    needs_inference = any(not (production_root / run_id).exists() for run_id in manifests)
    configure_ultralytics_environment()
    infer = build_ultralytics_infer(config.weights, config) if needs_inference else None
    warmups = _warmup_paths(config.historical_dataset, config.warmup_count)
    run_scores: dict[str, dict] = {}
    execution = {"identity": identity, "started_at_unix": time.time(), "runs": {}}

    for run_id in ("run_1", "run_2"):
        run_dir = production_root / run_id
        if not run_dir.exists():
            if infer is None:
                raise AssertionError("inference closure unavailable")
            run_dir = run_production_pass(
                run_id,
                manifests[run_id],
                config,
                production_root,
                identity,
                infer=infer,
                sync=_cuda_sync,
                warmup_paths=warmups,
            )
        _validate_completed_run(run_dir, identity, config.images_per_run)
        cached_identity, cached_rows = read_cache(run_dir / "predictions.jsonl.gz")
        if cached_identity != identity:
            raise ValueError(f"{run_id} cache identity mismatch")
        result_dir = private_results / run_id
        score = score_cached_run(
            run_dir / "predictions.jsonl.gz",
            manifests[run_id],
            identity,
            result_dir,
            hit_iou=config.hit_iou,
        )
        timing = analyze_timing(cached_rows, manifests[run_id], config.timing_brackets_ms, result_dir)
        run_scores[run_id] = score
        execution["runs"][run_id] = {
            "production_dir": str(run_dir),
            "result_dir": str(result_dir),
            "cache_sha256": _sha256(run_dir / "predictions.jsonl.gz"),
            "timing_sha256": _sha256(run_dir / "timing.csv"),
            "images": score["images"],
            "score": score,
            "timing": timing,
        }

    comparison_dir = private_results / "comparison"
    comparison = compare_runs(
        _read_csv(private_results / "run_1" / "timing_enriched.csv"),
        _read_csv(private_results / "run_2" / "timing_enriched.csv"),
        run_scores["run_1"],
        run_scores["run_2"],
        comparison_dir,
    )
    execution["comparison"] = comparison
    execution["finished_at_unix"] = time.time()
    (private_results / "execution_summary.json").write_text(
        json.dumps(execution, indent=2, sort_keys=True), encoding="utf-8"
    )
    return execution


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical-best exactly-once 2x500 evaluation")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--mode", choices=("smoke", "execute"), required=True)
    arguments = parser.parse_args()
    if arguments.mode == "smoke":
        path = run_smoke(arguments.config, arguments.data_root)
        print(f"real-model smoke completed: {path}")
    else:
        result = execute(arguments.config, arguments.data_root)
        print(json.dumps({"runs": list(result["runs"]), "finished_at_unix": result["finished_at_unix"]}, indent=2))


if __name__ == "__main__":
    main()
