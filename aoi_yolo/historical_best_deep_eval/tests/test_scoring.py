import json
from pathlib import Path

from aoi_yolo.historical_best_deep_eval.cache import (
    ImagePrediction,
    PredictionRecord,
    TimingRecord,
    write_cache,
)
from aoi_yolo.historical_best_deep_eval.dataset import ImageRecord
from aoi_yolo.historical_best_deep_eval.scoring import score_cached_run, wilson_interval


def square(x0: float, y0: float, x1: float, y1: float):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def timing(stem: str) -> TimingRecord:
    return TimingRecord(1, stem, 20, 20, 1, 2, 3, 4, 10, 3, 2)


def record(tmp_path: Path, stem: str, shapes: list[dict]) -> ImageRecord:
    image_path = tmp_path / f"{stem}.bmp"
    image_path.write_bytes(b"not-read-by-scoring")
    json_path = tmp_path / f"{stem}.json"
    json_path.write_text(json.dumps({"imageWidth": 20, "imageHeight": 20, "shapes": shapes}))
    counts: dict[str, int] = {}
    for shape in shapes:
        label = "pinhole" if shape["label"] == "inpinhole" else shape["label"]
        if label in {"pinhole", "scratch"}:
            counts[label] = counts.get(label, 0) + 1
    return ImageRecord(
        stem, image_path, json_path, f"sha-{stem}", 20, 20, f"g-{stem}",
        counts, counts, frozenset(counts), frozenset(),
    )


def prediction(pid: str, label: str, polygon, kept: bool = True) -> PredictionRecord:
    return PredictionRecord(pid, label, 0.9, 25, polygon, kept)


def test_score_cache_matches_one_to_one_and_reconciles(tmp_path):
    shapes = [
        {"label": "inpinhole", "points": square(1, 1, 6, 6)},
        {"label": "pinhole", "points": square(10, 10, 15, 15)},
        {"label": "scratch", "points": square(2, 10, 8, 13)},
    ]
    manifest = [record(tmp_path, "0001", shapes)]
    cached = ImagePrediction(
        "0001",
        (
            prediction("p1", "pinhole", square(1, 1, 6, 6)),
            prediction("p2", "pinhole", square(1, 1, 6, 6)),
            prediction("s1", "scratch", square(16, 16, 19, 19)),
        ),
        timing("0001"),
    )
    cache_path = tmp_path / "predictions.jsonl.gz"
    identity = {"experiment_sha256": "frozen"}
    write_cache(cache_path, [cached], identity)

    result = score_cached_run(cache_path, manifest, identity, tmp_path / "scored", hit_iou=0.05)

    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 2
    assert result["overall"]["fn"] == 2
    assert result["by_class"]["pinhole"]["tp"] == 1
    assert result["by_class"]["pinhole"]["fp"] == 1
    assert result["by_class"]["pinhole"]["fn"] == 1
    assert result["by_class"]["scratch"]["tp"] == 0
    assert result["by_class"]["scratch"]["fp"] == 1
    assert result["by_class"]["scratch"]["fn"] == 1
    assert result["overall"]["tp"] + result["overall"]["fn"] == 3
    assert len((tmp_path / "scored" / "slips.csv").read_text().splitlines()) == 3


def test_scoring_rejects_identity_or_manifest_mismatch(tmp_path):
    manifest = [record(tmp_path, "0001", [])]
    cache_path = tmp_path / "predictions.jsonl.gz"
    write_cache(cache_path, [ImagePrediction("0001", (), timing("0001"))], {"id": "a"})

    try:
        score_cached_run(cache_path, manifest, {"id": "b"}, tmp_path / "bad", hit_iou=0.05)
    except ValueError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("identity mismatch was accepted")


def test_wilson_interval_bounds():
    low, high = wilson_interval(8, 10)
    assert 0.49 < low < 0.50
    assert 0.94 < high < 0.95
    assert wilson_interval(0, 0) == (None, None)
