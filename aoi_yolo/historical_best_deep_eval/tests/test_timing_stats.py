import csv
from pathlib import Path

from aoi_yolo.historical_best_deep_eval.cache import ImagePrediction, TimingRecord
from aoi_yolo.historical_best_deep_eval.dataset import ImageRecord
from aoi_yolo.historical_best_deep_eval.timing_stats import (
    analyze_timing,
    defect_context,
    timing_bracket,
)


def manifest_row(stem: str, pinhole: int = 0, scratch: int = 0) -> ImageRecord:
    counts = {label: value for label, value in (("pinhole", pinhole), ("scratch", scratch)) if value}
    return ImageRecord(
        stem, Path(f"{stem}.bmp"), Path(f"{stem}.json"), f"sha-{stem}",
        100, 50, f"g-{stem}", counts, counts, frozenset(counts), frozenset(),
    )


def cached(stem: str, index: int, total_ms: float, raw: int = 0, kept: int = 0):
    timing = TimingRecord(index, stem, 100, 50, 1, 2, 3, 4, total_ms, raw, kept)
    return ImagePrediction(stem, (), timing)


def test_fixed_bracket_edges_and_defect_context():
    edges = (0.0, 25.0, 50.0, 100.0, float("inf"))
    assert timing_bracket(24.999, edges) == "<25 ms"
    assert timing_bracket(25.0, edges) == "25-50 ms"
    assert timing_bracket(100.0, edges) == ">=100 ms"
    assert defect_context({}) == "no-target"
    assert defect_context({"pinhole": 2}) == "pinhole-only"
    assert defect_context({"scratch": 1}) == "scratch-only"
    assert defect_context({"pinhole": 1, "scratch": 1}) == "mixed"


def test_timing_analysis_counts_all_rows_and_writes_graph(tmp_path):
    manifest = [
        manifest_row("a"),
        manifest_row("b", pinhole=1),
        manifest_row("c", scratch=1),
        manifest_row("d", pinhole=1, scratch=1),
    ]
    rows = [cached("a", 1, 20), cached("b", 2, 25, 1, 1), cached("c", 3, 75, 2, 1), cached("d", 4, 150, 3, 2)]
    result = analyze_timing(rows, manifest, (0, 25, 50, 100, float("inf")), tmp_path)

    assert result["images"] == 4
    assert sum(row["count"] for row in result["brackets"]) == 4
    assert {row["context"] for row in result["by_context"]} == {
        "no-target", "pinhole-only", "scratch-only", "mixed"
    }
    with (tmp_path / "timing_enriched.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert (tmp_path / "timing_graph.png").exists()
