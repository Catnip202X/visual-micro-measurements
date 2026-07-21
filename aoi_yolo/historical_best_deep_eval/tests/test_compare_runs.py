from aoi_yolo.historical_best_deep_eval.compare_runs import (
    bootstrap_differences,
    cliffs_delta,
    compare_runs,
    holm_adjust,
)


def score(tp_p, fp_p, fn_p, tp_s, fp_s, fn_s):
    def row(tp, fp, fn):
        return {"tp": tp, "fp": fp, "fn": fn}
    return {
        "by_class": {
            "pinhole": row(tp_p, fp_p, fn_p),
            "scratch": row(tp_s, fp_s, fn_s),
        }
    }


def timings(values, contexts):
    return [
        {"total_production_ms": value, "context": context, "timing_bracket": "<25 ms" if value < 25 else "25-50 ms"}
        for value, context in zip(values, contexts)
    ]


def test_cliffs_delta_bootstrap_and_holm():
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
    result = bootstrap_differences([1, 2, 3], [10, 11, 12], samples=200, seed=7)
    assert result["mean_difference"] < 0
    assert result["median_difference"] < 0
    adjusted = holm_adjust([0.01, 0.04, 0.20])
    assert adjusted == [0.03, 0.08, 0.20]


def test_compare_runs_writes_all_required_tests(tmp_path):
    run1 = timings([10, 12, 14, 16], ["no-target", "pinhole-only", "scratch-only", "mixed"])
    run2 = timings([30, 32, 34, 36], ["no-target", "pinhole-only", "scratch-only", "mixed"])
    result = compare_runs(
        run1,
        run2,
        score(8, 2, 2, 4, 1, 1),
        score(7, 3, 3, 3, 2, 2),
        tmp_path,
        bootstrap_samples=200,
        seed=11,
    )

    assert result["timing"]["cliffs_delta"] == -1.0
    assert result["timing"]["mann_whitney_p"] < 0.05
    assert "bracket_distribution" in result
    assert "composition_distribution" in result
    assert len(result["per_defect_metric_tests"]) == 4
    assert all("holm_p" in row for row in result["per_defect_metric_tests"])
    assert (tmp_path / "between_run_comparison.json").exists()
    assert (tmp_path / "per_defect_metric_tests.csv").exists()
