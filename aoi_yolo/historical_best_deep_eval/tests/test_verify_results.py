from aoi_yolo.historical_best_deep_eval.verify_results import validate_score_reconciliation


def test_score_reconciliation_accepts_exact_counts_and_rejects_mismatch():
    summary = {
        "overall": {"tp": 3, "fp": 2, "fn": 1, "annotated_objects": 4},
        "by_class": {
            "pinhole": {"tp": 2, "fp": 1, "fn": 1, "annotated_objects": 3},
            "scratch": {"tp": 1, "fp": 1, "fn": 0, "annotated_objects": 1},
        },
    }
    validate_score_reconciliation(summary, {"tp": 3, "fp": 2, "fn": 1}, slip_count=1)
    try:
        validate_score_reconciliation(summary, {"tp": 3, "fp": 2, "fn": 0}, slip_count=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("mismatched event counts were accepted")
