from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .cache import read_cache
from .config import experiment_identity, load_config
from .dataset import read_manifest
from .run_experiment import PACKAGE_ROOT, _validate_completed_run, validate_manifest_lock


def _csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_score_reconciliation(summary: dict, event_counts: dict[str, int], slip_count: int) -> None:
    overall = summary["overall"]
    assert overall["tp"] + overall["fn"] == overall["annotated_objects"]
    assert overall["tp"] == sum(row["tp"] for row in summary["by_class"].values())
    assert overall["fp"] == sum(row["fp"] for row in summary["by_class"].values())
    assert overall["fn"] == sum(row["fn"] for row in summary["by_class"].values())
    for outcome in ("tp", "fp", "fn"):
        assert int(overall[outcome]) == int(event_counts.get(outcome, 0))
    assert slip_count == overall["fn"]


def verify(config_path: Path, data_root: Path | None = None) -> dict:
    config = load_config(config_path, data_root=data_root)
    identity = experiment_identity(config)
    manifests_root = PACKAGE_ROOT / "generated" / "manifests"
    production_root = PACKAGE_ROOT / "generated" / "production"
    results_root = PACKAGE_ROOT / "results" / "private"
    validate_manifest_lock(manifests_root / "manifest_lock.json", identity)
    manifests = {
        run_id: read_manifest(manifests_root / f"{run_id}.csv")
        for run_id in ("run_1", "run_2")
    }
    assert not ({row.stem for row in manifests["run_1"]} & {row.stem for row in manifests["run_2"]})
    assert not ({row.sha256 for row in manifests["run_1"]} & {row.sha256 for row in manifests["run_2"]})
    assert not ({row.group for row in manifests["run_1"]} & {row.group for row in manifests["run_2"]})
    verification = {"identity": identity, "runs": {}, "cross_run": {"shared_stems": 0, "shared_hashes": 0, "shared_groups": 0}}

    for run_id in ("run_1", "run_2"):
        manifest = manifests[run_id]
        assert len(manifest) == config.images_per_run == 500
        run_dir = production_root / run_id
        _validate_completed_run(run_dir, identity, 500)
        cached_identity, cached = read_cache(run_dir / "predictions.jsonl.gz")
        assert cached_identity == identity
        assert len(cached) == 500
        assert len({row.stem for row in cached}) == 500
        assert [row.stem for row in cached] == [row.stem for row in manifest]
        result_dir = results_root / run_id
        summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
        events = _csv_rows(result_dir / "events.csv")
        slips = _csv_rows(result_dir / "slips.csv")
        validate_score_reconciliation(summary, Counter(row["outcome"] for row in events), len(slips))
        timing = _csv_rows(result_dir / "timing_enriched.csv")
        brackets = _csv_rows(result_dir / "timing_brackets.csv")
        assert len(timing) == 500
        assert len({row["stem"] for row in timing}) == 500
        assert sum(int(row["count"]) for row in brackets) == 500
        graph = result_dir / "timing_graph.png"
        assert graph.exists() and graph.stat().st_size > 10_000
        verification["runs"][run_id] = {
            "manifest_rows": len(manifest),
            "cache_rows": len(cached),
            "unique_cache_stems": len({row.stem for row in cached}),
            "timing_rows": len(timing),
            "bracket_rows_reconciled": sum(int(row["count"]) for row in brackets),
            "event_counts": dict(Counter(row["outcome"] for row in events)),
            "slip_rows": len(slips),
            "annotated_objects": summary["overall"]["annotated_objects"],
            "graph_bytes": graph.stat().st_size,
        }
    comparison = results_root / "comparison" / "between_run_comparison.json"
    assert comparison.exists()
    verification["comparison_present"] = True
    (results_root / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8"
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify historical-best 2x500 artifacts")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config.yaml")
    parser.add_argument("--data-root", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.config, arguments.data_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
