from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .config import experiment_identity, load_config
from .dataset import (
    audit_candidates,
    historical_hashes,
    historical_members,
    select_disjoint_runs,
    write_manifest,
)


def _manifest_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    package = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Freeze two leakage-free historical-best evaluation manifests.")
    parser.add_argument("--config", type=Path, default=package / "config.yaml")
    parser.add_argument("--output", type=Path, default=package / "generated" / "manifests")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    config = load_config(args.config, data_root=args.data_root)
    history = historical_members(config.historical_dataset)
    history_hashes = historical_hashes(config.historical_dataset)
    candidates = audit_candidates(
        config.canonical_source, history, history_hashes, config.sequence_group_size
    )
    run1, run2 = select_disjoint_runs(
        candidates, history, config.run_seeds, config.images_per_run
    )
    args.output.mkdir(parents=True, exist_ok=False)
    paths = []
    for index, records in enumerate((run1, run2), start=1):
        path = args.output / f"run_{index}.csv"
        write_manifest(path, f"run_{index}", config.run_seeds[index - 1], records)
        paths.append(path)
    audit = {
        "canonical_pairs": len(list(config.canonical_source.glob("*.bmp"))),
        "historical_stems": len(history),
        "historical_hashes": len(history_hashes),
        "eligible_records": len(candidates),
        "run_1_images": len(run1),
        "run_2_images": len(run2),
        "shared_stems": len({r.stem for r in run1} & {r.stem for r in run2}),
        "shared_groups": len({r.group for r in run1} & {r.group for r in run2}),
        "shared_hashes": len({r.sha256 for r in run1} & {r.sha256 for r in run2}),
        "run_1_target_objects": dict(sum((Counter(r.target_object_counts) for r in run1), Counter())),
        "run_2_target_objects": dict(sum((Counter(r.target_object_counts) for r in run2), Counter())),
        "run_1_positive_images": dict(Counter(label for r in run1 for label in r.positive_labels)),
        "run_2_positive_images": dict(Counter(label for r in run2 for label in r.positive_labels)),
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    lock = {
        "experiment": experiment_identity(config),
        "manifests": {path.name: _manifest_sha(path) for path in paths},
    }
    (args.output / "manifest_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

