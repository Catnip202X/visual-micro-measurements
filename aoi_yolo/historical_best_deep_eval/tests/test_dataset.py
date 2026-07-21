from pathlib import Path

import pytest

from aoi_yolo.historical_best_deep_eval.dataset import (
    ImageRecord,
    historical_members,
    select_disjoint_runs,
)


def record(stem: str, group: str, sha: str | None = None) -> ImageRecord:
    return ImageRecord(
        stem=stem,
        image_path=Path(f"{stem}.bmp"),
        json_path=Path(f"{stem}.json"),
        sha256=sha or f"sha-{stem}",
        width=10,
        height=10,
        group=group,
        source_object_counts={},
        target_object_counts={},
        positive_labels=frozenset(),
        context_labels=frozenset(),
    )


def test_historical_members_reads_train_and_val(tmp_path):
    for split, names in (("train", ("0001.bmp", "0002.png")), ("val", ("0003.bmp",))):
        folder = tmp_path / "images" / split
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_bytes(b"x")

    assert historical_members(tmp_path) == {"0001", "0002", "0003"}


def test_runs_are_group_disjoint_and_exclude_history():
    records = [
        record("0001", "g0"),
        record("0002", "g1"),
        record("0003", "g2"),
        record("0004", "g3"),
        record("0005", "g4"),
        record("0006", "g5"),
        record("0007", "g6"),
    ]
    run1, run2 = select_disjoint_runs(records, {"0001"}, (11, 22), 3)

    assert len(run1) == len(run2) == 3
    assert {r.stem for r in run1}.isdisjoint(r.stem for r in run2)
    assert {r.group for r in run1}.isdisjoint(r.group for r in run2)
    assert "0001" not in {r.stem for r in run1 + run2}


def test_duplicate_hashes_are_excluded_from_independent_pool():
    records = [
        record("0001", "g1", "same"),
        record("0002", "g2", "same"),
        record("0003", "g3"),
        record("0004", "g4"),
        record("0005", "g5"),
        record("0006", "g6"),
        record("0007", "g7"),
    ]
    run1, run2 = select_disjoint_runs(records, set(), (11, 22), 3)
    selected = [row.sha256 for row in run1 + run2]
    assert len(selected) == len(set(selected))


def test_selection_fails_if_pool_is_too_small():
    records = [record("0001", "g1"), record("0002", "g2")]
    with pytest.raises(ValueError, match="eligible independent images"):
        select_disjoint_runs(records, set(), (11, 22), 2)

