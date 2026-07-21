from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable

import cv2


TARGET_MAP = {"pinhole": "pinhole", "inpinhole": "pinhole", "scratch": "scratch"}


@dataclass(frozen=True)
class ImageRecord:
    stem: str
    image_path: Path
    json_path: Path
    sha256: str
    width: int
    height: int
    group: str
    source_object_counts: dict[str, int]
    target_object_counts: dict[str, int]
    positive_labels: frozenset[str]
    context_labels: frozenset[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_members(dataset: Path) -> set[str]:
    members: set[str] = set()
    for split in ("train", "val"):
        folder = Path(dataset) / "images" / split
        if folder.exists():
            members.update(path.stem for path in folder.iterdir() if path.is_file())
    return members


def historical_hashes(dataset: Path) -> set[str]:
    hashes: set[str] = set()
    for split in ("train", "val"):
        folder = Path(dataset) / "images" / split
        if folder.exists():
            hashes.update(_sha256(path) for path in folder.iterdir() if path.is_file())
    return hashes


def acquisition_group(stem: str, sequence_group_size: int) -> str:
    if stem.isdigit():
        number = int(stem)
        return f"numeric-{(max(number, 1) - 1) // sequence_group_size:05d}"
    match = re.match(r"^([^-]+)-([^-]+)-", stem)
    if match:
        return f"session-{match.group(1)}-{match.group(2)}"
    prefix = re.split(r"[_-]", stem, maxsplit=1)[0]
    return f"prefix-{prefix}"


def _dimensions(payload: dict, image_path: Path) -> tuple[int, int]:
    width = int(payload.get("imageWidth") or 0)
    height = int(payload.get("imageHeight") or 0)
    if width > 0 and height > 0:
        return width, height
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    return int(image.shape[1]), int(image.shape[0])


def audit_candidates(
    source: Path,
    historical_stems: set[str],
    historical_sha256: set[str],
    sequence_group_size: int,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen_hashes: set[str] = set()
    for image_path in sorted(Path(source).glob("*.bmp"), key=lambda path: path.stem):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists() or image_path.stem in historical_stems:
            continue
        image_hash = _sha256(image_path)
        if image_hash in historical_sha256 or image_hash in seen_hashes:
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        contexts: set[str] = set()
        for shape in payload.get("shapes", []):
            label = str(shape.get("label", "")).strip().lower()
            points = shape.get("points") or []
            if not label or len(points) < 3:
                continue
            source_counts[label] += 1
            target = TARGET_MAP.get(label)
            if target:
                target_counts[target] += 1
            else:
                contexts.add(label)
        width, height = _dimensions(payload, image_path)
        records.append(
            ImageRecord(
                stem=image_path.stem,
                image_path=image_path.resolve(),
                json_path=json_path.resolve(),
                sha256=image_hash,
                width=width,
                height=height,
                group=acquisition_group(image_path.stem, sequence_group_size),
                source_object_counts=dict(sorted(source_counts.items())),
                target_object_counts=dict(sorted(target_counts.items())),
                positive_labels=frozenset(target_counts),
                context_labels=frozenset(contexts),
            )
        )
        seen_hashes.add(image_hash)
    return records


def _select_exact_groups(records: list[ImageRecord], n: int, seed: int) -> list[ImageRecord]:
    by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)
    groups = sorted(by_group.items())
    random.Random(seed).shuffle(groups)
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, items) in enumerate(groups):
        size = len(items)
        for total, chosen in sorted(tuple(reachable.items()), reverse=True):
            candidate = total + size
            if candidate <= n and candidate not in reachable:
                reachable[candidate] = chosen + (index,)
        if n in reachable:
            break
    if n not in reachable:
        nearest = max(reachable)
        raise ValueError(f"cannot form {n} eligible independent images; nearest group-safe size is {nearest}")
    chosen = set(reachable[n])
    return sorted((row for index in chosen for row in groups[index][1]), key=lambda row: row.stem)


def select_disjoint_runs(
    records: list[ImageRecord],
    historical_stems: set[str],
    seeds: tuple[int, int],
    run_size: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    unique: list[ImageRecord] = []
    hashes: set[str] = set()
    for record in sorted(records, key=lambda row: row.stem):
        if record.stem in historical_stems or record.sha256 in hashes:
            continue
        hashes.add(record.sha256)
        unique.append(record)
    if len(unique) < run_size * 2:
        raise ValueError(
            f"need {run_size * 2} eligible independent images, found {len(unique)}"
        )
    run1 = _select_exact_groups(unique, run_size, seeds[0])
    used_groups = {row.group for row in run1}
    remainder = [row for row in unique if row.group not in used_groups]
    run2 = _select_exact_groups(remainder, run_size, seeds[1])
    if {row.group for row in run1} & {row.group for row in run2}:
        raise AssertionError("cross-run group leakage")
    return run1, run2


def write_manifest(path: Path, run_id: str, seed: int, records: Iterable[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id", "seed", "stem", "image_path", "json_path", "sha256", "group",
        "width", "height", "positive_labels", "context_labels",
        "source_object_counts", "target_object_counts",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "run_id": run_id,
                "seed": seed,
                "stem": row.stem,
                "image_path": str(row.image_path),
                "json_path": str(row.json_path),
                "sha256": row.sha256,
                "group": row.group,
                "width": row.width,
                "height": row.height,
                "positive_labels": "|".join(sorted(row.positive_labels)),
                "context_labels": "|".join(sorted(row.context_labels)),
                "source_object_counts": json.dumps(row.source_object_counts, sort_keys=True),
                "target_object_counts": json.dumps(row.target_object_counts, sort_keys=True),
            })


def read_manifest(path: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(ImageRecord(
                stem=row["stem"],
                image_path=Path(row["image_path"]),
                json_path=Path(row["json_path"]),
                sha256=row["sha256"],
                width=int(row["width"]),
                height=int(row["height"]),
                group=row["group"],
                source_object_counts=json.loads(row["source_object_counts"]),
                target_object_counts=json.loads(row["target_object_counts"]),
                positive_labels=frozenset(filter(None, row["positive_labels"].split("|"))),
                context_labels=frozenset(filter(None, row["context_labels"].split("|"))),
            ))
    return records

