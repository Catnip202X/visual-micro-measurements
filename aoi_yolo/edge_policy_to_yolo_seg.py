"""Convert dataset_edge_policy LabelMe outputs to YOLO segmentation format.

This combines:
- full_images: no true edge-contact defect, original image context
- edge_defect_strips: true edge-contact defects, edge-running crop context
- edge_corner_defects: corner cases, if any

Images are hardlinked by default to avoid duplicating large BMP data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def normalized_polygon(points: list[list[float]], width: float, height: float) -> list[float]:
    values: list[float] = []
    for point in points:
        if len(point) < 2:
            continue
        x = min(max(float(point[0]) / width, 0.0), 1.0)
        y = min(max(float(point[1]) / height, 0.0), 1.0)
        values.extend([x, y])
    return values


def write_yaml(path: Path, dataset_root: Path, classes: list[str]) -> None:
    names = ", ".join([f"{idx}: {name!r}" for idx, name in enumerate(classes)])
    root = str(dataset_root.resolve()).replace("\\", "/")
    path.write_text(
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {{{names}}}\n",
        encoding="utf-8",
    )


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_image(images_dir: Path, image_path_value: str | None, json_path: Path) -> Path | None:
    candidates: list[Path] = []
    if image_path_value:
        candidates.append(images_dir / image_path_value)
    for ext in IMAGE_EXTENSIONS:
        candidates.append(images_dir / f"{json_path.stem}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def iter_label_sources(root: Path) -> list[tuple[str, Path, Path]]:
    sources = []
    for category in ("full_images", "edge_defect_strips", "edge_corner_defects"):
        labels_dir = root / category / "labels"
        images_dir = root / category / "images"
        if labels_dir.exists() and images_dir.exists():
            sources.append((category, labels_dir, images_dir))
    return sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("dataset_edge_policy"))
    parser.add_argument("--out", type=Path, default=Path("dataset_edge_policy_yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--include", nargs="*", default=["pinhole", "inpinhole", "scratch"])
    parser.add_argument("--rename", nargs="*", default=["inpinhole=pinhole"])
    parser.add_argument("--copy-images", action="store_true", help="Copy instead of hardlinking images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    out = args.out.resolve()
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    include = set(args.include or [])
    rename: dict[str, str] = {}
    for item in args.rename or []:
        if "=" not in item:
            raise SystemExit(f"--rename expects old=new, got {item}")
        old, new = item.split("=", 1)
        rename[old.strip()] = new.strip()

    records = []
    classes = set()
    skipped = []
    for category, labels_dir, images_dir in iter_label_sources(src):
        for json_path in sorted(labels_dir.glob("*.json")):
            try:
                data = read_json(json_path)
            except Exception as exc:
                skipped.append((json_path.name, f"json error: {exc}"))
                continue
            width = data.get("imageWidth")
            height = data.get("imageHeight")
            if not width or not height:
                skipped.append((json_path.name, "missing image size"))
                continue
            image_path = find_image(images_dir, data.get("imagePath"), json_path)
            if image_path is None:
                skipped.append((json_path.name, "missing image"))
                continue
            yolo_shapes = []
            for shape in data.get("shapes") or []:
                label = str(shape.get("label") or "").strip()
                label = rename.get(label, label)
                if include and label not in include:
                    continue
                points = shape.get("points") or []
                if len(points) < 3:
                    continue
                poly = normalized_polygon(points, float(width), float(height))
                if len(poly) < 6:
                    continue
                yolo_shapes.append((label, poly))
                classes.add(label)
            if yolo_shapes:
                group = str(data.get("flags", {}).get("source_image") or image_path.name)
                records.append((category, group, json_path, image_path, yolo_shapes))

    class_names = sorted(classes)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    groups = sorted({record[1] for record in records})
    random.Random(args.seed).shuffle(groups)
    val_count = max(1, int(round(len(groups) * args.val_ratio))) if groups else 0
    val_groups = set(groups[:val_count])

    split_counts = {"train": 0, "val": 0}
    for category, group, json_path, image_path, shapes in records:
        split = "val" if group in val_groups else "train"
        stem = f"{category}_{image_path.stem}"
        image_ext = image_path.suffix.lower()
        out_image = out / "images" / split / f"{stem}{image_ext}"
        out_label = out / "labels" / split / f"{stem}.txt"
        link_or_copy(image_path, out_image, args.copy_images)
        lines = []
        for label, poly in shapes:
            coords = " ".join(f"{value:.6f}" for value in poly)
            lines.append(f"{class_to_id[label]} {coords}")
        out_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        split_counts[split] += 1

    (out / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    write_yaml(out / "data.yaml", out, class_names)
    print(f"converted_images: {len(records)}")
    print(f"train_images: {split_counts['train']}")
    print(f"val_images: {split_counts['val']}")
    print(f"classes: {class_names}")
    print(f"skipped_files: {len(skipped)}")
    for name, reason in skipped[:20]:
        print(f"  skipped {name}: {reason}")


if __name__ == "__main__":
    main()
