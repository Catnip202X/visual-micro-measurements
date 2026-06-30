"""Turn edge-policy crop decisions into full-image ROI mask annotations.

The key idea is to keep the original BMP context and use prior crop windows as
mask labels that teach a context/attention model where close inspection should
happen.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


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


def rect_points(left: int, top: int, right: int, bottom: int) -> list[list[float]]:
    return [
        [float(left), float(top)],
        [float(right), float(top)],
        [float(right), float(bottom)],
        [float(left), float(bottom)],
    ]


def full_frame_shape(width: int, height: int) -> dict[str, Any]:
    return {
        "label": "full_roi",
        "points": rect_points(0, 0, width - 1, height - 1),
        "group_id": None,
        "description": "Full-frame ROI for non-edge/interior defect inspection",
        "shape_type": "polygon",
        "flags": {"source": "edge_policy_full_image"},
    }


def crop_roi_shape(data: dict[str, Any]) -> dict[str, Any]:
    flags = data.get("flags") or {}
    left = int(flags["crop_left"])
    top = int(flags["crop_top"])
    right = int(flags["crop_right"])
    bottom = int(flags["crop_bottom"])
    category = str(flags.get("dataset_category") or "")
    label = "corner_roi" if category == "edge_corner_defect" else "edge_roi"
    return {
        "label": label,
        "points": rect_points(left, top, right, bottom),
        "group_id": None,
        "description": "Full-image ROI reconstructed from edge-policy crop window",
        "shape_type": "polygon",
        "flags": {
            "source": "edge_policy_crop_window",
            "source_crop_image": data.get("imagePath"),
            "crop_left": left,
            "crop_top": top,
            "crop_right": right,
            "crop_bottom": bottom,
            "source_category": category,
            "auto_pass_defective": bool(flags.get("auto_pass_defective")),
        },
    }


def draw_review(image: np.ndarray, shapes: list[dict[str, Any]]) -> np.ndarray:
    review = image.copy()
    overlay = image.copy()
    colors = {
        "full_roi": (0, 150, 255),
        "edge_roi": (0, 220, 80),
        "corner_roi": (255, 0, 255),
    }
    for shape in shapes:
        label = str(shape.get("label") or "")
        color = colors.get(label, (255, 255, 0))
        pts = np.array(shape.get("points") or [], dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) < 3:
            continue
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(review, [pts], True, color, 4, cv2.LINE_AA)
        x, y, _, _ = cv2.boundingRect(pts)
        cv2.putText(review, label, (max(6, x + 6), max(24, y + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return cv2.addWeighted(review, 0.72, overlay, 0.28, 0)


def collect_edge_shapes(edge_policy: Path) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for category in ("edge_defect_strips", "edge_corner_defects"):
        labels_dir = edge_policy / category / "labels"
        if not labels_dir.exists():
            continue
        for json_path in sorted(labels_dir.glob("*.json")):
            data = read_json(json_path)
            flags = data.get("flags") or {}
            source_image = flags.get("source_image")
            if not source_image:
                continue
            by_source[str(source_image)].append(crop_roi_shape(data))
    return by_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-images", type=Path, default=Path("../test images"))
    parser.add_argument("--edge-policy", type=Path, default=Path("dataset_edge_policy"))
    parser.add_argument("--output", type=Path, default=Path("dataset_roi_masks_labelme"))
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--write-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_images = args.source_images.resolve()
    edge_policy = args.edge_policy.resolve()
    output = args.output.resolve()
    images_dir = output / "images"
    labels_dir = output / "labels"
    review_dir = output / "review_bmp"

    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.write_review:
        review_dir.mkdir(parents=True, exist_ok=True)

    edge_shapes = collect_edge_shapes(edge_policy)
    full_label_dir = edge_policy / "full_images" / "labels"
    full_sources = set()
    if full_label_dir.exists():
        for json_path in full_label_dir.glob("*.json"):
            data = read_json(json_path)
            source_image = (data.get("flags") or {}).get("source_image") or data.get("imagePath")
            if source_image:
                full_sources.add(str(source_image))

    source_names = sorted(set(edge_shapes) | full_sources)
    written = 0
    for image_name in source_names:
        image_path = source_images / image_name
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        shapes = list(edge_shapes.get(image_name) or [])
        if not shapes:
            shapes = [full_frame_shape(width, height)]

        link_or_copy(image_path, images_dir / image_name, args.copy_images)
        payload = {
            "version": "5.0.1",
            "flags": {
                "source_image": image_name,
                "roi_mask_strategy": "full_frame_with_edge_attention_windows",
                "has_edge_roi": any(shape["label"] in {"edge_roi", "corner_roi"} for shape in shapes),
            },
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        write_json(labels_dir / f"{Path(image_name).stem}.json", payload)
        if args.write_review:
            cv2.imwrite(str(review_dir / image_name), draw_review(image, shapes))
        written += 1

    print(f"written_images: {written}")
    print(f"edge_roi_images: {sum(1 for shapes in edge_shapes.values() if shapes)}")
    print(f"full_roi_images: {len(full_sources)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
