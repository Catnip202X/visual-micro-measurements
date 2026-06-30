"""Prepare an edge-policy dataset for AOI defect handling.

Policy:
- First detect likely object edges from image pixels (dry sweep).
- If defects are near a detected object edge, save edge-running strip crops.
- If no edge-near defects are present in an image, keep the original full image.
- Always write LabelMe JSON with ``imageData`` removed.
- Optionally write baked review BMPs with annotations and detected edge lines.

This does not use the JSON ``edge`` shapes to choose crops. JSON edges are only
recorded as cross-reference metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crop_labelme_dataset import (
    DEFAULT_EXCLUDE_LABELS,
    DetectedEdge,
    EdgeAssignment,
    assign_detected_edge,
    clip_shape_to_crop,
    detect_object_edges,
    read_labelme,
    selected_shapes,
    shape_bounds,
    write_labelme,
)


def defect_edge_gap(shape: dict[str, Any], assignment: EdgeAssignment) -> float:
    bounds = shape_bounds(shape)
    if bounds is None:
        return float("inf")
    x1, y1, x2, y2 = bounds
    edge = assignment.edge
    if edge.orientation == "vertical":
        if x1 <= edge.position <= x2:
            return 0.0
        return min(abs(x1 - edge.position), abs(x2 - edge.position))
    if y1 <= edge.position <= y2:
        return 0.0
    return min(abs(y1 - edge.position), abs(y2 - edge.position))


def shape_center(shape: dict[str, Any]) -> tuple[float, float]:
    bounds = shape_bounds(shape)
    if bounds is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = bounds
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def json_edge_orientation_from_bounds(shape: dict[str, Any]) -> str:
    bounds = shape_bounds(shape)
    if bounds is None:
        return "unknown"
    x1, y1, x2, y2 = bounds
    return "vertical" if (y2 - y1) >= (x2 - x1) else "horizontal"


def nearest_json_edge_distance(
    defect: dict[str, Any],
    source_shapes: list[dict[str, Any]],
    orientation: str,
) -> float | None:
    cx, cy = shape_center(defect)
    best: float | None = None
    for shape in source_shapes:
        if str(shape.get("label", "")) != "edge":
            continue
        if json_edge_orientation_from_bounds(shape) != orientation:
            continue
        bounds = shape_bounds(shape)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        edge_pos = (x1 + x2) / 2.0 if orientation == "vertical" else (y1 + y2) / 2.0
        distance = abs(cx - edge_pos) if orientation == "vertical" else abs(cy - edge_pos)
        if best is None or distance < best:
            best = float(distance)
    return best


def assignment_for_orientation(
    defect: dict[str, Any],
    detected_edges: list[DetectedEdge],
    source_shapes: list[dict[str, Any]],
    orientation: str,
) -> EdgeAssignment | None:
    cx, cy = shape_center(defect)
    candidates = [edge for edge in detected_edges if edge.orientation == orientation]
    if not candidates:
        return None
    if orientation == "vertical":
        edge = min(candidates, key=lambda e: abs(cx - e.position))
        distance = abs(cx - edge.position)
        side = "right_of_edge" if cx >= edge.position else "left_of_edge"
    else:
        edge = min(candidates, key=lambda e: abs(cy - e.position))
        distance = abs(cy - edge.position)
        side = "below_edge" if cy >= edge.position else "above_edge"
    json_distance = nearest_json_edge_distance(defect, source_shapes, orientation)
    return EdgeAssignment(edge, float(distance), side, orientation if json_distance is not None else "none", json_distance)


def edge_contacts_for_defect(
    defect: dict[str, Any],
    detected_edges: list[DetectedEdge],
    source_shapes: list[dict[str, Any]],
    edge_gap_px: float,
    json_edge_max_distance: float,
) -> list[tuple[EdgeAssignment, float]]:
    contacts: list[tuple[EdgeAssignment, float]] = []
    for orientation in ("vertical", "horizontal"):
        assignment = assignment_for_orientation(defect, detected_edges, source_shapes, orientation)
        if assignment is None:
            continue
        gap = defect_edge_gap(defect, assignment)
        if is_edge_policy_defect(assignment, gap, edge_gap_px, json_edge_max_distance):
            contacts.append((assignment, gap))
    return contacts


def dynamic_edge_strip(
    shape: dict[str, Any],
    assignment: EdgeAssignment,
    image_w: int,
    image_h: int,
    pad: int,
    min_width: int,
    max_width: int,
) -> tuple[int, int, int, int]:
    bounds = shape_bounds(shape)
    if bounds is None:
        return 0, 0, image_w, image_h

    x1, y1, x2, y2 = bounds
    edge = assignment.edge
    if edge.orientation == "vertical":
        low = min(edge.position, x1, x2) - pad
        high = max(edge.position, x1, x2) + pad
        width = int(math.ceil(high - low))
        width = min(max(width, min_width), max_width, image_w)
        center = (low + high) / 2.0
        left = int(round(center - width / 2.0))
        right = left + width
        if left < 0:
            right -= left
            left = 0
        if right > image_w:
            left -= right - image_w
            right = image_w
        return max(0, left), 0, min(image_w, right), image_h

    low = min(edge.position, y1, y2) - pad
    high = max(edge.position, y1, y2) + pad
    height = int(math.ceil(high - low))
    height = min(max(height, min_width), max_width, image_h)
    center = (low + high) / 2.0
    top = int(round(center - height / 2.0))
    bottom = top + height
    if top < 0:
        bottom -= top
        top = 0
    if bottom > image_h:
        top -= bottom - image_h
        bottom = image_h
    return 0, max(0, top), image_w, min(image_h, bottom)


def dynamic_corner_crop(
    shape: dict[str, Any],
    vertical: EdgeAssignment,
    horizontal: EdgeAssignment,
    image_w: int,
    image_h: int,
    pad: int,
    min_width: int,
    max_width: int,
) -> tuple[int, int, int, int]:
    vertical_crop = dynamic_edge_strip(shape, vertical, image_w, image_h, pad, min_width, max_width)
    horizontal_crop = dynamic_edge_strip(shape, horizontal, image_w, image_h, pad, min_width, max_width)
    left = vertical_crop[0]
    right = vertical_crop[2]
    top = horizontal_crop[1]
    bottom = horizontal_crop[3]
    return left, top, right, bottom


def make_labelme(
    source_json: dict[str, Any],
    image_name: str,
    image_w: int,
    image_h: int,
    shapes: list[dict[str, Any]],
    flags: dict[str, Any],
) -> dict[str, Any]:
    out_flags = dict(source_json.get("flags") or {})
    out_flags.update(flags)
    return {
        "version": source_json.get("version", "5.0.1"),
        "flags": out_flags,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": image_h,
        "imageWidth": image_w,
    }


def draw_review(
    image: np.ndarray,
    shapes: list[dict[str, Any]],
    assignment: EdgeAssignment | None = None,
    extra_assignment: EdgeAssignment | None = None,
) -> np.ndarray:
    review = image.copy()
    colors = {
        "pinhole": (0, 0, 255),
        "inpinhole": (0, 128, 255),
        "scratch": (255, 0, 0),
        "edge": (0, 255, 255),
    }
    for shape in shapes:
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        label = str(shape.get("label", ""))
        color = colors.get(label, (0, 220, 80))
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(review, [pts], True, color, 2, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(pts)
        cv2.putText(review, label, (max(2, x), max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    for line_assignment in [assignment, extra_assignment]:
        if line_assignment is None:
            continue
        assignment = line_assignment
        edge = assignment.edge
        if edge.orientation == "vertical":
            x = int(edge.position)
            cv2.line(review, (x, 0), (x, review.shape[0] - 1), (255, 255, 0), 2, cv2.LINE_AA)
        else:
            y = int(edge.position)
            cv2.line(review, (0, y), (review.shape[1] - 1, y), (255, 255, 0), 2, cv2.LINE_AA)
    return review


def add_assignment_flags(shape: dict[str, Any], assignment: EdgeAssignment, gap: float, is_edge_defect: bool) -> dict[str, Any]:
    out = dict(shape)
    out["flags"] = dict(shape.get("flags") or {})
    out["flags"].update(
        {
            "edge_policy_is_edge_defect": is_edge_defect,
            "detected_edge_orientation": assignment.edge.orientation,
            "detected_edge_position": assignment.edge.position,
            "detected_edge_distance_center_px": round(assignment.distance, 3),
            "detected_edge_gap_px": round(gap, 3),
            "detected_edge_side": assignment.side,
            "json_edge_orientation_nearest": assignment.json_edge_orientation,
            "json_edge_distance_px": None
            if assignment.json_edge_distance is None
            else round(assignment.json_edge_distance, 3),
        }
    )
    return out


def is_edge_policy_defect(assignment: EdgeAssignment, gap: float, edge_gap_px: float, json_edge_max_distance: float) -> bool:
    if gap > edge_gap_px:
        return False
    if assignment.json_edge_orientation == "none":
        return False
    if assignment.json_edge_orientation != assignment.edge.orientation:
        return False
    if assignment.json_edge_distance is None:
        return False
    return assignment.json_edge_distance <= json_edge_max_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../test images"))
    parser.add_argument("--output", type=Path, default=Path("dataset_edge_policy"))
    parser.add_argument("--edge-gap-px", type=float, default=5.0)
    parser.add_argument("--json-edge-max-distance", type=float, default=120.0)
    parser.add_argument("--pad", type=int, default=48)
    parser.add_argument("--min-strip-width", type=int, default=128)
    parser.add_argument("--max-strip-width", type=int, default=640)
    parser.add_argument("--include-label", action="append", default=None)
    parser.add_argument("--exclude-label", action="append", default=sorted(DEFAULT_EXCLUDE_LABELS))
    parser.add_argument("--write-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    edge_img_dir = output / "edge_defect_strips" / "images"
    edge_label_dir = output / "edge_defect_strips" / "labels"
    edge_review_dir = output / "edge_defect_strips" / "review_bmp"
    corner_img_dir = output / "edge_corner_defects" / "images"
    corner_label_dir = output / "edge_corner_defects" / "labels"
    corner_review_dir = output / "edge_corner_defects" / "review_bmp"
    full_img_dir = output / "full_images" / "images"
    full_label_dir = output / "full_images" / "labels"
    full_review_dir = output / "full_images" / "review_bmp"

    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    for directory in (edge_img_dir, edge_label_dir, corner_img_dir, corner_label_dir, full_img_dir, full_label_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.write_review:
        edge_review_dir.mkdir(parents=True, exist_ok=True)
        corner_review_dir.mkdir(parents=True, exist_ok=True)
        full_review_dir.mkdir(parents=True, exist_ok=True)

    include_labels = set(args.include_label) if args.include_label else None
    exclude_labels = set(args.exclude_label or [])

    json_paths = sorted(source.glob("*.json"))
    if args.limit:
        json_paths = json_paths[: args.limit]

    rows: list[dict[str, Any]] = []
    malformed = 0
    skipped = 0
    edge_images = 0
    corner_images = 0
    full_images = 0
    edge_strips = 0
    corner_crops = 0

    for json_path in json_paths:
        data = read_labelme(json_path)
        if not isinstance(data, dict):
            malformed += 1
            continue
        image_path = source / str(data.get("imagePath") or f"{json_path.stem}.bmp")
        if not image_path.exists():
            image_path = json_path.with_suffix(".bmp")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue

        image_h, image_w = image.shape[:2]
        source_shapes = list(data.get("shapes") or [])
        defects = selected_shapes(source_shapes, include_labels, exclude_labels)
        if not defects:
            continue

        detected_edges = detect_object_edges(image)
        assignments = [(shape, assign_detected_edge(shape, detected_edges, source_shapes)) for shape in defects]
        classified = [(shape, assignment, defect_edge_gap(shape, assignment)) for shape, assignment in assignments]
        contact_map = {
            id(shape): edge_contacts_for_defect(
                shape,
                detected_edges,
                source_shapes,
                args.edge_gap_px,
                args.json_edge_max_distance,
            )
            for shape in defects
        }
        corner_defects = [(shape, contacts) for shape in defects if len(contact_map[id(shape)]) >= 2 for contacts in [contact_map[id(shape)]]]
        edge_defects = [(shape, contacts[0][0], contacts[0][1]) for shape in defects if len(contact_map[id(shape)]) == 1 for contacts in [contact_map[id(shape)]]]

        if corner_defects:
            corner_images += 1
            seen_crops: set[tuple[int, int, int, int]] = set()
            for idx, (anchor, contacts) in enumerate(corner_defects):
                vertical = next(item for item in contacts if item[0].edge.orientation == "vertical")[0]
                horizontal = next(item for item in contacts if item[0].edge.orientation == "horizontal")[0]
                crop = dynamic_corner_crop(
                    anchor,
                    vertical,
                    horizontal,
                    image_w,
                    image_h,
                    args.pad,
                    args.min_strip_width,
                    args.max_strip_width,
                )
                if crop in seen_crops:
                    continue
                seen_crops.add(crop)
                left, top, right, bottom = crop
                crop_img = image[top:bottom, left:right]
                crop_shapes = []
                for shape, shape_assignment, shape_gap in classified:
                    clipped = clip_shape_to_crop(shape, crop, image_w, image_h, 0.15)
                    if clipped is None:
                        continue
                    contacts_for_shape = contact_map[id(shape)]
                    is_edge_defect = bool(contacts_for_shape)
                    flag_assignment, flag_gap = contacts_for_shape[0] if contacts_for_shape else (shape_assignment, shape_gap)
                    crop_shapes.append(add_assignment_flags(clipped, flag_assignment, flag_gap, is_edge_defect))
                    if len(contacts_for_shape) >= 2:
                        crop_shapes[-1]["flags"]["edge_policy_corner_defect"] = True
                if not crop_shapes:
                    continue
                stem = f"{image_path.stem}_corner{idx:03d}_x{left}_y{top}"
                crop_name = f"{stem}.bmp"
                cv2.imwrite(str(corner_img_dir / crop_name), crop_img)
                flags = {
                    "source_image": image_path.name,
                    "source_json": json_path.name,
                    "dataset_category": "edge_corner_defect",
                    "auto_pass_defective": True,
                    "crop_left": left,
                    "crop_top": top,
                    "crop_right": right,
                    "crop_bottom": bottom,
                    "edge_gap_threshold_px": args.edge_gap_px,
                    "json_edge_max_distance_px": args.json_edge_max_distance,
                }
                write_labelme(corner_label_dir / f"{stem}.json", make_labelme(data, crop_name, crop_img.shape[1], crop_img.shape[0], crop_shapes, flags))
                if args.write_review:
                    local_vertical = EdgeAssignment(
                        edge=type(vertical.edge)("vertical", vertical.edge.position - left, vertical.edge.strength),
                        distance=vertical.distance,
                        side=vertical.side,
                        json_edge_orientation=vertical.json_edge_orientation,
                        json_edge_distance=vertical.json_edge_distance,
                    )
                    local_horizontal = EdgeAssignment(
                        edge=type(horizontal.edge)("horizontal", horizontal.edge.position - top, horizontal.edge.strength),
                        distance=horizontal.distance,
                        side=horizontal.side,
                        json_edge_orientation=horizontal.json_edge_orientation,
                        json_edge_distance=horizontal.json_edge_distance,
                    )
                    cv2.imwrite(str(corner_review_dir / crop_name), draw_review(crop_img, crop_shapes, local_vertical, local_horizontal))
                corner_crops += 1
                rows.append({"image": image_path.name, "category": "edge_corner_defect", "output": crop_name, "gap_px": ""})

        if edge_defects:
            edge_images += 1
            seen_crops: set[tuple[int, int, int, int]] = set()
            for idx, (anchor, assignment, gap) in enumerate(edge_defects):
                crop = dynamic_edge_strip(
                    anchor,
                    assignment,
                    image_w,
                    image_h,
                    args.pad,
                    args.min_strip_width,
                    args.max_strip_width,
                )
                if crop in seen_crops:
                    continue
                seen_crops.add(crop)
                left, top, right, bottom = crop
                crop_img = image[top:bottom, left:right]
                crop_shapes = []
                for shape, shape_assignment, shape_gap in classified:
                    clipped = clip_shape_to_crop(shape, crop, image_w, image_h, 0.15)
                    if clipped is None:
                        continue
                    contacts_for_shape = contact_map[id(shape)]
                    is_edge_defect = bool(contacts_for_shape)
                    flag_assignment, flag_gap = contacts_for_shape[0] if contacts_for_shape else (shape_assignment, shape_gap)
                    crop_shapes.append(add_assignment_flags(clipped, flag_assignment, flag_gap, is_edge_defect))
                if not crop_shapes:
                    continue

                stem = f"{image_path.stem}_edge{idx:03d}_{assignment.edge.orientation}_x{left}_y{top}"
                crop_name = f"{stem}.bmp"
                cv2.imwrite(str(edge_img_dir / crop_name), crop_img)
                crop_flags = {
                    "source_image": image_path.name,
                    "source_json": json_path.name,
                    "dataset_category": "edge_defect_strip",
                    "auto_pass_defective": True,
                    "crop_left": left,
                    "crop_top": top,
                    "crop_right": right,
                    "crop_bottom": bottom,
                    "anchor_detected_edge_orientation": assignment.edge.orientation,
                    "anchor_detected_edge_position": assignment.edge.position,
                    "anchor_detected_edge_gap_px": round(gap, 3),
                    "edge_gap_threshold_px": args.edge_gap_px,
                    "json_edge_max_distance_px": args.json_edge_max_distance,
                }
                write_labelme(
                    edge_label_dir / f"{stem}.json",
                    make_labelme(data, crop_name, crop_img.shape[1], crop_img.shape[0], crop_shapes, crop_flags),
                )
                if args.write_review:
                    local_assignment = EdgeAssignment(
                        edge=type(assignment.edge)(
                            assignment.edge.orientation,
                            assignment.edge.position - left if assignment.edge.orientation == "vertical" else assignment.edge.position - top,
                            assignment.edge.strength,
                        ),
                        distance=assignment.distance,
                        side=assignment.side,
                        json_edge_orientation=assignment.json_edge_orientation,
                        json_edge_distance=assignment.json_edge_distance,
                    )
                    cv2.imwrite(str(edge_review_dir / crop_name), draw_review(crop_img, crop_shapes, local_assignment))
                edge_strips += 1
                rows.append({"image": image_path.name, "category": "edge_defect_strip", "output": crop_name, "gap_px": round(gap, 3)})
        if not edge_defects and not corner_defects:
            full_images += 1
            out_name = image_path.name
            shutil.copy2(image_path, full_img_dir / out_name)
            full_shapes = [add_assignment_flags(shape, assignment, gap, False) for shape, assignment, gap in classified]
            flags = {
                "source_image": image_path.name,
                "source_json": json_path.name,
                "dataset_category": "full_image_no_edge_defect",
                "auto_pass_defective": False,
                "edge_gap_threshold_px": args.edge_gap_px,
                "json_edge_max_distance_px": args.json_edge_max_distance,
            }
            write_labelme(full_label_dir / f"{image_path.stem}.json", make_labelme(data, out_name, image_w, image_h, full_shapes, flags))
            if args.write_review:
                cv2.imwrite(str(full_review_dir / out_name), draw_review(image, full_shapes))
            rows.append({"image": image_path.name, "category": "full_image_no_edge_defect", "output": out_name, "gap_px": ""})

    summary_path = output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "category", "output", "gap_px"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"source_jsons={len(json_paths)} edge_images={edge_images} corner_images={corner_images} "
        f"full_images={full_images} edge_strips={edge_strips} corner_crops={corner_crops} "
        f"skipped={skipped} malformed={malformed}"
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
