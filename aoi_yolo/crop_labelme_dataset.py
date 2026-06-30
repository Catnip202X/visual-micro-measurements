"""Create compact LabelMe crop datasets from full-size BMP/JSON pairs.

The transformer keeps useful defect content while reducing storage:
- crops BMP images around labeled defects,
- shifts polygon points into crop coordinates,
- drops LabelMe's base64 ``imageData`` field,
- writes a fresh sidecar dataset without modifying the original files.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_EXCLUDE_LABELS = {"edge"}


@dataclass(frozen=True)
class DetectedEdge:
    orientation: str
    position: int
    strength: float


@dataclass(frozen=True)
class EdgeAssignment:
    edge: DetectedEdge
    distance: float
    side: str
    json_edge_orientation: str
    json_edge_distance: float | None


def read_labelme(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_labelme(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def shape_bounds(shape: dict[str, Any]) -> tuple[float, float, float, float] | None:
    points = shape.get("points") or []
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def crop_window_for_shape(
    shape: dict[str, Any],
    image_w: int,
    image_h: int,
    crop_size: int,
    pad: int,
) -> tuple[int, int, int, int]:
    bounds = shape_bounds(shape)
    if bounds is None:
        return 0, 0, min(crop_size, image_w), min(crop_size, image_h)

    x1, y1, x2, y2 = bounds
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    side = int(math.ceil(max(float(crop_size), box_w + 2 * pad, box_h + 2 * pad)))
    side = min(side, max(image_w, image_h))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_w:
        left -= right - image_w
        right = image_w
    if bottom > image_h:
        top -= bottom - image_h
        bottom = image_h

    left = max(0, left)
    top = max(0, top)
    return left, top, right, bottom


def strip_window_for_shape(
    shape: dict[str, Any],
    image_w: int,
    image_h: int,
    strip_width: int,
    orientation: str,
) -> tuple[int, int, int, int]:
    bounds = shape_bounds(shape)
    if bounds is None:
        bounds = (image_w / 2.0, image_h / 2.0, image_w / 2.0, image_h / 2.0)
    x1, y1, x2, y2 = bounds

    if orientation == "horizontal":
        band_h = min(strip_width, image_h)
        cy = (y1 + y2) / 2.0
        top = int(round(cy - band_h / 2.0))
        bottom = top + band_h
        if top < 0:
            bottom -= top
            top = 0
        if bottom > image_h:
            top -= bottom - image_h
            bottom = image_h
        return 0, max(0, top), image_w, bottom

    band_w = min(strip_width, image_w)
    cx = (x1 + x2) / 2.0
    left = int(round(cx - band_w / 2.0))
    right = left + band_w
    if left < 0:
        right -= left
        left = 0
    if right > image_w:
        left -= right - image_w
        right = image_w
    return max(0, left), 0, right, image_h


def center_of_shape(shape: dict[str, Any]) -> tuple[float, float]:
    bounds = shape_bounds(shape)
    if bounds is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = bounds
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def smooth_projection(values: np.ndarray, window: int) -> np.ndarray:
    window = max(5, int(window) | 1)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def local_projection_peaks(values: np.ndarray, min_distance: int, max_count: int) -> list[tuple[int, float]]:
    if values.size == 0:
        return []
    threshold = float(values.mean() + values.std() * 0.7)
    order = np.argsort(values)[::-1]
    peaks: list[tuple[int, float]] = []
    for raw_idx in order:
        idx = int(raw_idx)
        strength = float(values[idx])
        if strength < threshold and peaks:
            break
        if any(abs(idx - existing_idx) < min_distance for existing_idx, _ in peaks):
            continue
        peaks.append((idx, strength))
        if len(peaks) >= max_count:
            break
    return sorted(peaks)


def detect_object_edges(image: np.ndarray) -> list[DetectedEdge]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 40, 120)
    vertical_projection = smooth_projection(edges.sum(axis=0), max(11, w // 120))
    horizontal_projection = smooth_projection(edges.sum(axis=1), max(11, h // 120))

    candidates: list[DetectedEdge] = []
    for x, strength in local_projection_peaks(vertical_projection, max(30, w // 18), 8):
        candidates.append(DetectedEdge("vertical", x, strength))
    for y, strength in local_projection_peaks(horizontal_projection, max(30, h // 18), 8):
        candidates.append(DetectedEdge("horizontal", y, strength))

    # Line segments help when a product edge is crisp but only covers part of the frame.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=120, minLineLength=min(h, w) // 4, maxLineGap=30)
    if lines is not None:
        for line in lines[:80]:
            x1, y1, x2, y2 = [int(v) for v in line[0]]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = float(math.hypot(dx, dy))
            if length < min(h, w) * 0.22:
                continue
            if dy > dx * 4:
                candidates.append(DetectedEdge("vertical", int(round((x1 + x2) / 2.0)), length * 255.0))
            elif dx > dy * 4:
                candidates.append(DetectedEdge("horizontal", int(round((y1 + y2) / 2.0)), length * 255.0))

    if not candidates:
        candidates = [
            DetectedEdge("vertical", 0, 1.0),
            DetectedEdge("vertical", w - 1, 1.0),
            DetectedEdge("horizontal", 0, 1.0),
            DetectedEdge("horizontal", h - 1, 1.0),
        ]

    merged: list[DetectedEdge] = []
    for orientation in ("vertical", "horizontal"):
        same = sorted([edge for edge in candidates if edge.orientation == orientation], key=lambda e: e.position)
        cluster: list[DetectedEdge] = []
        merge_distance = max(16, (w if orientation == "vertical" else h) // 80)
        for edge in same:
            if not cluster or abs(edge.position - cluster[-1].position) <= merge_distance:
                cluster.append(edge)
                continue
            merged.append(_merge_edge_cluster(cluster))
            cluster = [edge]
        if cluster:
            merged.append(_merge_edge_cluster(cluster))
    return merged


def _merge_edge_cluster(cluster: list[DetectedEdge]) -> DetectedEdge:
    strength_sum = sum(max(1.0, edge.strength) for edge in cluster)
    position = int(round(sum(edge.position * max(1.0, edge.strength) for edge in cluster) / strength_sum))
    return DetectedEdge(cluster[0].orientation, position, max(edge.strength for edge in cluster))


def json_edge_orientation(shape: dict[str, Any]) -> str:
    bounds = shape_bounds(shape)
    if bounds is None:
        return "unknown"
    x1, y1, x2, y2 = bounds
    return "vertical" if (y2 - y1) >= (x2 - x1) else "horizontal"


def nearest_json_edge(defect: dict[str, Any], source_shapes: list[dict[str, Any]]) -> tuple[str, float | None]:
    cx, cy = center_of_shape(defect)
    best: tuple[str, float] | None = None
    for shape in source_shapes:
        if str(shape.get("label", "")) != "edge":
            continue
        orientation = json_edge_orientation(shape)
        bounds = shape_bounds(shape)
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        ex = (x1 + x2) / 2.0
        ey = (y1 + y2) / 2.0
        distance = abs(cx - ex) if orientation == "vertical" else abs(cy - ey)
        if best is None or distance < best[1]:
            best = (orientation, float(distance))
    if best is None:
        return "none", None
    return best


def assign_detected_edge(
    defect: dict[str, Any],
    detected_edges: list[DetectedEdge],
    source_shapes: list[dict[str, Any]],
) -> EdgeAssignment:
    cx, cy = center_of_shape(defect)
    best: tuple[DetectedEdge, float] | None = None
    for edge in detected_edges:
        distance = abs(cx - edge.position) if edge.orientation == "vertical" else abs(cy - edge.position)
        weighted_distance = distance / max(1.0, math.log(edge.strength + 2.0))
        if best is None or weighted_distance < best[1]:
            best = (edge, weighted_distance)
    if best is None:
        best = (DetectedEdge("vertical", int(round(cx)), 1.0), 0.0)

    edge = best[0]
    distance = abs(cx - edge.position) if edge.orientation == "vertical" else abs(cy - edge.position)
    if edge.orientation == "vertical":
        side = "right_of_edge" if cx >= edge.position else "left_of_edge"
    else:
        side = "below_edge" if cy >= edge.position else "above_edge"
    ref_orientation, ref_distance = nearest_json_edge(defect, source_shapes)
    return EdgeAssignment(edge, float(distance), side, ref_orientation, ref_distance)


def edge_strip_window(
    assignment: EdgeAssignment,
    shape: dict[str, Any],
    image_w: int,
    image_h: int,
    strip_width: int,
    pad: int,
) -> tuple[int, int, int, int]:
    bounds = shape_bounds(shape)
    if bounds is None:
        return 0, 0, min(strip_width, image_w), image_h
    x1, y1, x2, y2 = bounds
    edge = assignment.edge
    if edge.orientation == "vertical":
        if assignment.side == "right_of_edge":
            left = max(0, min(edge.position - pad, int(math.floor(x1 - pad))))
            right = min(image_w, left + strip_width)
            if right < x2 + pad:
                right = min(image_w, int(math.ceil(x2 + pad)))
                left = max(0, right - strip_width)
        else:
            right = min(image_w, max(edge.position + pad, int(math.ceil(x2 + pad))))
            left = max(0, right - strip_width)
            if left > x1 - pad:
                left = max(0, int(math.floor(x1 - pad)))
                right = min(image_w, left + strip_width)
        left, right = normalize_axis_window(int(left), int(right), image_w, strip_width)
        return left, 0, right, image_h

    if assignment.side == "below_edge":
        top = max(0, min(edge.position - pad, int(math.floor(y1 - pad))))
        bottom = min(image_h, top + strip_width)
        if bottom < y2 + pad:
            bottom = min(image_h, int(math.ceil(y2 + pad)))
            top = max(0, bottom - strip_width)
    else:
        bottom = min(image_h, max(edge.position + pad, int(math.ceil(y2 + pad))))
        top = max(0, bottom - strip_width)
        if top > y1 - pad:
            top = max(0, int(math.floor(y1 - pad)))
            bottom = min(image_h, top + strip_width)
    top, bottom = normalize_axis_window(int(top), int(bottom), image_h, strip_width)
    return 0, top, image_w, bottom


def normalize_axis_window(start: int, end: int, axis_length: int, target_width: int) -> tuple[int, int]:
    target_width = min(target_width, axis_length)
    start = max(0, min(start, axis_length))
    end = max(start, min(end, axis_length))
    current = end - start
    if current >= target_width:
        return start, end
    missing = target_width - current
    grow_before = min(start, missing // 2)
    start -= grow_before
    missing -= grow_before
    grow_after = min(axis_length - end, missing)
    end += grow_after
    missing -= grow_after
    start = max(0, start - missing)
    return start, end


def polygon_mask(points: list[list[float]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask


def clip_shape_to_crop(
    shape: dict[str, Any],
    crop: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    min_overlap: float,
) -> dict[str, Any] | None:
    points = shape.get("points") or []
    if len(points) < 3:
        return None

    left, top, right, bottom = crop
    original_mask = polygon_mask(points, image_w, image_h)
    original_area = int(np.count_nonzero(original_mask))
    if original_area <= 0:
        return None

    crop_mask = original_mask[top:bottom, left:right]
    kept_area = int(np.count_nonzero(crop_mask))
    if kept_area / original_area < min_overlap:
        return None

    contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 1.0:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.003), True)
    crop_points = approx.reshape(-1, 2).astype(float).tolist()
    if len(crop_points) < 3:
        return None

    new_shape = dict(shape)
    new_shape["points"] = crop_points
    new_shape["flags"] = dict(shape.get("flags") or {})
    new_shape["flags"]["cropped_from_full_image"] = True
    return new_shape


def selected_shapes(
    shapes: list[dict[str, Any]],
    include_labels: set[str] | None,
    exclude_labels: set[str],
) -> list[dict[str, Any]]:
    out = []
    for shape in shapes:
        label = str(shape.get("label", ""))
        if include_labels is not None and label not in include_labels:
            continue
        if label in exclude_labels:
            continue
        if len(shape.get("points") or []) >= 3:
            out.append(shape)
    return out


def make_crop_record(
    source_json: dict[str, Any],
    crop_name: str,
    crop_w: int,
    crop_h: int,
    crop_shapes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": source_json.get("version", "5.0.1"),
        "flags": dict(source_json.get("flags") or {}),
        "shapes": crop_shapes,
        "imagePath": crop_name,
        "imageData": None,
        "imageHeight": crop_h,
        "imageWidth": crop_w,
    }


def draw_review(image: np.ndarray, shapes: list[dict[str, Any]]) -> np.ndarray:
    review = image.copy()
    colors = {
        "pinhole": (0, 0, 255),
        "inpinhole": (0, 128, 255),
        "scratch": (255, 0, 0),
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
        cv2.rectangle(review, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
        cv2.putText(
            review,
            label,
            (max(2, x), max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../test images"))
    parser.add_argument("--output", type=Path, default=Path("dataset_crops_300"))
    parser.add_argument(
        "--mode",
        choices=["square", "vertical-strip", "horizontal-strip", "auto-edge-strip"],
        default="square",
        help="square makes local crops; strip modes preserve one full image axis.",
    )
    parser.add_argument("--crop-size", type=int, default=300)
    parser.add_argument("--pad", type=int, default=32)
    parser.add_argument("--min-overlap", type=float, default=0.15)
    parser.add_argument("--include-label", action="append", default=None)
    parser.add_argument("--exclude-label", action="append", default=sorted(DEFAULT_EXCLUDE_LABELS))
    parser.add_argument("--copy-empty-backgrounds", action="store_true")
    parser.add_argument("--write-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
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

    include_labels = set(args.include_label) if args.include_label else None
    exclude_labels = set(args.exclude_label or [])

    json_paths = sorted(source.glob("*.json"))
    if args.limit:
        json_paths = json_paths[: args.limit]

    total_crops = 0
    total_shapes = 0
    skipped = 0
    malformed = 0
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
        anchors = selected_shapes(source_shapes, include_labels, exclude_labels)
        detected_edges = detect_object_edges(image) if args.mode == "auto-edge-strip" else []
        if not anchors and not args.copy_empty_backgrounds:
            continue

        if not anchors:
            cx = image_w // 2
            cy = image_h // 2
            half = args.crop_size // 2
            if args.mode == "vertical-strip":
                crop = (max(0, cx - half), 0, min(image_w, cx + half), image_h)
            elif args.mode == "horizontal-strip":
                crop = (0, max(0, cy - half), image_w, min(image_h, cy + half))
            else:
                crop = (max(0, cx - half), max(0, cy - half), min(image_w, cx + half), min(image_h, cy + half))
            anchors = [{"label": "background", "points": [[crop[0], crop[1]], [crop[2], crop[1]], [crop[2], crop[3]], [crop[0], crop[3]]]}]

        seen_crops: set[tuple[int, int, int, int]] = set()
        for anchor_index, anchor in enumerate(anchors):
            assignment: EdgeAssignment | None = None
            if args.mode == "auto-edge-strip":
                assignment = assign_detected_edge(anchor, detected_edges, source_shapes)
                crop = edge_strip_window(assignment, anchor, image_w, image_h, args.crop_size, args.pad)
            elif args.mode == "vertical-strip":
                crop = strip_window_for_shape(anchor, image_w, image_h, args.crop_size, "vertical")
            elif args.mode == "horizontal-strip":
                crop = strip_window_for_shape(anchor, image_w, image_h, args.crop_size, "horizontal")
            else:
                crop = crop_window_for_shape(anchor, image_w, image_h, args.crop_size, args.pad)
            if crop in seen_crops:
                continue
            seen_crops.add(crop)

            left, top, right, bottom = crop
            crop_shapes = []
            for shape in source_shapes:
                label = str(shape.get("label", ""))
                if include_labels is not None and label not in include_labels:
                    continue
                if label in exclude_labels:
                    continue
                clipped = clip_shape_to_crop(shape, crop, image_w, image_h, args.min_overlap)
                if clipped is not None:
                    if args.mode == "auto-edge-strip":
                        shape_assignment = assign_detected_edge(shape, detected_edges, source_shapes)
                        clipped["flags"] = dict(clipped.get("flags") or {})
                        clipped["flags"]["detected_edge_orientation"] = shape_assignment.edge.orientation
                        clipped["flags"]["detected_edge_position"] = shape_assignment.edge.position
                        clipped["flags"]["detected_edge_distance_px"] = round(shape_assignment.distance, 3)
                        clipped["flags"]["detected_edge_side"] = shape_assignment.side
                        clipped["flags"]["json_edge_orientation_nearest"] = shape_assignment.json_edge_orientation
                        clipped["flags"]["json_edge_distance_px"] = (
                            None
                            if shape_assignment.json_edge_distance is None
                            else round(shape_assignment.json_edge_distance, 3)
                        )
                    crop_shapes.append(clipped)

            if not crop_shapes and not args.copy_empty_backgrounds:
                continue

            crop_img = image[top:bottom, left:right]
            crop_name = f"{image_path.stem}_crop{anchor_index:03d}_x{left}_y{top}.bmp"
            cv2.imwrite(str(images_dir / crop_name), crop_img)
            if args.write_review:
                cv2.imwrite(str(review_dir / crop_name), draw_review(crop_img, crop_shapes))
            label_payload = make_crop_record(data, crop_name, crop_img.shape[1], crop_img.shape[0], crop_shapes)
            label_payload["flags"]["source_image"] = image_path.name
            label_payload["flags"]["source_json"] = json_path.name
            label_payload["flags"]["crop_left"] = left
            label_payload["flags"]["crop_top"] = top
            if assignment is not None:
                label_payload["flags"]["crop_mode"] = "auto-edge-strip"
                label_payload["flags"]["anchor_detected_edge_orientation"] = assignment.edge.orientation
                label_payload["flags"]["anchor_detected_edge_position"] = assignment.edge.position
                label_payload["flags"]["anchor_detected_edge_distance_px"] = round(assignment.distance, 3)
                label_payload["flags"]["anchor_detected_edge_side"] = assignment.side
                label_payload["flags"]["anchor_json_edge_orientation_nearest"] = assignment.json_edge_orientation
                label_payload["flags"]["anchor_json_edge_distance_px"] = (
                    None if assignment.json_edge_distance is None else round(assignment.json_edge_distance, 3)
                )
            write_labelme(labels_dir / f"{Path(crop_name).stem}.json", label_payload)
            total_crops += 1
            total_shapes += len(crop_shapes)

    print(
        f"source_jsons={len(json_paths)} crops={total_crops} shapes={total_shapes} "
        f"skipped={skipped} malformed={malformed}"
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
