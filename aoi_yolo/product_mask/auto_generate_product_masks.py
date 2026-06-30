"""Generate first-pass product-face masks and baked-in BMP review annotations.

The source BMP/JSON files are treated as read-only. Outputs are written under
this product_mask sidecar so generated annotations can be reviewed before any
training dataset is built from them.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


MASK_LABEL = "product_face_auto"


@dataclass
class MaskResult:
    polygon: list[list[float]]
    area_ratio: float
    bbox: tuple[int, int, int, int]
    method: str


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def largest_reasonable_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    h, w = mask.shape[:2]
    image_area = h * w
    candidates: list[tuple[int, int]] = []
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        area_ratio = area / image_area
        touches_most_edges = (
            int(x <= 2)
            + int(y <= 2)
            + int(x + bw >= w - 3)
            + int(y + bh >= h - 3)
        ) >= 3
        if 0.03 <= area_ratio <= 0.96 and not touches_most_edges:
            candidates.append((area, idx))

    if not candidates:
        candidates = [(int(stats[idx, cv2.CC_STAT_AREA]), idx) for idx in range(1, num_labels)]

    _, best_idx = max(candidates)
    return np.where(labels == best_idx, 255, 0).astype(np.uint8)


def build_candidate_masks(gray: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Product faces in this dataset are usually mid/bright material against dark
    # voids or fixture regions. Try several thresholds and let geometry choose.
    for pct in (20, 25, 30, 35, 40, 45, 50, 55):
        threshold = int(np.percentile(blur, pct))
        _, mask = cv2.threshold(blur, max(8, threshold), 255, cv2.THRESH_BINARY)
        yield f"percentile_{pct}", mask

    # Otsu catches batches where the contrast split is clean.
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "otsu_binary", otsu

    # Inverted Otsu is useful for darker product faces on bright backgrounds.
    yield "otsu_inverse", cv2.bitwise_not(otsu)


def score_mask(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    area = float(np.count_nonzero(mask))
    area_ratio = area / float(h * w)
    if area_ratio <= 0:
        return -1e9

    ys, xs = np.where(mask > 0)
    x, y, bw, bh = cv2.boundingRect(np.column_stack((xs, ys)))
    bbox_ratio = (bw * bh) / float(h * w)
    extent = area / max(1.0, float(bw * bh))
    center_x = x + bw / 2.0
    center_y = y + bh / 2.0
    centered = 1.0 - min(1.0, abs(center_x - w / 2.0) / (w / 2.0))
    vertical_centered = 1.0 - min(1.0, abs(center_y - h / 2.0) / (h / 2.0))

    score = 0.0
    score += 3.0 * min(area_ratio, 0.82)
    score += 2.0 * min(extent, 1.0)
    score += 1.0 * centered + 0.5 * vertical_centered
    if 0.12 <= area_ratio <= 0.90:
        score += 2.0
    if 0.15 <= bbox_ratio <= 0.96:
        score += 1.0
    if bw < w * 0.18 or bh < h * 0.18:
        score -= 4.0
    return score


def mean_band(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    h, w = gray.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return -1.0
    return float(np.mean(gray[y1:y2, x1:x2]))


def edge_mask_from_annotation(annotation_path: Path, image: np.ndarray) -> tuple[np.ndarray, str] | None:
    if not annotation_path.exists():
        return None
    try:
        data = read_json(annotation_path)
    except (OSError, json.JSONDecodeError):
        return None

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    horizontal_masks: list[tuple[np.ndarray, str]] = []
    vertical_masks: list[tuple[np.ndarray, str]] = []
    horizontal_edges: list[tuple[int, int]] = []
    vertical_edges: list[tuple[int, int]] = []

    for shape in data.get("shapes") or []:
        if str(shape.get("label") or "").strip() != "edge":
            continue
        points = np.array(shape.get("points") or [], dtype=np.float32)
        if len(points) < 2:
            continue
        xs = points[:, 0]
        ys = points[:, 1]
        x1 = int(np.floor(np.min(xs)))
        x2 = int(np.ceil(np.max(xs)))
        y1 = int(np.floor(np.min(ys)))
        y2 = int(np.ceil(np.max(ys)))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mask = np.zeros((h, w), dtype=np.uint8)

        is_horizontal_edge = bw >= w * 0.25 and bh <= h * 0.20 and bw >= bh * 3
        is_vertical_edge = bh >= h * 0.25 and bw <= w * 0.20 and bh >= bw * 3

        if is_horizontal_edge:
            horizontal_edges.append((max(0, y1), min(h, y2)))
            gap = max(12, min(80, bh // 2 + 12))
            band = max(48, min(220, h // 10))
            above = mean_band(gray, x1, y1 - gap - band, x2, y1 - gap)
            below = mean_band(gray, x1, y2 + gap, x2, y2 + gap + band)
            choose_down = below >= above
            if h - y2 < h * 0.05 and y1 > h * 0.05:
                choose_down = False
            elif y1 < h * 0.05 and h - y2 > h * 0.05:
                choose_down = True
            if choose_down:
                mask[max(0, y2) : h, :] = 255
                side = "down"
            else:
                mask[: max(0, y1), :] = 255
                side = "up"
            area_ratio = float(np.count_nonzero(mask)) / float(h * w)
            if 0.03 <= area_ratio <= 0.985:
                horizontal_masks.append((mask, f"edge_json_horizontal_{side}"))
        elif is_vertical_edge:
            vertical_edges.append((max(0, x1), min(w, x2)))
            gap = max(12, min(80, bw // 2 + 12))
            band = max(48, min(220, w // 10))
            left = mean_band(gray, x1 - gap - band, y1, x1 - gap, y2)
            right = mean_band(gray, x2 + gap, y1, x2 + gap + band, y2)
            choose_right = right >= left
            if w - x2 < w * 0.05 and x1 > w * 0.05:
                choose_right = False
            elif x1 < w * 0.05 and w - x2 > w * 0.05:
                choose_right = True
            if choose_right:
                mask[:, max(0, x2) : w] = 255
                side = "right"
            else:
                mask[:, : max(0, x1)] = 255
                side = "left"
            area_ratio = float(np.count_nonzero(mask)) / float(h * w)
            if 0.03 <= area_ratio <= 0.985:
                vertical_masks.append((mask, f"edge_json_vertical_{side}"))

    if len(horizontal_edges) >= 2:
        edges = sorted(horizontal_edges, key=lambda item: (item[0] + item[1]) / 2.0)
        best_gap: tuple[int, int] | None = None
        for upper, lower in zip(edges, edges[1:]):
            start = upper[1]
            end = lower[0]
            if end > start and (best_gap is None or end - start > best_gap[1] - best_gap[0]):
                best_gap = (start, end)
        if best_gap is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[best_gap[0] : best_gap[1], :] = 255
            horizontal_masks = [(mask, "edge_json_horizontal_between")]

    if len(vertical_edges) >= 2:
        edges = sorted(vertical_edges, key=lambda item: (item[0] + item[1]) / 2.0)
        best_gap = None
        for left_edge, right_edge in zip(edges, edges[1:]):
            start = left_edge[1]
            end = right_edge[0]
            if end > start and (best_gap is None or end - start > best_gap[1] - best_gap[0]):
                best_gap = (start, end)
        if best_gap is not None:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:, best_gap[0] : best_gap[1]] = 255
            vertical_masks = [(mask, "edge_json_vertical_between")]

    if not horizontal_masks and not vertical_masks:
        return None

    if horizontal_masks:
        horizontal = np.bitwise_or.reduce([item[0] for item in horizontal_masks])
    else:
        horizontal = np.full((h, w), 255, dtype=np.uint8)
    if vertical_masks:
        vertical = np.bitwise_or.reduce([item[0] for item in vertical_masks])
    else:
        vertical = np.full((h, w), 255, dtype=np.uint8)

    mask = cv2.bitwise_and(horizontal, vertical)
    if np.count_nonzero(mask) == 0:
        mask = cv2.bitwise_or(horizontal, vertical)
    method_parts = [horizontal_masks[0][1] if horizontal_masks else "", vertical_masks[0][1] if vertical_masks else ""]
    method = "+".join(part for part in method_parts if part)
    return mask, method


def clean_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    close_k = max(9, (min(h, w) // 80) | 1)
    open_k = max(5, (min(h, w) // 220) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cleaned = largest_reasonable_component(cleaned)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return cleaned


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(2.0, perimeter * 0.006), True)
    points = approx.reshape(-1, 2)
    if len(points) < 3:
        x, y, w, h = cv2.boundingRect(contour)
        points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return [[float(x), float(y)] for x, y in points]


def generate_mask(image: np.ndarray, max_dim: int, annotation_path: Path | None = None) -> tuple[np.ndarray, MaskResult]:
    original_h, original_w = image.shape[:2]
    if annotation_path is not None:
        edge_result = edge_mask_from_annotation(annotation_path, image)
        if edge_result is not None:
            edge_mask, method = edge_result
            polygon = mask_to_polygon(edge_mask)
            ys, xs = np.where(edge_mask > 0)
            x, y, bw, bh = cv2.boundingRect(np.column_stack((xs, ys)))
            return edge_mask, MaskResult(
                polygon=polygon,
                area_ratio=float(np.count_nonzero(edge_mask)) / float(original_h * original_w),
                bbox=(int(x), int(y), int(bw), int(bh)),
                method=method,
            )

    scale = 1.0
    working = image
    if max_dim > 0:
        longest = max(original_h, original_w)
        if longest > max_dim:
            scale = max_dim / float(longest)
            working = cv2.resize(
                image,
                (max(1, int(round(original_w * scale))), max(1, int(round(original_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    best: tuple[float, str, np.ndarray] | None = None
    for method, raw_mask in build_candidate_masks(gray):
        cleaned = clean_mask(raw_mask)
        score = score_mask(cleaned)
        if best is None or score > best[0]:
            best = (score, method, cleaned)

    assert best is not None
    _, method, mask = best
    if mask.shape[:2] != (original_h, original_w):
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        mask = clean_mask(mask)

    polygon = mask_to_polygon(mask)
    if not polygon:
        h, w = original_h, original_w
        margin_x = max(8, int(w * 0.04))
        margin_y = max(8, int(h * 0.04))
        polygon = [
            [float(margin_x), float(margin_y)],
            [float(w - margin_x), float(margin_y)],
            [float(w - margin_x), float(h - margin_y)],
            [float(margin_x), float(h - margin_y)],
        ]
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
        method = "empty_fallback"

    ys, xs = np.where(mask > 0)
    x, y, bw, bh = cv2.boundingRect(np.column_stack((xs, ys)))
    result = MaskResult(
        polygon=polygon,
        area_ratio=float(np.count_nonzero(mask)) / float(mask.shape[0] * mask.shape[1]),
        bbox=(int(x), int(y), int(bw), int(bh)),
        method=method,
    )
    return mask, result


def overlay_review(image: np.ndarray, mask: np.ndarray, result: MaskResult) -> np.ndarray:
    review = image.copy()
    green = np.zeros_like(review)
    green[:, :] = (0, 210, 80)
    masked = mask > 0
    review[masked] = cv2.addWeighted(review, 0.72, green, 0.28, 0)[masked]

    contour_points = np.array(result.polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(review, [contour_points], True, (0, 0, 255), 4, cv2.LINE_AA)
    x, y, w, h = result.bbox
    cv2.rectangle(review, (x, y), (x + w, y + h), (255, 200, 0), 2, cv2.LINE_AA)

    label = f"{MASK_LABEL} {result.area_ratio:.1%} {result.method}"
    cv2.rectangle(review, (18, 18), (18 + 720, 72), (0, 0, 0), -1)
    cv2.putText(
        review,
        label,
        (30, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return review


def make_labelme_annotation(image_path: Path, image: np.ndarray, result: MaskResult) -> dict:
    h, w = image.shape[:2]
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": MASK_LABEL,
                "points": result.polygon,
                "group_id": None,
                "description": "Auto-generated product-face pseudo-label for review",
                "shape_type": "polygon",
                "flags": {"auto_generated": True, "method": result.method},
            }
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def summarize_existing_outputs(bmp_path: Path, mask_path: Path, annotation_path: Path) -> dict | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    data = read_json(annotation_path)
    shape = (data.get("shapes") or [{}])[0]
    flags = shape.get("flags") or {}
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x, y, bw, bh = cv2.boundingRect(np.column_stack((xs, ys)))
    area_ratio = float(np.count_nonzero(mask)) / float(mask.shape[0] * mask.shape[1])
    return {
        "image": bmp_path.name,
        "status": "ok_existing",
        "method": flags.get("method", "unknown"),
        "area_ratio": f"{area_ratio:.6f}",
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(bw),
        "bbox_h": int(bh),
        "points": len(shape.get("points") or []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../../test images"))
    parser.add_argument("--output", type=Path, default=Path("dataset_product_mask"))
    parser.add_argument("--limit", type=int, default=0, help="Optional small dry-run limit.")
    parser.add_argument(
        "--max-dim",
        type=int,
        default=768,
        help="Run mask selection at this maximum image dimension, then upscale to full BMP size.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip images with existing mask/review/annotation outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    masks_dir = output / "masks"
    review_dir = output / "review_bmp"
    annotations_dir = output / "annotations"
    for directory in (masks_dir, review_dir, annotations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bmp_paths = sorted(source.glob("*.bmp"))
    if args.limit:
        bmp_paths = bmp_paths[: args.limit]

    rows = []
    for idx, bmp_path in enumerate(bmp_paths, 1):
        mask_path = masks_dir / f"{bmp_path.stem}_mask.png"
        review_path = review_dir / bmp_path.name
        annotation_path = annotations_dir / f"{bmp_path.stem}.json"
        if args.resume and mask_path.exists() and review_path.exists() and annotation_path.exists():
            existing_row = summarize_existing_outputs(bmp_path, mask_path, annotation_path)
            if existing_row is not None:
                rows.append(existing_row)
                if idx % 50 == 0:
                    print(f"processed {idx}/{len(bmp_paths)}", flush=True)
                continue

        image = cv2.imread(str(bmp_path), cv2.IMREAD_COLOR)
        if image is None:
            rows.append({"image": bmp_path.name, "status": "read_failed"})
            continue

        source_annotation_path = source / f"{bmp_path.stem}.json"
        mask, result = generate_mask(image, args.max_dim, source_annotation_path)
        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(review_path), overlay_review(image, mask, result))
        write_json(annotation_path, make_labelme_annotation(bmp_path, image, result))

        rows.append(
            {
                "image": bmp_path.name,
                "status": "ok",
                "method": result.method,
                "area_ratio": f"{result.area_ratio:.6f}",
                "bbox_x": result.bbox[0],
                "bbox_y": result.bbox[1],
                "bbox_w": result.bbox[2],
                "bbox_h": result.bbox[3],
                "points": len(result.polygon),
            }
        )
        if idx % 50 == 0:
            print(f"processed {idx}/{len(bmp_paths)}", flush=True)

    summary_path = output / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image", "status", "method", "area_ratio", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "points"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
