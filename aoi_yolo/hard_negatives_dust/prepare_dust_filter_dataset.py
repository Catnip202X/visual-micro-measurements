"""Build a sidecar hard-negative dust/benign-speck dataset.

Outputs:
- source_clean_test_images: original BMPs plus LabelMe JSON with imageData removed.
- candidate_filter_dataset: classification crops split into
  real_defect and dust_or_benign.

The original ``test images`` folder is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")
CORE_LABELS = {"pinhole", "inpinhole", "scratch"}
RENAMES = {"inpinhole": "pinhole"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_files:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_clean_backup(source: Path, out: Path, copy_bmps: bool) -> tuple[int, int]:
    out.mkdir(parents=True, exist_ok=True)
    bmp_count = 0
    json_count = 0
    for image_path in sorted(source.iterdir()):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            link_or_copy(image_path, out / image_path.name, copy_bmps)
            bmp_count += 1
    for json_path in sorted(source.glob("*.json")):
        data = read_json(json_path)
        if isinstance(data, dict):
            data["imageData"] = None
        write_json(out / json_path.name, data)
        json_count += 1
    return bmp_count, json_count


def find_image(source: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = source / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def polygon_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    pts = points.astype(np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def bounds_from_points(points: np.ndarray) -> tuple[int, int, int, int]:
    x1 = int(math.floor(float(np.min(points[:, 0]))))
    y1 = int(math.floor(float(np.min(points[:, 1]))))
    x2 = int(math.ceil(float(np.max(points[:, 0]))))
    y2 = int(math.ceil(float(np.max(points[:, 1]))))
    return x1, y1, x2, y2


def padded_square_crop(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
    crop_size: int,
    pad: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bounds
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = max(crop_size, bw + 2 * pad, bh + 2 * pad)
    side = min(side, max(w, h))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    right = left + side
    bottom = top + side
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > w:
        left -= right - w
        right = w
    if bottom > h:
        top -= bottom - h
        bottom = h
    left = max(0, left)
    top = max(0, top)
    crop = image[top:bottom, left:right]
    return crop, (left, top, right, bottom)


def collect_ground_truth(json_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    data = read_json(json_path)
    if not isinstance(data, dict):
        return []
    gts = []
    for shape in data.get("shapes") or []:
        label = str(shape.get("label") or "").strip()
        if label not in CORE_LABELS:
            continue
        points = np.array(shape.get("points") or [], dtype=np.float32)
        if len(points) < 3:
            continue
        label = RENAMES.get(label, label)
        mask = polygon_mask(points, width, height)
        if mask.sum() == 0:
            continue
        gts.append({"label": label, "points": points, "mask": mask, "bounds": bounds_from_points(points)})
    return gts


def save_crop(path: Path, crop: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if crop.size == 0:
        return False
    return bool(cv2.imwrite(str(path), crop))


def mine_dataset(args: argparse.Namespace) -> None:
    source = args.clean_source.resolve()
    out = args.out.resolve()
    crop_root = out / "candidate_filter_dataset"
    rows = []
    model = YOLO(str(args.weights))

    rng = random.Random(args.seed)
    stems = sorted(path.stem for path in source.glob("*.json") if path.name.lower() != "drive_manifest.json")
    if args.limit:
        stems = stems[: args.limit]

    all_positive: list[tuple[str, np.ndarray, tuple[int, int, int, int], str]] = []
    all_negative: list[tuple[str, np.ndarray, tuple[int, int, int, int], str, float, float]] = []

    for idx, stem in enumerate(stems, 1):
        image_path = find_image(source, stem)
        json_path = source / f"{stem}.json"
        if image_path is None or not json_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        gts = collect_ground_truth(json_path, width, height)

        for gt_index, gt in enumerate(gts):
            crop, crop_box = padded_square_crop(image, gt["bounds"], args.crop_size, args.pad)
            all_positive.append((stem, crop, crop_box, gt["label"]))

        result = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )[0]
        if result.boxes is None or result.masks is None:
            continue
        for pred_index, (cls_id, conf, points) in enumerate(
            zip(result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy(), result.masks.xy)
        ):
            label = result.names[int(cls_id)]
            if label not in {"pinhole", "scratch"} or len(points) < 3:
                continue
            pts = np.array(points, dtype=np.float32)
            pred_mask = polygon_mask(pts, width, height)
            if pred_mask.sum() < args.min_area:
                continue
            best_iou = max((mask_iou(pred_mask, gt["mask"]) for gt in gts if gt["label"] == label), default=0.0)
            if best_iou >= args.fp_iou:
                continue
            crop, crop_box = padded_square_crop(image, bounds_from_points(pts), args.crop_size, args.pad)
            all_negative.append((stem, crop, crop_box, label, float(conf), best_iou))

        if idx % 50 == 0:
            print(f"processed {idx}/{len(stems)} images")

    rng.shuffle(all_positive)
    rng.shuffle(all_negative)
    if args.max_negatives:
        all_negative = all_negative[: args.max_negatives]
    if args.balance_positives:
        all_positive = all_positive[: max(len(all_negative), 1)]

    records = []
    for class_name, items in [
        ("real_defect", all_positive),
        ("dust_or_benign", all_negative),
    ]:
        for item_index, item in enumerate(items):
            split = "val" if rng.random() < args.val_ratio else "train"
            if class_name == "real_defect":
                stem, crop, crop_box, source_label = item
                conf = ""
                best_iou = ""
            else:
                stem, crop, crop_box, source_label, conf, best_iou = item
            out_name = f"{stem}_{class_name}_{item_index:05d}.bmp"
            out_path = crop_root / split / class_name / out_name
            if not save_crop(out_path, crop):
                continue
            record = {
                "split": split,
                "class": class_name,
                "file": str(out_path.relative_to(out)),
                "source_image": f"{stem}.bmp",
                "source_label": source_label,
                "crop_left": crop_box[0],
                "crop_top": crop_box[1],
                "crop_right": crop_box[2],
                "crop_bottom": crop_box[3],
                "prediction_conf": conf,
                "best_gt_iou": best_iou,
            }
            records.append(record)

    manifest = out / "candidate_filter_dataset" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["split", "class", "file"])
        writer.writeheader()
        writer.writerows(records)

    print(f"positive_crops: {sum(1 for r in records if r['class'] == 'real_defect')}")
    print(f"negative_crops: {sum(1 for r in records if r['class'] == 'dust_or_benign')}")
    print(f"output: {crop_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("../../test images"))
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--clean-source", type=Path, default=Path("source_clean_test_images"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--fp-iou", type=float, default=0.05)
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--pad", type=int, default=48)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-negatives", type=int, default=2500)
    parser.add_argument("--balance-positives", action="store_true")
    parser.add_argument("--copy-bmps", action="store_true", help="Copy BMPs instead of hardlinking them in clean source.")
    parser.add_argument("--skip-backup", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    args.clean_source = (args.out / args.clean_source).resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.skip_backup:
        bmp_count, json_count = make_clean_backup(args.source.resolve(), args.clean_source, args.copy_bmps)
        print(f"clean_backup_bmps: {bmp_count}")
        print(f"clean_backup_jsons: {json_count}")
    mine_dataset(args)


if __name__ == "__main__":
    main()
