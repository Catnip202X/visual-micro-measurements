import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_classes(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def val_stems(dataset_dir):
    image_dir = os.path.join(dataset_dir, "images", "val")
    stems = set()
    for name in os.listdir(image_dir):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            stems.add(os.path.splitext(name)[0])
    return stems


def polygon_mask(points, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def polygon_bounds(points):
    pts = np.array(points, dtype=np.float32)
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    return x_min, y_min, x_max, y_max


def parse_key_values(items, value_type=float):
    values = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected class=value pair, got: {item}")
        key, value = item.split("=", 1)
        values[key.strip()] = value_type(value.strip())
    return values


def parse_rois(items):
    rois = []
    for item in items:
        values = [float(value) for value in item.split(",")]
        if len(values) != 4:
            raise SystemExit(f"--roi expects x1,y1,x2,y2, got: {item}")
        rois.append(tuple(values))
    return rois


def mask_centroid(mask):
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def in_any_roi(mask, width, height, rois):
    if not rois:
        return True
    centroid = mask_centroid(mask)
    if centroid is None:
        return False
    cx, cy = centroid
    nx = cx / width
    ny = cy / height
    return any(x1 <= nx <= x2 and y1 <= ny <= y2 for x1, y1, x2, y2 in rois)


def mask_iou(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def collect_ground_truth(src_dir, stems, classes, rename):
    class_set = set(classes)
    records = {}
    for stem in sorted(stems):
        json_path = os.path.join(src_dir, stem + ".json")
        if not os.path.exists(json_path):
            continue
        data = load_json(json_path)
        width = int(data["imageWidth"])
        height = int(data["imageHeight"])
        gts = []
        ignored_gts = []
        edge_mask = np.zeros((height, width), dtype=np.uint8)
        for shape in data.get("shapes") or []:
            label = str(shape.get("label") or "").strip()
            label = rename.get(label, label)
            points = shape.get("points") or []
            if len(points) < 3:
                continue
            if label == "edge":
                pts = np.array(points, dtype=np.float32)
                pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
                cv2.polylines(edge_mask, [pts.astype(np.int32)], True, 1, 3)
            item = {
                "label": label,
                "mask": polygon_mask(points, width, height),
                "matched": False,
                "best_iou": 0.0,
            }
            if label not in class_set:
                ignored_gts.append(item)
                continue
            gts.append(
                item
            )
        records[stem] = {
            "width": width,
            "height": height,
            "gts": gts,
            "ignored_gts": ignored_gts,
            "edge_mask": edge_mask,
        }
    return records


def find_image(src_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(src_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def evaluate(args):
    classes = read_classes(os.path.join(args.dataset, "classes.txt"))
    rename = parse_rename(args.rename)
    class_conf = parse_key_values(args.class_conf)
    class_area_min = parse_key_values(args.class_area_min)
    class_area_max = parse_key_values(args.class_area_max)
    rois = parse_rois(args.roi)
    stems = val_stems(args.dataset)
    records = collect_ground_truth(args.src, stems, classes, rename)
    model = YOLO(args.weights)

    stats = defaultdict(lambda: defaultdict(int))
    rows = []
    pred_rows = []

    for stem in sorted(records):
        image_path = find_image(args.src, stem)
        if not image_path:
            continue

        record = records[stem]
        width = record["width"]
        height = record["height"]
        preds = []
        result = model.predict(
            image_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )[0]

        boxes = result.boxes
        masks = result.masks
        if boxes is not None and masks is not None:
            for cls_id, conf, points in zip(
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy(),
                masks.xy,
            ):
                label = result.names[int(cls_id)]
                if label not in classes or len(points) < 3:
                    continue
                mask = polygon_mask(points, width, height)
                area = int(mask.sum())
                if float(conf) < class_conf.get(label, args.conf):
                    continue
                if area < class_area_min.get(label, 0):
                    continue
                if area > class_area_max.get(label, float("inf")):
                    continue
                if not in_any_roi(mask, width, height, rois):
                    continue
                if args.edge_exclusion_px > 0:
                    edge_near = cv2.dilate(
                        record["edge_mask"],
                        np.ones((args.edge_exclusion_px * 2 + 1, args.edge_exclusion_px * 2 + 1), dtype=np.uint8),
                    )
                    if np.logical_and(mask, edge_near).any():
                        continue
                preds.append(
                    {
                        "label": label,
                        "conf": float(conf),
                        "mask": mask,
                        "points": points,
                        "matched": False,
                        "best_iou": 0.0,
                    }
                )

        for label in classes:
            stats[label]["gt"] += sum(1 for gt in record["gts"] if gt["label"] == label)
            stats[label]["pred"] += sum(1 for pred in preds if pred["label"] == label)

        for gt in record["gts"]:
            best = None
            best_iou = 0.0
            for pred in preds:
                if pred["matched"]:
                    continue
                if not args.class_agnostic_match and pred["label"] != gt["label"]:
                    continue
                iou = mask_iou(gt["mask"], pred["mask"])
                pred["best_iou"] = max(pred["best_iou"], iou)
                if iou > best_iou:
                    best = pred
                    best_iou = iou
            gt["best_iou"] = best_iou
            if best is not None and best_iou >= args.hit_iou:
                gt["matched"] = True
                best["matched"] = True
                stats[gt["label"]]["hit"] += 1
            else:
                stats[gt["label"]]["miss"] += 1
                rows.append([stem, gt["label"], "miss", f"{best_iou:.4f}", ""])

        for pred in preds:
            best_ignored_label = ""
            best_ignored_iou = 0.0
            if args.pred_csv:
                for ignored_gt in record["ignored_gts"]:
                    iou = mask_iou(ignored_gt["mask"], pred["mask"])
                    if iou > best_ignored_iou:
                        best_ignored_iou = iou
                        best_ignored_label = ignored_gt["label"]
            if not pred["matched"]:
                stats[pred["label"]]["fp"] += 1
                rows.append([stem, pred["label"], "false_positive", "", f"{pred['conf']:.4f}"])
            if args.pred_csv:
                x_min, y_min, x_max, y_max = polygon_bounds(pred["points"])
                pred_rows.append(
                    [
                        stem,
                        pred["label"],
                        "hit" if pred["matched"] else "false_positive",
                        f"{pred['conf']:.6f}",
                        int(pred["mask"].sum()),
                        f"{x_max - x_min:.2f}",
                        f"{y_max - y_min:.2f}",
                        f"{pred['best_iou']:.6f}",
                        best_ignored_label,
                        f"{best_ignored_iou:.6f}",
                    ]
                )

    print(f"weights: {args.weights}")
    print(f"dataset: {args.dataset}")
    print(f"val_images: {len(records)}")
    print(f"hit_iou: {args.hit_iou}")
    print()
    print("class,gt,hits,misses,preds,false_pos,recall,precision")

    total = defaultdict(int)
    for label in classes:
        item = stats[label]
        recall = item["hit"] / item["gt"] if item["gt"] else 0.0
        precision = item["hit"] / item["pred"] if item["pred"] else 0.0
        print(
            f"{label},{item['gt']},{item['hit']},{item['miss']},{item['pred']},"
            f"{item['fp']},{recall:.4f},{precision:.4f}"
        )
        for key, value in item.items():
            total[key] += value

    recall = total["hit"] / total["gt"] if total["gt"] else 0.0
    precision = total["hit"] / total["pred"] if total["pred"] else 0.0
    print(
        f"ALL,{total['gt']},{total['hit']},{total['miss']},{total['pred']},"
        f"{total['fp']},{recall:.4f},{precision:.4f}"
    )

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "class", "event", "best_iou", "confidence"])
            writer.writerows(rows)

    if args.pred_csv:
        with open(args.pred_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "image",
                    "class",
                    "event",
                    "confidence",
                    "area_px",
                    "bbox_w",
                    "bbox_h",
                    "best_iou",
                    "best_ignored_label",
                    "best_ignored_iou",
                ]
            )
            writer.writerows(pred_rows)


def parse_rename(items):
    rename = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--rename expects old=new pairs, got: {item}")
        old, new = item.split("=", 1)
        rename[old.strip()] = new.strip()
    return rename


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO masks against LabelMe JSON polygons.")
    parser.add_argument("--src", required=True, help="Folder containing source BMP/JSON pairs.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset folder with images/val and classes.txt.")
    parser.add_argument("--weights", required=True, help="YOLO segmentation weights to evaluate.")
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument(
        "--class-conf",
        nargs="*",
        default=[],
        help="Optional per-class confidence thresholds as class=value pairs.",
    )
    parser.add_argument(
        "--class-area-min",
        nargs="*",
        default=[],
        help="Optional per-class minimum mask area in pixels as class=value pairs.",
    )
    parser.add_argument(
        "--class-area-max",
        nargs="*",
        default=[],
        help="Optional per-class maximum mask area in pixels as class=value pairs.",
    )
    parser.add_argument(
        "--roi",
        nargs="*",
        default=[],
        help="Optional normalized rectangular ROIs as x1,y1,x2,y2. Keep predictions whose centroid falls inside any ROI.",
    )
    parser.add_argument(
        "--edge-exclusion-px",
        type=int,
        default=0,
        help="Reject predictions touching an annotated edge after dilating edge labels by this many pixels.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--hit-iou", type=float, default=0.1, help="Mask IoU threshold counted as an AOI hit.")
    parser.add_argument("--device", default="0", help="Inference device.")
    parser.add_argument("--out-csv", default=None, help="Optional CSV path for misses and false positives.")
    parser.add_argument("--pred-csv", default=None, help="Optional CSV path for every prediction and its geometry.")
    parser.add_argument(
        "--rename",
        nargs="*",
        default=[],
        help="Optional ground-truth label remaps as old=new pairs, matching dataset conversion.",
    )
    parser.add_argument(
        "--class-agnostic-match",
        action="store_true",
        help="Count any overlapping core-class prediction as a hit, even if the class name differs.",
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
