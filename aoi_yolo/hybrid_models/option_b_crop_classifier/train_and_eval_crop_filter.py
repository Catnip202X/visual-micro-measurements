import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def parse_rename(items):
    rename = {}
    for item in items:
        old, new = item.split("=", 1)
        rename[old.strip()] = new.strip()
    return rename


def read_classes(dataset_dir):
    with open(Path(dataset_dir) / "classes.txt", "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def split_stems(dataset_dir, split):
    image_dir = Path(dataset_dir) / "images" / split
    stems = []
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            stems.append(path.stem)
    return stems


def find_image(src_dir, stem):
    for ext in IMAGE_EXTENSIONS:
        path = Path(src_dir) / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def polygon_mask(points, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def mask_iou(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def polygon_bounds(points):
    pts = np.array(points, dtype=np.float32)
    return (
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    )


def expanded_square(bounds, width, height, margin):
    x1, y1, x2, y2 = bounds
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * margin
    side = max(side, 32)
    x1 = int(max(0, round(cx - side / 2)))
    y1 = int(max(0, round(cy - side / 2)))
    x2 = int(min(width, round(cx + side / 2)))
    y2 = int(min(height, round(cy + side / 2)))
    return x1, y1, x2, y2


def load_ground_truth(src_dir, stem, classes, rename):
    data = load_json(Path(src_dir) / f"{stem}.json")
    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    gts = []
    for shape in data.get("shapes") or []:
        label = str(shape.get("label") or "").strip()
        label = rename.get(label, label)
        points = shape.get("points") or []
        if label not in classes or len(points) < 3:
            continue
        gts.append({"label": label, "mask": polygon_mask(points, width, height), "matched": False})
    return width, height, gts


def generate_candidates(args, split, model, classes, rename):
    out_dir = Path(args.out) / "crops" / split
    true_dir = out_dir / "true"
    false_dir = out_dir / "false"
    true_dir.mkdir(parents=True, exist_ok=True)
    false_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for stem in split_stems(args.dataset, split):
        image_path = find_image(args.src, stem)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        height, width = image.shape[:2]
        gt_width, gt_height, gts = load_ground_truth(args.src, stem, classes, rename)
        if gt_width != width or gt_height != height:
            continue

        result = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )[0]

        boxes = result.boxes
        masks = result.masks
        if boxes is None or masks is None:
            continue

        for idx, (cls_id, conf, points) in enumerate(
            zip(boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy(), masks.xy)
        ):
            label = result.names[int(cls_id)]
            if label not in classes or len(points) < 3:
                continue

            pred_mask = polygon_mask(points, width, height)
            best_iou = 0.0
            for gt in gts:
                if gt["label"] != label:
                    continue
                best_iou = max(best_iou, mask_iou(gt["mask"], pred_mask))
            target = 1 if best_iou >= args.hit_iou else 0

            crop_box = expanded_square(polygon_bounds(points), width, height, args.crop_margin)
            x1, y1, x2, y2 = crop_box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
            crop_name = f"{stem}_{idx:04d}_{label}_{float(conf):.3f}_{target}.png"
            crop_path = (true_dir if target else false_dir) / crop_name
            cv2.imwrite(str(crop_path), crop)
            rows.append(
                {
                    "image": stem,
                    "crop": str(crop_path),
                    "class": label,
                    "target": target,
                    "yolo_conf": float(conf),
                    "area_px": int(pred_mask.sum()),
                    "best_iou": best_iou,
                    "points": json.dumps(np.asarray(points, dtype=float).tolist()),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    manifest = Path(args.out) / f"{split}_candidates.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


class CandidateDataset(Dataset):
    def __init__(self, rows, train):
        self.rows = rows
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = cv2.imread(row["crop"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image), torch.tensor(float(row["target"]), dtype=torch.float32)


class SmallCandidateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, 1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x).squeeze(1)


def train_classifier(args, train_rows, val_rows):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.classifier_device != "cpu" else "cpu")

    train_loader = DataLoader(CandidateDataset(train_rows, True), batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(CandidateDataset(val_rows, False), batch_size=args.batch, shuffle=False, num_workers=0)

    model = SmallCandidateNet().to(device)
    pos = sum(row["target"] == 1 for row in train_rows)
    neg = max(1, len(train_rows) - pos)
    pos_weight = torch.tensor([neg / max(1, pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_score = -1.0
    best_path = Path(args.out) / "candidate_filter.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()

        metrics = classifier_metrics(model, val_loader, device, threshold=0.5)
        score = metrics["f1"]
        print(
            f"epoch={epoch} val_precision={metrics['precision']:.4f} "
            f"val_recall={metrics['recall']:.4f} val_f1={metrics['f1']:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "crop_size": args.crop_size}, best_path)
    return best_path


def classifier_metrics(model, loader, device, threshold):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, targets in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu()
            preds = probs >= threshold
            targets = targets.bool()
            tp += int((preds & targets).sum())
            fp += int((preds & ~targets).sum())
            fn += int((~preds & targets).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def add_classifier_probs(args, rows, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() and args.classifier_device != "cpu" else "cpu")
    model = SmallCandidateNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    loader = DataLoader(CandidateDataset(rows, False), batch_size=args.batch, shuffle=False, num_workers=0)

    offset = 0
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            for prob in probs:
                rows[offset]["filter_prob"] = float(prob)
                offset += 1


def evaluate_filtered(args, rows, classes, rename, thresholds):
    for threshold in thresholds:
        by_image = defaultdict(list)
        for row in rows:
            if row.get("filter_prob", 0.0) < threshold:
                continue
            by_image[row["image"]].append(row)

        stats = defaultdict(lambda: defaultdict(int))
        for stem in split_stems(args.dataset, "val"):
            width, height, gts = load_ground_truth(args.src, stem, classes, rename)
            preds = []
            for row in by_image.get(stem, []):
                points = json.loads(row["points"])
                preds.append(
                    {
                        "label": row["class"],
                        "mask": polygon_mask(points, width, height),
                        "matched": False,
                    }
                )

            for label in classes:
                stats[label]["gt"] += sum(1 for gt in gts if gt["label"] == label)
                stats[label]["pred"] += sum(1 for pred in preds if pred["label"] == label)

            for gt in gts:
                best = None
                best_iou = 0.0
                for pred in preds:
                    if pred["matched"] or pred["label"] != gt["label"]:
                        continue
                    iou = mask_iou(gt["mask"], pred["mask"])
                    if iou > best_iou:
                        best = pred
                        best_iou = iou
                if best is not None and best_iou >= args.hit_iou:
                    best["matched"] = True
                    stats[gt["label"]]["hit"] += 1
                else:
                    stats[gt["label"]]["miss"] += 1

            for pred in preds:
                if not pred["matched"]:
                    stats[pred["label"]]["fp"] += 1

        totals = defaultdict(int)
        for item in stats.values():
            for key, value in item.items():
                totals[key] += value

        recall = totals["hit"] / totals["gt"] if totals["gt"] else 0.0
        precision = totals["hit"] / totals["pred"] if totals["pred"] else 0.0
        print(
            f"filter_threshold={threshold:.2f},gt={totals['gt']},hits={totals['hit']},"
            f"misses={totals['miss']},preds={totals['pred']},false_pos={totals['fp']},"
            f"recall={recall:.4f},precision={precision:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a crop classifier for YOLO candidates.")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--hit-iou", type=float, default=0.05)
    parser.add_argument("--rename", nargs="*", default=["inpinhole=pinhole"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--classifier-device", default="cuda")
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--crop-margin", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    classes = read_classes(args.dataset)
    rename = parse_rename(args.rename)
    yolo = YOLO(args.weights)

    train_rows = generate_candidates(args, "train", yolo, classes, rename)
    val_rows = generate_candidates(args, "val", yolo, classes, rename)
    print(f"train_candidates={len(train_rows)} val_candidates={len(val_rows)}")

    model_path = train_classifier(args, train_rows, val_rows)
    add_classifier_probs(args, val_rows, model_path)

    prob_csv = Path(args.out) / "val_candidates_with_filter_probs.csv"
    with open(prob_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(val_rows[0].keys()) if val_rows else ["image"])
        writer.writeheader()
        writer.writerows(val_rows)

    evaluate_filtered(args, val_rows, classes, rename, thresholds=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])


if __name__ == "__main__":
    main()
