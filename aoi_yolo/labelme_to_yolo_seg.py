import argparse
import glob
import json
import os
import random
import shutil


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_image(src_dir, json_path, image_path_value, images_dir=None):
    candidates = []
    if image_path_value:
        if images_dir:
            candidates.append(os.path.join(images_dir, image_path_value))
        candidates.append(os.path.join(src_dir, image_path_value))

    stem = os.path.splitext(os.path.basename(json_path))[0]
    for ext in IMAGE_EXTENSIONS:
        if images_dir:
            candidates.append(os.path.join(images_dir, stem + ext))
        candidates.append(os.path.join(src_dir, stem + ext))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def normalized_polygon(points, width, height):
    values = []
    for point in points:
        if len(point) < 2:
            continue
        x = min(max(float(point[0]) / width, 0.0), 1.0)
        y = min(max(float(point[1]) / height, 0.0), 1.0)
        values.extend([x, y])
    return values


def write_yaml(path, dataset_root, classes):
    names = ", ".join([f"{idx}: {name!r}" for idx, name in enumerate(classes)])
    normalized_root = os.path.abspath(dataset_root).replace("\\", "/")
    text = (
        f"path: {normalized_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {{{names}}}\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe polygons to YOLO segmentation labels.")
    parser.add_argument("--src", required=True, help="Folder containing .bmp/.json pairs.")
    parser.add_argument("--out", required=True, help="Output YOLO dataset folder.")
    parser.add_argument("--images-dir", default=None, help="Optional image folder when JSON labels are stored separately.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic split seed.")
    parser.add_argument("--include", nargs="*", default=None, help="Optional labels to include.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Optional labels to exclude.")
    parser.add_argument(
        "--rename",
        nargs="*",
        default=[],
        help="Optional label remaps as old=new pairs, applied before include/exclude filtering.",
    )
    parser.add_argument(
        "--include-backgrounds",
        action="store_true",
        help="Include image/JSON pairs with no selected shapes as empty-label background images.",
    )
    parser.add_argument("--no-copy-images", action="store_true", help="Write labels only; do not copy source images.")
    args = parser.parse_args()

    include = set(args.include) if args.include else None
    exclude = set(args.exclude or [])
    rename = {}
    for item in args.rename:
        if "=" not in item:
            raise SystemExit(f"--rename expects old=new pairs, got: {item}")
        old, new = item.split("=", 1)
        rename[old.strip()] = new.strip()

    json_paths = [
        path
        for path in sorted(glob.glob(os.path.join(args.src, "*.json")))
        if os.path.basename(path).lower() != "drive_manifest.json"
    ]

    records = []
    class_names = set()
    skipped = []

    for json_path in json_paths:
        try:
            data = load_json(json_path)
        except Exception as exc:
            skipped.append((json_path, f"json error: {exc}"))
            continue

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            skipped.append((json_path, "missing imageWidth/imageHeight"))
            continue

        image_path = find_image(args.src, json_path, data.get("imagePath"), args.images_dir)
        if not image_path:
            skipped.append((json_path, "missing image file"))
            continue

        yolo_shapes = []
        for shape in data.get("shapes") or []:
            label = str(shape.get("label") or "").strip()
            label = rename.get(label, label)
            points = shape.get("points") or []
            if not label or len(points) < 3:
                continue
            if include is not None and label not in include:
                continue
            if label in exclude:
                continue
            poly = normalized_polygon(points, float(width), float(height))
            if len(poly) < 6:
                continue
            yolo_shapes.append((label, poly))
            class_names.add(label)

        if yolo_shapes or args.include_backgrounds:
            records.append((json_path, image_path, yolo_shapes))

    classes = sorted(class_names)
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    random.Random(args.seed).shuffle(records)
    val_count = max(1, int(round(len(records) * args.val_ratio))) if records else 0
    val_set = set(json_path for json_path, _, _ in records[:val_count])

    for split in ("train", "val"):
        ensure_dir(os.path.join(args.out, "images", split))
        ensure_dir(os.path.join(args.out, "labels", split))

    for json_path, image_path, shapes in records:
        split = "val" if json_path in val_set else "train"
        stem = os.path.splitext(os.path.basename(image_path))[0]
        image_ext = os.path.splitext(image_path)[1].lower()
        out_image = os.path.join(args.out, "images", split, stem + image_ext)
        out_label = os.path.join(args.out, "labels", split, stem + ".txt")

        if not args.no_copy_images:
            shutil.copy2(image_path, out_image)

        lines = []
        for label, poly in shapes:
            coords = " ".join(f"{value:.6f}" for value in poly)
            lines.append(f"{class_to_id[label]} {coords}")

        with open(out_label, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            if lines:
                handle.write("\n")

    ensure_dir(args.out)
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as handle:
        for name in classes:
            handle.write(name + "\n")
    write_yaml(os.path.join(args.out, "data.yaml"), args.out, classes)

    print(f"converted_images: {len(records)}")
    print(f"train_images: {len(records) - val_count}")
    print(f"val_images: {val_count}")
    print(f"classes: {classes}")
    print(f"skipped_files: {len(skipped)}")
    for path, reason in skipped[:20]:
        print(f"  skipped {os.path.basename(path)}: {reason}")


if __name__ == "__main__":
    main()
