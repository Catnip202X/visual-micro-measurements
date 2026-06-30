import argparse
import collections
import glob
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Audit LabelMe-style AOI annotations.")
    parser.add_argument("--src", required=True, help="Folder containing .bmp/.json pairs.")
    args = parser.parse_args()

    labels = collections.Counter()
    shape_types = collections.Counter()
    points_by_label = collections.defaultdict(list)
    image_rows = []
    bad = []

    for path in sorted(glob.glob(os.path.join(args.src, "*.json"))):
        if os.path.basename(path).lower() == "drive_manifest.json":
            continue
        try:
            data = load_json(path)
            shapes = data.get("shapes") or []
            image_rows.append(
                (
                    os.path.basename(path),
                    data.get("imagePath"),
                    data.get("imageWidth"),
                    data.get("imageHeight"),
                    len(shapes),
                )
            )
            for shape in shapes:
                label = str(shape.get("label") or "").strip() or "<blank>"
                labels[label] += 1
                shape_types[str(shape.get("shape_type") or "")] += 1
                points_by_label[label].append(len(shape.get("points") or []))
        except Exception as exc:
            bad.append((os.path.basename(path), str(exc)))

    print(f"json_files: {len(image_rows)}")
    print(f"bad_json: {len(bad)}")
    print()
    print("labels:")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")
    print()
    print("shape_types:")
    for shape_type, count in shape_types.most_common():
        print(f"  {shape_type}: {count}")
    print()
    print("point_counts_by_label:")
    for label in sorted(points_by_label):
        values = points_by_label[label]
        avg = sum(values) / len(values)
        print(f"  {label}: min={min(values)} max={max(values)} avg={avg:.1f} n={len(values)}")
    print()
    print("sample_images:")
    for row in image_rows[:10]:
        print(f"  {row}")

    if bad:
        print()
        print("bad_files:")
        for row in bad[:20]:
            print(f"  {row}")


if __name__ == "__main__":
    main()
