# AOI YOLO Pilot

This folder is a sidecar workflow for testing a lightweight YOLO-style inspection model without changing the existing C# machine-control application.

## Why This Exists

The current application already handles the hard industrial-control work:

- camera sequencing
- ring/spot light selection
- tray and QR state
- pixel-to-tray coordinate mapping
- MySQL defect records
- STM32/Modbus control

The safest first step is to replace only the inspection backend offline, then compare the output against the current HALCON pipeline.

## Dataset Shape

The `test images` folder contains paired files:

- `0001.bmp`: source inspection image
- `0001.json`: LabelMe-style polygon annotation metadata

The JSON files include `shapes`, `label`, polygon `points`, `imageWidth`, and `imageHeight`.

Current observed labels include:

- `scratch`
- `pinhole`
- `inpinhole`
- `edge`
- `outswab`
- `inswab`
- `splash`
- `corrosion`
- `fingerprint`
- `chipping`

For a defect detector, consider excluding `edge` if it represents product boundary rather than a defect.

## Audit The Dataset

Use the bundled Python runtime if `python` is not on PATH:

```powershell
& "C:\Users\Catnip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" aoi_yolo\audit_dataset.py --src "test images"
```

## Convert To YOLO Segmentation

Default conversion includes every label:

```powershell
& "C:\Users\Catnip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" aoi_yolo\labelme_to_yolo_seg.py --src "test images" --out "aoi_yolo\dataset"
```

For a defect-only model that excludes product edges:

```powershell
& "C:\Users\Catnip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" aoi_yolo\labelme_to_yolo_seg.py --src "test images" --out "aoi_yolo\dataset_defects" --exclude edge
```

For the current best focused dataset, merge `inpinhole` into `pinhole` and keep `scratch` separate:

```powershell
.\.venv_yolo\Scripts\python.exe aoi_yolo\labelme_to_yolo_seg.py --src "test images" --out "aoi_yolo\dataset_core2" --include pinhole scratch --rename inpinhole=pinhole
```

For a fast dry run that writes labels but does not copy the large BMP files:

```powershell
& "C:\Users\Catnip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" aoi_yolo\labelme_to_yolo_seg.py --src "test images" --out "aoi_yolo\dataset_dryrun" --exclude edge --no-copy-images
```

The output layout is compatible with YOLO segmentation training:

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
  classes.txt
```

## Recommended Model Direction

Use segmentation rather than plain boxes for this AOI task. Scratches and pinholes are small, thin, and measurement-sensitive; masks let us compute area, length, width, centroid, and physical dimensions more reliably.

Start small:

- YOLO segmentation nano/small model
- image size around 1024 to start
- defect-only classes first
- export to ONNX for C# inference after offline validation

## Current Focused Results

Best focused checkpoint so far:

```text
aoi_yolo/runs/core2_yolo11n_seg_e120_i1280_mask2_from_core3/weights/best.pt
```

This model uses two classes:

- `pinhole` with original `inpinhole` merged into it
- `scratch`

COCO-style validation is still below production target, but AOI-style polygon hit/miss evaluation reaches the requested ballpark when tuned for recall:

```powershell
.\.venv_yolo\Scripts\python.exe aoi_yolo\evaluate_json_polygons.py --src "test images" --dataset "aoi_yolo\dataset_core2" --weights "aoi_yolo\runs\core2_yolo11n_seg_e120_i1280_mask2_from_core3\weights\best.pt" --imgsz 1280 --conf 0.03 --hit-iou 0.05 --rename inpinhole=pinhole
```

Current AOI-style result at `conf=0.03`, `hit_iou=0.05`:

```text
pinhole recall: 95.53%, precision: 61.04%
scratch recall: 91.89%, precision: 37.99%
overall recall: 95.08%, precision: 56.85%
```

At `conf=0.01`, overall recall rises to about 97.96%, but precision drops to about 45.87%. This is useful for high-recall screening, not yet production-grade reject logic.

The checkpoint has also been exported to ONNX:

```text
aoi_yolo/runs/core2_yolo11n_seg_e120_i1280_mask2_from_core3/weights/best.onnx
```

PyTorch validation on the RTX 5060 at `imgsz=1280` measured approximately:

```text
preprocess: 0.8 ms/image
inference: 8.6 ms/image
postprocess: 1.8 ms/image
```

## Expanded Data Precision Experiment

After adding the larger data batch, the current audit is:

```text
json_files: 907
pinhole: 1816
inpinhole: 1593
scratch: 543
edge: 1051
```

The expanded merged/background dataset is:

```text
aoi_yolo/dataset_core2_bg_v2
train_images: 726
val_images: 181
classes: pinhole, scratch
```

The latest expanded-data checkpoint is:

```text
aoi_yolo/runs/core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best/weights/best.pt
```

AOI-style eval at `conf=0.05`, `hit_iou=0.05`, `imgsz=1280`:

```text
pinhole recall: 96.20%, precision: 56.72%
scratch recall: 90.82%, precision: 47.85%
overall recall: 95.48%, precision: 55.41%
```

Raising confidence improves precision but loses too much recall:

```text
conf=0.10: overall recall 92.74%, precision 64.91%
conf=0.20: overall recall 88.08%, precision 76.82%
```

A sweep over simple confidence, area, and geometry filters could not keep recall around 95% while approaching 95% precision. Best recall-preserving precision was about 58%. Most pinhole false positives are isolated normal-looking specks, while many scratch false positives overlap real scratches and may be reducible with merge/postprocessing logic.

## Precision-Focused Operating Points

The evaluator supports class-specific filters:

```powershell
.\.venv_yolo\Scripts\python.exe aoi_yolo\evaluate_json_polygons.py --src "test images" --dataset "aoi_yolo\dataset_core2_bg_v2" --weights "aoi_yolo\runs\core2_bg_v2_yolo11n_seg_e120_i1280_b1_from_bg_best\weights\best.pt" --imgsz 1280 --conf 0.05 --class-conf pinhole=0.30 scratch=0.20 --class-area-min pinhole=150 scratch=300 --hit-iou 0.05 --rename inpinhole=pinhole
```

Current precision-oriented choices:

```text
Balanced:
  pinhole conf >= 0.30, scratch conf >= 0.20
  pinhole area >= 150 px, scratch area >= 300 px
  overall recall 83.56%, precision 83.22%

High-confidence:
  pinhole conf >= 0.50, scratch conf >= 0.40
  pinhole area >= 150 px, scratch area >= 300 px
  overall recall 69.59%, precision 93.21%

Precision-first:
  pinhole conf >= 0.60, scratch conf >= 0.45
  pinhole area >= 150 px, scratch area >= 300 px
  overall recall 60.27%, precision 95.86%
```

The best practical setting right now is probably the balanced preset. The precision-first preset reaches the requested precision ballpark but misses too many defects to be considered a broad inspection backend.

## Integration Target

The existing C# app currently calls HALCON detection from `Detectionui.Start_Detection()` through `DetectionWithDL2.ImageProcess(...)`.

The clean migration path is to introduce an inspection engine interface later:

```csharp
IInspectionEngine engine = new YoloInspectionEngine();
List<InspectionDefect> defects = engine.ProcessFolder(picDir);
```

Then reuse the existing coordinate mapping, database writes, UI display, and machine-control decisions.
